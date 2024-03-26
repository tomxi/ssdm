import numpy as np
import os, json, itertools, pkg_resources
from glob import glob

from . import base
import ssdm
from ssdm import scluster

import torch
from torch.utils.data import Dataset

import xarray as xr
from tqdm import tqdm
import pandas as pd
# import jams

class Track(base.Track):
    def __init__(
            self,
            tid='0077', 
            feature_dir='/scratch/qx244/data/audio_features_crema_mfcc_openl3_tempogram_yamnet/crema_mfcc_openl3_tempogram_yamnet_features/',
            dataset_dir='/home/qx244/harmonixset/dataset/',
            output_dir='/vast/qx244/harmonix2/',
                ):
        super().__init__(tid=tid, feature_dir=feature_dir, output_dir=output_dir, dataset_dir=dataset_dir)

        basename = os.path.basename(glob(os.path.join(self.feature_dir, f'{self.tid}*mfcc.npz'))[0])
        self.title = '_'.join(basename.split('_')[:2])

    def audio(self, **kwargs):
        print('Audio not Available')
        return None


def get_ids(
    split: str = None,
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    with open(id_path, 'r') as f:
        id_json = json.load(f)
    all_ids = id_json['harmonix']
    
    if split:
    # Get different splits: can be train test val
        split_dict = ssdm.create_splits(all_ids, val_ratio=0.15, test_ratio=0.15, random_state=20230327)
        ids = split_dict[split]
    else:
        ids = all_ids

    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


class DS(Dataset):
    """ 
    mode='rep', # {'rep', 'loc', 'both}
    """
    def __init__(self, mode='rep', infer=True, tids=None, split=None, transform=None, sample_select_fn=ssdm.utils.select_samples_using_outstanding_l_score):
        if mode not in ('rep', 'loc', 'both'):
            raise AssertionError('bad dataset mode, can only be rep or loc both')
        self.mode = mode
        if tids is None:
            self.tids = get_ids(split=split, out_type='list')
            self.split = split
        else:
            self.tids = tids
            self.split = f'custom{len(tids)}'
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.infer=infer

        # sample building
        if self.infer:
            if self.mode == 'both':
                self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES, ssdm.AVAL_FEAT_TYPES))
            else:
                self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))
        else:
            self.labels = sample_select_fn(self)
            self.samples = list(self.labels.keys())
        self.samples.sort()
        self.transform=transform


    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)


    def __len__(self):
        return len(self.samples)


    def __repr__(self):
        if self.split:
            return f'hmx{self.mode}{self.split}'
        else:
            return f'hmx{self.mode}{len(self.tids)}'


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, *feats = self.samples[idx]
        track = Track(tid)

        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        s_info = (tid, *feats, self.mode)

        if self.mode == 'rep':
            data = track.ssm(feature=feats[0], 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feats[0]]
                                )
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
    
        elif self.mode == 'loc':
            data = track.path_sim(feature=feats[0], 
                                  distance=config['loc_metric'],
                                  **ssdm.LOC_FEAT_CONFIG[feats[0]])
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        elif self.mode == 'both':
            save_path = os.path.join(self.track_obj().output_dir, 'evecs/'+'_'.join(s_info)+'.pt')
            # Try to see if it's already calculated. if so load:
            try:
                first_evecs = torch.load(save_path) # load
            # else: calculate
            except:
                # print(save_path)
                lsd_config = dict(rep_ftype=feats[0], loc_ftype=feats[1])
                rec_mat = torch.tensor(track.combined_rec_mat(config_update=lsd_config), dtype=torch.float32, device=self.device)
                # compute normalized laplacian
                with torch.no_grad():
                    rec_mat += 1e-30 # make inverses nice...
                    # Compute the degree matrix
                    degree_matrix = torch.diag(torch.sum(rec_mat, dim=1))
                    unnormalized_laplacian = degree_matrix - rec_mat
                    # Compute the normalized Laplacian matrix
                    degree_inv = torch.inverse(degree_matrix)
                    normalized_laplacian = degree_inv @ unnormalized_laplacian

                    evals, evecs = torch.linalg.eig(normalized_laplacian)
                    first_evecs = evecs.real[:, :20]
                    torch.save(first_evecs, save_path)
            data = first_evecs.to(torch.float32).to(self.device)
        else:
            assert KeyError('bad mode: can onpy be rep or loc or both')
        
        datum = {'data': data[None, None, :],
                 'info': s_info,
                 'uniq_segs': torch.tensor([track.num_dist_segs() - 1], dtype=torch.long, device=self.device)}

        if not self.infer:
            datum['label'] = torch.tensor([self.labels[self.samples[idx]]], dtype=torch.float32, device=self.device)[None, :]
        
        if self.transform:
            datum = self.transform(datum)
        
        return datum