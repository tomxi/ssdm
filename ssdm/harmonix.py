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

        if self.mode == 'rep':
            data = track.ssm(feature=feats[0], 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feats[0]]
                                )
    
        elif self.mode == 'loc':
            data = track.path_sim(feature=feats[0], 
                                  distance=config['loc_metric'],
                                  **ssdm.LOC_FEAT_CONFIG[feats[0]])

        elif self.mode == 'both':
            # repd = track.ssm(feature=feats[0], 
            #                  distance=config['rep_metric'],
            #                  width=config['rec_width'],
            #                  full=config['rec_full'],
            #                  **ssdm.REP_FEAT_CONFIG[feats[0]]
            #                  )
            # locd = track.path_sim(feature=feats[1], 
            #                       distance=config['loc_metric'],
            #                       **ssdm.LOC_FEAT_CONFIG[feats[1]])
            # data = scluster.combine_ssms(repd, locd, rec_smooth=config['rec_smooth'])
            data = track.combined_rec_mat(config_update=config)

        else:
            assert KeyError('bad mode: can onpy be rep or loc or both')
        
        datum = {'data': torch.tensor(data[None, None, :], dtype=torch.float32, device=self.device),
                 'info': (tid, *feats, self.mode)}

        if not self.infer:
            datum['label'] = torch.tensor([self.labels[self.samples[idx]]], dtype=torch.float32)[None, :]
        
        if self.transform:
            datum = self.transform(datum)
        
        return datum