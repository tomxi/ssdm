import numpy as np
import os, json, itertools, pkg_resources
from glob import glob

from . import base
import ssdm

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
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    with open(id_path, 'r') as f:
        id_json = json.load(f)
    ids = id_json['harmonix']
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


def get_taus(
    tids=[], 
    **tau_kwargs,
) -> xr.DataArray:
    tau_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        track_tau = track.tau(**tau_kwargs)
        tau_per_track.append(track_tau)
    
    return xr.concat(tau_per_track, pd.Index(tids, name='tid')).rename()


def get_lsd_scores(
    tids=[], 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        track_score = track.lsd_score(**lsd_score_kwargs)
        score_per_track.append(track_score)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


class DS(Dataset):
    """ 
    mode='rep', # {'rep', 'loc'}
    """
    def __init__(self, mode='rep'):
        if mode not in ('rep', 'loc'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.tids = ssdm.get_ids('harmonix', out_type='list')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, feat = self.samples[idx]
        track = Track(tid)

        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        if self.mode == 'rep':
            rep_ssm = track.ssm(feature=feat, 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feat]
                                )

            return {'data': torch.tensor(rep_ssm[None, None, :], dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
        
        elif self.mode == 'loc':
            path_sim = track.path_sim(feature=feat, 
                                      distance=config['loc_metric'],
                                      **ssdm.LOC_FEAT_CONFIG[feat])

            return {'data': torch.tensor(path_sim[None, None, :], dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
       
        else:
            assert KeyError
