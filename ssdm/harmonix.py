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

HMX_TITLE_DICT = dict()
title_dict_dir = '/scratch/qx244/data/audio_features_crema_mfcc_openl3_tempogram_yamnet/crema_mfcc_openl3_tempogram_yamnet_features/'
for path in glob(os.path.join(title_dict_dir, f'*mfcc.npz')):
    tid, title, feat = os.path.basename(os.path.basename(path).replace('_24', '')).split('_')
    HMX_TITLE_DICT[tid] = title

class Track(base.Track):
    def __init__(
            self,
            tid='0077', 
            feature_dir='/scratch/qx244/data/audio_features_crema_mfcc_openl3_tempogram_yamnet/crema_mfcc_openl3_tempogram_yamnet_features/',
            dataset_dir='/home/qx244/harmonixset/dataset/',
            output_dir='/vast/qx244/harmonix2/',
                ):
        super().__init__(tid=tid, feature_dir=feature_dir, output_dir=output_dir, dataset_dir=dataset_dir)
        self.title = self.tid + '_' + HMX_TITLE_DICT[self.tid]

    def audio(self, **kwargs): 
        print('Audio not Available')
        raise NotImplementedError


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

class NewDS(base.DS):
    def __init__(self, mode='rep', infer=False, 
                 split='train', tids=None, transform=None, lap_norm='random_walk',
                 sample_select_fn=ssdm.select_samples_using_tau_percentile):
        self.name = 'hmx'

        if tids is None:
            self.tids = get_ids(split=split, out_type='list')
            self.split = split
        else:
            self.tids = tids
            self.split = f'custom{len(tids)}'
        
        super().__init__(mode=mode, infer=infer, lap_norm=lap_norm, sample_select_fn=sample_select_fn, transform=transform)
    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)
    

class DS(NewDS):
    def __init__(self, mode='rep', infer=False, 
                 split='train', tids=None, transform=None, lap_norm='random_walk',
                 sample_select_fn=ssdm.select_samples_using_tau_percentile):
        super().__init__(self, mode=mode, infer=infer, 
                 split=split, tids=tids, transform=transform, lap_norm=lap_norm,
                 sample_select_fn=sample_select_fn)