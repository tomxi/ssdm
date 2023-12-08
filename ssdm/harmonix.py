import numpy as np
import os, json
from glob import glob

from . import base

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