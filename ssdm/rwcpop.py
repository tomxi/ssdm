import ssdm
from . import base
import xarray as xr

import torch
from torch.utils.data import Dataset

import os, jams, itertools
import pandas as pd

from tqdm import tqdm
from glob import glob

import librosa

class Track(base.Track):
    def __init__(
        self,
        tid: str = '1', #(1-100)
        dataset_dir: str = '/scratch/work/marl/datasets/mir_datasets/rwc_popular/', 
        output_dir: str = '/vast/qx244/rwc_pop/',
        feature_dir: str = '/vast/qx244/rwc_pop/features/',
        audio_dir: str = '/scratch/work/marl/datasets/mir_datasets/rwc_popular/audio'
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)

        metadata_csv = pd.read_csv(os.path.join(dataset_dir, 'metadata-master/rwc-p.csv'))
        self.metadata = metadata_csv[metadata_csv['Piece No.'] == f'No. {tid}']

        self.title = self.metadata['Title'].item()
        self.audio_path = os.path.join(audio_dir, f'{tid}.wav')


    def jam(self):
        if self._jam is None:
            # create jams file from metadata 
            j = jams.JAMS()
            j.file_metadata.duration = librosa.get_duration(path=self.audio_path)
            j.file_metadata.title = self.metadata['Title'].item()
            j.file_metadata.artist = self.metadata['Artist (Vocal)'].item()


            anno_path = os.path.join(self.dataset_dir, f'annotations/AIST.RWC-MDB-P-2001.CHORUS/RM-P{int(self.tid):03d}.CHORUS.TXT')
            org_anno = pd.read_csv(anno_path, delimiter='\t', header=None, usecols=[0,1,2])
            new_anno = jams.Annotation('segment_open')

            for _, row in org_anno.iterrows():
                new_anno.append(time = row[0] / 100,
                                duration = (row[1] - row[0]) / 100,
                                value = row[2])
                
            j.annotations.append(new_anno)
            self._jam = j
        return self._jam


def get_ids(out_type: str = 'list'):
    tids = [str(tid) for tid in range(1, 101)]

    if out_type == 'set':
        return set(tids)
    else:
        return tids
    

def get_lsd_scores(
    tids=None, 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    if tids == None:
        tids = get_ids()
    for tid in tqdm(tids):
        track = Track(tid)
        score_per_track.append(track.lsd_score(**lsd_score_kwargs))
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()
    

def get_taus(
    tids=[], 
    **tau_kwargs,
) -> xr.DataArray:
    tau_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        tau_per_track.append(track.tau(**tau_kwargs))
    
    return xr.concat(tau_per_track, pd.Index(tids, name='tid')).rename()


class NewDS(base.DS):
    def __init__(self, tids=None, **kwargs):
        self.name = 'rwcp'
        if not tids:
            self.tids=get_ids(out_type='list')
            self.split=''
        else:
            self.tids=tids
            self.split=f'custom{len(tids)}'
        
        super().__init__(infer=True, **kwargs)

    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)
