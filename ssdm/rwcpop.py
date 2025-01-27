import ssdm
import ssdm.rwcpop
from . import base
import xarray as xr

import torch
from torch.utils.data import Dataset

import os, jams, itertools
import pandas as pd

from tqdm import tqdm
from glob import glob

import librosa
NAME = 'rwcpop'
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
        self.ds_name = 'rwc'

        metadata_csv = pd.read_csv(os.path.join(dataset_dir, 'metadata-master/rwc-p.csv'))
        self.metadata = metadata_csv[metadata_csv['Piece No.'] == f'No. {tid}']
        self.title = self.metadata['Title'].item()
        self.audio_path = os.path.join(audio_dir, f'{tid}.wav')


    def jam(self):
        if self._jam is None:
            # create jams file from metadata 
            j = jams.JAMS()
            j.file_metadata.duration = librosa.get_duration(path=self.audio_path)
            j.file_metadata.title = self.title
            j.file_metadata.artist = self.metadata['Artist (Vocal)'].item()


            anno_path = os.path.join(self.dataset_dir, f'annotations/AIST.RWC-MDB-P-2001.CHORUS/RM-P{int(self.tid):03d}.CHORUS.TXT')
            org_anno = pd.read_csv(anno_path, delimiter='\t', header=None, usecols=[0,1,2])
            new_anno = jams.Annotation('segment_open')

            for _, row in org_anno.iterrows():
                new_anno.append(time = row[0] / 100,
                                duration = (row[1] - row[0]) / 100,
                                value = row[2])
            # return new_anno
            j.annotations.append(new_anno)
            # return j
            self._jam = j
        return self._jam


def get_ids(split: str = None, out_type: str = 'list'):
    all_ids = [str(tid) for tid in range(1, 101)]

    if split == 'single':
        tids = ['65']
    elif split == 'dev':
        tids = ['47', '37', '65']
    elif split:
    # Get different splits: can be train test val
        split_dict = ssdm.create_splits(all_ids, val_ratio=0.15, test_ratio=0.15, random_state=20230327)
        tids = split_dict[split]
    else:
        tids = all_ids

    if out_type == 'set':
        return set(tids)
    else:
        tids.sort()
        return tids
    

class PairDS(base.PairDS):
    def __init__(self, split='val', transform=None, perf_margin=0.05):
        super().__init__(ds_module=ssdm.rwcpop, name='rwcpop', split=split, transform=transform, perf_margin=perf_margin)


class InferDS(base.InferDS):
    def __init__(self, **kwargs):
        super().__init__(ds_module=ssdm.rwcpop, name='rwcpop', **kwargs)
