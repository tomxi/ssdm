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
        tid: str = '1', #(1-61)
        dataset_dir: str = '/scratch/work/marl/datasets/mir_datasets/rwc_classical/', 
        output_dir: str = '/vast/qx244/rwc_classical/',
        feature_dir: str = '/vast/qx244/rwc_classical/features/',
        audio_dir: str = '/scratch/work/marl/datasets/mir_datasets/rwc_classical/audio'
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)

        self.metadata_csv = pd.read_csv(os.path.join(dataset_dir, 'metadata-master/rwc-c.csv'))
        self.metadata = self.metadata_csv.loc[int(tid) - 1]

        self.title = self.metadata['Title'].replace('/', '|')
        self.disc = self.metadata['Cat. Suffix'][-2:]
        self.track_num = self.metadata['Tr. No.'][-2:]
        self.audio_path = os.path.join(audio_dir, f'rwc-c-m{int(self.disc):02d}/{int(self.track_num)}.wav')
        


    def jam(self):
        if self._jam is None:
            # create jams file from metadata 
            j = jams.JAMS()
            j.file_metadata.duration = librosa.get_duration(path=self.audio_path)
            j.file_metadata.title = self.title
            j.file_metadata.artist = self.metadata['Composer']

            anno_file_list = glob(os.path.join(self.dataset_dir, f'annotations/AIST.RWC-MDB-C-2001.CHORUS/RM-C*'))
            anno_file_list.sort()

            anno_path = anno_file_list[int(self.tid) - 1]
            org_anno = pd.read_csv(anno_path, delimiter='\t', header=None, usecols=[0,1,2])
            new_anno = jams.Annotation('segment_open')

            nothing_counter = 0
            for idx, row in org_anno.iterrows():
                if row[2] == 'nothing': # labels with nothing should be different segments
                    segment_label = 'nothing' + str(nothing_counter)
                    nothing_counter += 1
                else:
                    segment_label = row[2]

                new_anno.append(time = row[0] / 100,
                                duration = (row[1] - row[0]) / 100,
                                value = segment_label)
                
            j.annotations.append(new_anno)
            self._jam = j
        return self._jam


def get_ids(out_type: str = 'list'):
    tids = [str(tid) for tid in range(1, 62) if tid not in (31, 34)] # empty annotatinos
    if out_type == 'set':
        return set(tids)
    else:
        return tids

def get_lsd_scores(
    tids=[], 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
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


class DS(Dataset):
    """ 
    mode='rep', # {'rep', 'loc'}
    """
    def __init__(self, mode='rep', tids=None):
        if mode not in ('rep', 'loc'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        if tids == None:
            self.tids = get_ids()
        else:
            self.tids=tids
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))
        self.samples.sort()


    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)

    def __repr__(self):
        return 'rwcc' + self.mode

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
