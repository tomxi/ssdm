from . import base
import ssdm
import xarray as xr

import torch
from torch.utils.data import Dataset

import os, jams, itertools
import pandas as pd

from tqdm import tqdm
from glob import glob

class Track(base.Track):
    def __init__(
        self,
        tid: str = '01-01', # 'disc-track'
        dataset_dir: str = '/home/qx244/msaf-data/BeatlesTUT', 
        output_dir: str = '/vast/qx244/beatles/',
        feature_dir: str = '/vast/qx244/beatles/features/',
        audio_dir: str = '/scratch/work/sonyc/marl/private_datasets/Beatles/Audio'
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)
        disc, track = tid.split('-')
        if disc == '10':
            basename = f'10_-_The_Beatles_CD1/{track}*'
        elif disc == '11':
            basename = f'10_-_The_Beatles_CD2/{track}*'
        else:
            basename = f'{disc}*/{track}*'
        self.audio_path = glob(os.path.join(audio_dir, basename))[0]
        self.title = os.path.basename(self.audio_path)[:-4]

    def jam(self):
        if self._jam is None:
            jam_path = os.path.join(self.dataset_dir, f'references/{self.title}.jams')
            self._jam = jams.load(jam_path)
        return self._jam

       
       
def get_ids(out_type: str = 'list'):
    audio_dir = '/scratch/work/sonyc/marl/private_datasets/Beatles/Audio/'
    albums = os.listdir(audio_dir)

    tids = []
    for disc in albums:
        disc_id = disc.split('_')[0]
        if disc[-1] == '2':
            disc_id = '11'
        disc_dir = os.path.join(audio_dir, disc)
        tracks = glob(os.path.join(disc_dir, '*.wav'))
        tids += [f'{disc_id}-{tid+1:02d}' for tid in range(len(tracks))]
    tids.sort()

    valid_set = set(tids) - set(['10-05', '10-08', '11-06', '11-12', '12-09', '12-16']) # these tracks are missing annotation
    tids = list(valid_set)
    tids.sort()
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
    def __init__(self, mode='rep'):
        if mode not in ('rep', 'loc'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.tids = get_ids()
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
                    'info': ('jsd', tid, feat, self.mode),
                    }
       
        else:
            assert KeyError