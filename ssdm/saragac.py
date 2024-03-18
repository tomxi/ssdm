import ssdm
from . import base
import xarray as xr

import torch
from torch.utils.data import Dataset

import os, jams, itertools
from glob import glob
import pandas as pd

import json
from tqdm import tqdm

class Track(base.Track):
    def __init__(
        self,
        tid: str = '14',
        dataset_dir: str = '/scratch/qx244/data/saraga1.5_carnatic', 
        output_dir: str = '/vast/qx244/saraga_carnatic/',
        feature_dir: str = '/vast/qx244/saraga_carnatic/features/',
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)
        self.metadata_csv = pd.read_csv(os.path.join(dataset_dir, 'file_paths.csv'))
        self.path_part = self.metadata_csv.loc[int(self.tid)].filepath.replace('.', '')
        self.glob_search_str = os.path.join(self.dataset_dir, self.path_part.replace('/', '/*').replace('t', 't*'))
        self.audio_path = glob(self.glob_search_str + '.mp3*')[0]
        self.metadata_path = glob(self.glob_search_str + '.json')[0]

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.seg_anno_path = self.metadata_path.replace('.json', '.sections-manual-p.txt')
        self.artists = self.metadata['album_artists'][0]['name']
        self.title = self.metadata['title']
        self.duration = self.metadata['length']
        self.concert = self.metadata['concert'][0]['title']


    def jam(self):
        if self._jam is None:
            j = jams.JAMS()
            j.file_metadata.duration = self.duration
            j.file_metadata.title = self.title
            j.file_metadata.artist = self.artists

            org_anno = pd.read_csv(self.seg_anno_path, delimiter='\t', header=None)
            new_anno = jams.Annotation('segment_open')

            for _, row in org_anno.iterrows():
                new_anno.append(time = row[0],
                                duration = row[2],
                                value = row[3])
                
            j.annotations.append(new_anno)
            self._jam = j
        return self._jam


def get_ids(out_type: str = 'list'):
    tids = []
    for tid in range(249):
        tr = Track(str(tid))
        if os.path.exists(tr.seg_anno_path): # there are only 119 tracks with section annotations
            if tr.duration < 60*25*1000: # Track that are longer than 25 min are discarded due to resource limit.
                tids.append(str(tid))

    single_seg = ['104', '112', '124', '138', '172', '177', '179', '21', '34', '53']
    tid_set = set(tids) - set(single_seg)
    if out_type == 'set':
        tid_set
    else:
        tids = list(tid_set)
        tids.sort()
        return tids


class DS(Dataset):
    """ 
    mode='rep', # {'rep', 'loc'}
    tids=list()
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
        return 'srgc' + self.mode

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
