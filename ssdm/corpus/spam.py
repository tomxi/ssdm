from .. import base
import ssdm
import json
import numpy as np

import os, jams
import pandas as pd

AVAL_FEAT_TYPES = ['mfcc', 'pcp', 'cqt', 'tempogram']

class Track(base.Track):
    def __init__(
        self,
        tid: str = '0', # 0-49
        dataset_dir: str = '/scratch/qx244/msaf-data/SPAM', 
        output_dir: str = '/vast/qx244/spam/',
        feature_dir: str = '/scratch/qx244/msaf-data/SPAM/features/',
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)
        # Track IDs for JSD looks like this
        metadata = pd.read_csv(os.path.join(dataset_dir, 'metadata.tsv'), delimiter='\t')

        self.info = metadata.loc[metadata.id==int(tid)]
        self.title = self.info['title'].item()
        self.file_name = self.info['File Name'].item().replace('.mp3', '').replace('/', ':')
        self.ds_name = 'spam'

        for directory in [output_dir, feature_dir]:
            if not os.path.exists(directory):
                os.system(f'mkdir {directory}')

    def jam(self) -> jams.JAMS:
        if self._jam is None:
            jams_path = os.path.join(self.dataset_dir, f'references/{self.file_name}.jams')
            self._jam = jams.load(jams_path, validate=False)
        return self._jam
    
    def audio(
        self, 
        sr: float = 22050
    ) -> None:
        print('no audio available')

    def representation(self, feat_type='mfcc', beat_sync=True, **delay_emb_kwargs):
        assert beat_sync == True
        """delay_emb_kwargs: add_noise, n_steps, delay"""
        delay_emb_config = dict(add_noise=False, n_steps=1, delay=1)
        delay_emb_config.update(delay_emb_kwargs)
        feature_json_path = os.path.join(self.feature_dir, self.file_name + '.json')
        with open(feature_json_path, 'r') as f:
            song_reps = json.load(f)
        return base.delay_embed(np.array(song_reps[feat_type]['est_beatsync']).T, **delay_emb_kwargs)
    
    def ts(self, mode='beat'):
        assert mode == 'beat'
        raise NotImplementedError

def get_ids(split='full', out_type='list'):
    return [str(i) for i in range(50)]

class InferDS(base.InferDS):
    def __init__(self, split=None, transform=None):
        self.split = 'full'
        self.transform = transform
        self.ds_module = ssdm.spam
        self.name = 'spam'
        self.tids = get_ids(self.split)
        self.tids.sort()
        self.AVAL_FEAT_TYPES = AVAL_FEAT_TYPES
