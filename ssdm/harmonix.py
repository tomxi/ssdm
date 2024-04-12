import numpy as np
import os, json, pkg_resources
from glob import glob

from . import base


class Track(base.Track):
    def __init__(
            self,
            tid='0077', 
            dataset_dir: str = '/Users/tomxi/data/harmonix', 
            output_dir: str = '/Users/tomxi/data/harmonix',
            feature_dir: str = '/Users/tomxi/data/harmonix'
                ):
        super().__init__(tid=tid, feature_dir=feature_dir, output_dir=output_dir, dataset_dir=dataset_dir)
        self.title = self.tid + '_'

    def audio(self, **kwargs): 
        print('Audio not Available')
        raise NotImplementedError
    
    def _madmom_beats(self, **kwargs):
        save_path = os.path.join(self.dataset_dir.replace('dataset', 'results'), f'beats/Korzeniowski/{self.title}.txt')
        with open(save_path, 'r') as file:
            beat_times_str = file.readlines()
        beat_times = np.array([float(beat_time.strip()) for beat_time in beat_times_str])
        if beat_times[0] > 0:
            beat_times = np.insert(beat_times, 0, 0)
        if beat_times[-1] < self.ts(mode='frame')[-1]:
            beat_times = np.append(beat_times, self.ts(mode='frame')[-1])
        return beat_times


def get_ids(
    split: str = None,
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    id_path = '/Users/tomxi/ssdm/ssdm/splits.json'
    with open(id_path, 'r') as f:
        id_json = json.load(f)
    ids = id_json[f'hmx_{split}']
    ids.sort()
    return ids

def get_samps_tau(split):
    with open('/Users/tomxi/ssdm/ssdm/labels.json', 'r') as f:
        labels = json.load(f)

    for k in labels:
        labels[k] = {
            tuple(k.replace('(', '').replace(')', '').replace("'", '').split(', ')): value for k, value in labels[k].items()
        }

    return labels[f'hmx_{split}_labels']


def get_samps_score(split):
    with open('/Users/tomxi/ssdm/ssdm/score_based_labels.json', 'r') as f:
        labels = json.load(f)

    for k in labels:
        labels[k] = {
            tuple(k.replace('(', '').replace(')', '').replace("'", '').split(', ')): value for k, value in labels[k].items()
        }

    return labels[f'hmx_{split}_labels']


class NewDS(base.DS):
    def __init__(self, split='train', tids=None, infer=True, 
                 sample_select_fn=get_samps_tau, **kwargs):
        self.name = 'hmx'

        if tids is None:
            self.tids = get_ids(split=split, out_type='list')
            self.split = split
        else:
            self.tids = tids
            self.split = f'custom{len(tids)}'
        
        super().__init__(infer=infer, sample_select_fn=sample_select_fn, **kwargs)
    
    
    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)
    
