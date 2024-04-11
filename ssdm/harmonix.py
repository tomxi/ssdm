import numpy as np
import os, json, pkg_resources
from glob import glob

from . import base
import ssdm

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
    def __init__(self, split='train', tids=None, infer=True, 
                 sample_select_fn=ssdm.sel_samp_l, **kwargs):
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