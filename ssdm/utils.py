import pkg_resources
from functools import reduce
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from librosa import ParameterError
from tqdm import tqdm
import numpy as np

import ssdm

def get_ids(
    split: str = 'dev',
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    """ split can be ['audio', 'jams', 'excluded', 'new_val', 'new_test', 'new_train']
        Dicts sotred in id_path json file.
    """
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    with open(id_path, 'r') as f:
        id_json = json.load(f)
        
    ids = id_json[split]
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


# add new splits to split_ids.json
def update_split_json(split_name='', split_idx=[]):
    # add new splits to split_id.json file at json_path
    # read from json and get dict
    json_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    with open(json_path, 'r') as f:
        split_dict = json.load(f)
    
    # add new split to dict
    split_dict[split_name] = split_idx
    
    # save json again
    with open(json_path, 'w') as f:
        json.dump(split_dict, f)
        
    with open(json_path, 'r') as f:
        return json.load(f)

    
def create_splits(arr, val_ratio=0.15, test_ratio=0.15, random_state=20230327):
    dev_set, test_set = train_test_split(arr, test_size = test_ratio, random_state = random_state)
    train_set, val_set = train_test_split(dev_set, test_size = val_ratio / (1 - test_ratio), random_state = random_state)
    return train_set, val_set, test_set


# Collaters:
def collate_l_score(
    heuristic: str = 'all_lsd_pairs',
    l_type: str = 'l',
    anno_mode: str = 'expand',
    salami_split: str = 'working',
) -> pd.Series:
    """
    heuristic in {'all_lsd_pairs', 'adobe_oracle', 'lsd_best_pair', 'lsd_adaptive_oracle', 'lsd_tau_pick', 'lsd_tau_hat_pick'}
    """
    if heuristic == 'all_lsd_pairs':
        # get average for all pairs by lsd_l
        corpus_score_mats = []
        index = []
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            track_score_mats = []
            for anno_id in range(track.num_annos()):
                track_score_mats.append(track.lsd_l(anno_mode=anno_mode, l_type=l_type, anno_id=anno_id))
                index.append(f'{tid}:{anno_id}')
            corpus_score_mats += track_score_mats
        return pd.Series(corpus_score_mats, index=index)
    
    elif heuristic == 'adobe_oracle':
        scores = []
        index = []
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            for anno_id in range(track.num_annos()):
                scores.append(track.adobe_l(anno_mode=anno_mode, l_type=l_type, anno_id=anno_id))
                index.append(f'{tid}:{anno_id}')
        return pd.Series(scores, index=index)
    
    elif heuristic in ['lsd_best_pair', 'lsd_adaptive_oracle', 'lsd_tau_pick', 'lsd_tau_hat_pick']:
        best_scores = []
        index = []
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            for anno_id in range(track.num_annos()):
                track_score = track.lsd_l(anno_mode=anno_mode, l_type=l_type, anno_id=anno_id)
                if heuristic == 'lsd_best_pair':
                    best_scores.append(track_score.loc['openl3', 'mfcc'])
                elif heuristic == 'lsd_adaptive_oracle':
                    best_scores.append(track_score.max().max())
                elif heuristic == 'lsd_tau_pick':
                    tau_rep_pick = track.tau(anno_id=anno_id).idxmax(axis=0)[f'full_{anno_mode}']
                    tau_loc_pick = track.tau(anno_id=anno_id).idxmax(axis=0)[f'path_{anno_mode}']
                    best_scores.append(track_score.loc[tau_rep_pick, tau_loc_pick])
                elif heuristic == 'lsd_tau_hat_pick':
                    tau_hat_rep_pick = 'openl3' #TODO
                    tau_hat_loc_pick = 'mfcc' # TODO
                    best_scores.append(track_score.loc[tau_hat_rep_pick, tau_hat_loc_pick])
                index.append(f'{tid}:{anno_id}')
        return pd.Series(best_scores, index=index)
    
    else:
        raise ParameterError('bad heuristic')
    

def collate_tau(
    anno_mode: str = 'expand',
    tau_type: str = 'rep', # or 'loc'
    salami_split: str = 'working',
) -> pd.Series:
    """
    """
    tau_df = pd.DataFrame(columns=AVAL_FEAT_TYPES, dtype='float')
    for tid in tqdm(get_ids(split=salami_split, out_type='list')):
        track = ssdm.Track(tid=tid)
        for anno_id in range(track.num_annos()):
            track_tau = track.tau(anno_id=anno_id)
            if tau_type == 'rep':
                tau_df.loc[f'{tid}:{anno_id}'] = track_tau[f'full_{anno_mode}']
            elif tau_type == 'loc':
                tau_df.loc[f'{tid}:{anno_id}'] = track_tau[f'path_{anno_mode}']
    return tau_df.astype('float')


def lucky_track(tids=get_ids(split='working'), announce=True):
    """Randomly pick a track"""
    rand_idx = int(np.floor(np.random.rand() * len(tids)))
    tid = tids[rand_idx]
    if announce:
        print(f'track {tid} is the lucky track!')
    return ssdm.Track(tid)

