import pkg_resources
from functools import reduce
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from librosa import ParameterError
from tqdm import tqdm
import numpy as np

import ssdm

AVAL_FEAT_TYPES = ['chroma', 'crema', 'tempogram', 'mfcc', 'yamnet', 'openl3']
DEFAULT_LSD_CONFIG = {
    'rec_width': 13,
    'rec_smooth': 7,
    'evec_smooth': 13,
    'rep_ftype': 'chroma', # grid
    'loc_ftype': 'mfcc', # grid
    'rep_metric': 'cosine',
    'hier': True,
    'num_layers': 10
}

REPRESENTATION_KWARGS = {
    'chroma': {'add_noise': True, 'time_delay_emb': True},
    'crema': {'add_noise': True, 'time_delay_emb': True},
    'tempogram': {'add_noise': True, 'time_delay_emb': False},
    'mfcc': {'add_noise': True, 'time_delay_emb': True},
    'yamnet': {'add_noise': True, 'time_delay_emb': False},
    'openl3': {'add_noise': True, 'time_delay_emb': False},
}


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
                    tau_rep_pick = 'openl3' #TODO
                    tau_loc_pick = 'mfcc' # TODO
                    best_scores.append(track_score.loc[tau_rep_pick, tau_loc_pick])
                elif heuristic == 'lsd_tau_hat_pick':
                    tau_hat_rep_pick = 'openl3' #TODO
                    tau_hat_loc_pick = 'mfcc' # TODO
                    best_scores.append(track_score.loc[tau_hat_rep_pick, tau_hat_loc_pick])
                index.append(f'{tid}:{anno_id}')
        return pd.Series(best_scores, index=index)
    
    else:
        raise ParameterError('bad heuristic')


# def score_comparison_df():
#     l_score_path = pkg_resources.resource_filename('ssdm', 'l_score_df.pkl')
#     l_score = pd.read_pickle(l_score_path)

#     tau_df = pd.read_pickle('./tau_df.pkl')
#     tau_loc = tau_df.loc[:, (slice(None), ['loc'])]
#     tau_rep = tau_df.loc[:, (slice(None), ['rep'])]

#     taus_pick_loc = tau_loc.idxmax(axis=1)
#     taus_pick_rep = tau_rep.idxmax(axis=1)

#     lr_score = l_score.loc[(slice(None), ['lr']), :]
#     lr_score.index = lr_score.index.droplevel(1)

#     heuristics = [
#         'Best Avg Pair',
#         'Loc Pick',
#         'Rep Pick',
#         'Both Pick',
#         'Oracle'
#     ]

#     compare_scores_df = pd.DataFrame(index=lr_score.index, columns=heuristics)

#     for tid in compare_scores_df.index:
#         loc_pick = taus_pick_loc.loc[tid][0]
#         rep_pick = taus_pick_rep.loc[tid][0]
        
#         compare_scores_df.loc[tid]['Best Avg Pair'] =  lr_score.loc[tid]['openl3', 'mfcc']
#         compare_scores_df.loc[tid]['Loc Pick'] =  lr_score.loc[tid]['openl3', loc_pick]
#         compare_scores_df.loc[tid]['Rep Pick'] =  lr_score.loc[tid][rep_pick, 'mfcc']
#         compare_scores_df.loc[tid]['Both Pick'] =  lr_score.loc[tid][rep_pick, loc_pick]
        
#     oracle_pick = lr_score.idxmax(axis=1)
#     compare_scores_df['Oracle'] = lr_score.max(axis=1)
#     return compare_scores_df


# # DEPRE
# def get_l_df(l_type = 'lr'):
#     l_score_path = pkg_resources.resource_filename('ssdm', 'l_score_df.pkl')
#     l_score = pd.read_pickle(l_score_path)

#     l_df = l_score.loc[(slice(None), l_type), slice(None)]
#     l_df.index = l_df.index.droplevel(1)
#     return l_df

# def get_adobe_l_df():
#     l_score_path = pkg_resources.resource_filename('ssdm', 'l_score_justin_df.pkl')
#     l_score = pd.read_pickle(l_score_path)

#     return l_score