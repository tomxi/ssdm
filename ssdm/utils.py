import pandas as pd
import numpy as np
import json, os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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


# add new splits to split_ids.json
def update_split_json(split_name='', split_idx=[], json_path='/home/qx244/scanning-ssm/revive/split_ids.json'):
    # add new splits to split_id.json file at json_path
    # read from json and get dict
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


def score_comparison_df():
    l_score = pd.read_pickle('./l_score_df.pkl')

    tau_df = pd.read_pickle('./tau_df.pkl')
    tau_loc = tau_df.loc[:, (slice(None), ['loc'])]
    tau_rep = tau_df.loc[:, (slice(None), ['rep'])]

    taus_pick_loc = tau_loc.idxmax(axis=1)
    taus_pick_rep = tau_rep.idxmax(axis=1)

    lr_score = l_score.loc[(slice(None), ['lr']), :]
    lr_score.index = lr_score.index.droplevel(1)

    heuristics = [
        'Best Avg Pair',
        'Loc Pick',
        'Rep Pick',
        'Both Pick',
        'Oracle'
    ]

    compare_scores_df = pd.DataFrame(index=lr_score.index, columns=heuristics)

    for tid in compare_scores_df.index:
        loc_pick = taus_pick_loc.loc[tid][0]
        rep_pick = taus_pick_rep.loc[tid][0]
        
        compare_scores_df.loc[tid]['Best Avg Pair'] =  lr_score.loc[tid]['openl3', 'mfcc']
        compare_scores_df.loc[tid]['Loc Pick'] =  lr_score.loc[tid]['openl3', loc_pick]
        compare_scores_df.loc[tid]['Rep Pick'] =  lr_score.loc[tid][rep_pick, 'mfcc']
        compare_scores_df.loc[tid]['Both Pick'] =  lr_score.loc[tid][rep_pick, loc_pick]
        
    oracle_pick = lr_score.idxmax(axis=1)
    compare_scores_df['Oracle'] = lr_score.max(axis=1)
    return compare_scores_df

def get_l_df(l_type = 'lr'):
    l_score = pd.read_pickle('./l_score_df.pkl')
    l_df = l_score.loc[(slice(None), l_type), slice(None)]
    l_df.index = l_df.index.droplevel(1)
    return l_df