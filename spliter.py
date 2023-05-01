import pandas as pd
import numpy as np
import json, os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import salami



AVAL_FEAT_TYPES = ['chroma', 'crema', 'tempogram', 'mfcc', 'yamnet', 'openl3']
DEFAULT_CONFIG = {'rec_width': 13,
                  'rec_smooth': 7,
                  'evec_smooth': 13,
                  'rep_ftype': 'chroma', # grid
                  'loc_ftype': 'mfcc', # grid
                  'rep_metric': 'cosine',
                  'hier': True,
                  'num_layers': 10}

pidx = pd.IndexSlice # for slicing ndarrays in pandas

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


def collect_score_df(track_ids):
    track_ids = list(track_ids)
    track_ids.sort()
    # POPULATE THE SCORE_DF this is saved as ./l_score_df.pkl
    
    cols = pd.MultiIndex.from_product([AVAL_FEAT_TYPES, AVAL_FEAT_TYPES], names=["rep_f", "loc_f"])
    rows = pd.MultiIndex.from_product([track_ids, ['lp', 'lr', 'l']], names=["tid", "score"])
    
    s_df = pd.DataFrame(data = None, index = rows, columns = cols, dtype = np.float)
    
    
    for i in tqdm(track_ids):  
        track = salami.Track(tid=i)
        scores_path = os.path.join(track.salami_dir, f'l_scores/{track.tid}.json')
        config = DEFAULT_CONFIG.copy()

        for rep_t in AVAL_FEAT_TYPES:
            for loc_t in AVAL_FEAT_TYPES:
                config['rep_ftype'] = rep_t
                config['loc_ftype'] = loc_t
                # Get score from l_score/tid.json
                l_mat = np.array(track.l_scores(config=config)['l_scores'])
                # read the score with max l-measure
                which_anno = l_mat.argmax(axis=0)[2] # index 2 is the l-measrue
                score = l_mat[which_anno]
                s_df.loc[pidx[i, :], pidx[rep_t, loc_t]] = score
    return s_df


def collect_rho_df(track_ids):
    track_ids = list(track_ids)
    track_ids.sort()
    # build the pd dataframe with multi tindx cols and rows
    rho_types = ('rho_rep', 'rho_loc')
    rho_key_names = {'rho_rep': 'rep_k13', 'rho_loc': 'local_k1'}
    cols = pd.MultiIndex.from_product(
        [AVAL_FEAT_TYPES, rho_types],
        names = ['feature', 'rho']
    )
    rho_df = pd.DataFrame(data = None, index = track_ids, columns = cols, dtype = np.float)

    for tid in tqdm(track_ids):
        track = salami.Track(tid = tid)
        rho_path = os.path.join(track.salami_dir, f'rhos/{track.tid}.json')
        # get rho score for feat, rho pair
        rho_dict = track.rhos()
        for feat in AVAL_FEAT_TYPES:
            for rho in rho_types:
                rho_scores = []
                # max over annotations
                for anno_idx in range(rho_dict['num_annos']):
                    rho_scores.append(
                        rho_dict[f'{feat}_cosine_{rho_key_names[rho]}_a{anno_idx}']
                    )
                rho_score = max(np.asarray(rho_scores))

                rho_df.loc[tid, (feat, rho)] = rho_score

    return rho_df

def collect_tau_df(track_ids, loc_type='lm_30'):
    track_ids = list(track_ids)
    track_ids.sort()
    # build the pd dataframe with multi tindx cols and rows
    tau_types = ('rep', 'loc')
    cols = pd.MultiIndex.from_product(
        [AVAL_FEAT_TYPES, tau_types],
        names = ['feature', 'tau']
    )
    tau_df = pd.DataFrame(data = None, index = track_ids, columns = cols, dtype = np.float)

    for tid in tqdm(track_ids):
        track = salami.Track(tid = tid)
        tau_path = os.path.join(track.salami_dir, f'taus/{track.tid}.json')
        # get tau score for feat, tau pair
        tau_dict = track.taus()
        for feat in AVAL_FEAT_TYPES:
            for tau in tau_types:
                tau_scores = []
                # max over annotations
                
                for anno_idx in range(track.num_annos):
                    if tau == 'rep':
                        tau_scores.append(
                            max(tau_dict[tau][feat])
                        )

                    elif tau == 'loc':
                        tau_scores.append(
                            max(tau_dict[tau][feat][loc_type])
                        )
                tau_score = max(np.asarray(tau_scores))

                tau_df.loc[tid, (feat, tau)] = tau_score

    return tau_df


# def generate_label_df(track_ids):
#     # read rho_df
#     rho_df = pd.read_pickle('./rho_df.pkl')
#     return rho_df

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