import pkg_resources
from functools import reduce
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
from sklearn import cluster
from tqdm import tqdm

import jams
import mir_eval
from librosa import ParameterError

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


def lucky_track(tids=get_ids(split='working'), announce=True):
    """Randomly pick a track from tids"""
    rand_idx = int(np.floor(np.random.rand() * len(tids)))
    tid = tids[rand_idx]
    if announce:
        print(f'track {tid} is the lucky track!')
    return ssdm.Track(tid)


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
    salami_split: str = 'working',
    anno_merge_fn: any = None,
) -> pd.Series:
    """
    heuristic in {'all_lsd_pairs', 'adobe_oracle', 'lsd_best_pair', 'lsd_adaptive_oracle', 'lsd_tau_pick', 'lsd_tau_hat_pick'}
    returns a series of dataframes
    """
    if heuristic == 'all_lsd_pairs':
        # get average for all pairs by lsd_l
        scores = []
        index = []
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            anno_scores = dict()
            for anno_id in range(track.num_annos()):
                anno_scores[str(anno_id)] = track.lsd_l(anno_mode='expand', l_type=l_type, anno_id=anno_id)

            if anno_merge_fn is None:
                index += [tid+f':{a}' for a in anno_scores.keys()]
                scores += [anno_scores[a] for a in anno_scores.keys()]
            else:
                index.append(tid)
                anno_scores = [anno_scores[a] for a in anno_scores.keys()]
                anno_scores_out = pd.DataFrame(data=anno_merge_fn(anno_scores, axis=0),
                                               index=anno_scores[0].index, 
                                               columns=anno_scores[0].columns,
                                               )
                scores.append(anno_scores_out)
            
        return pd.Series(scores, index=index)
    
    elif heuristic == 'adobe_oracle':
        scores = []
        index = []
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            anno_scores = dict()
            for anno_id in range(track.num_annos()):
                anno_scores[str(anno_id)] = track.adobe_l(anno_mode='expand', l_type=l_type, anno_id=anno_id)

            if anno_merge_fn is None:
                index += [tid+f':{a}' for a in anno_scores.keys()]
                scores += [anno_scores[a] for a in anno_scores.keys()]
            else:
                index.append(tid)
                anno_scores = [anno_scores[a] for a in anno_scores.keys()]
                scores.append(anno_merge_fn(anno_scores, axis=0))

        return pd.Series(scores, index=index)
    
    # All kinds of heuristics for picking feature combinations for LSD
    elif heuristic.split('_')[0] == 'lsd':
        scores = []
        index = []
        rep_heur, loc_heur = heuristic.split('_')[1:3]
        for tid in tqdm(get_ids(split=salami_split, out_type='list')):
            track = ssdm.Track(tid=tid)
            anno_scores = dict()
            for anno_id in range(track.num_annos()):
                anno_l_scores = track.lsd_l(anno_mode='expand', l_type=l_type, anno_id=anno_id)
                if rep_heur in ssdm.AVAL_FEAT_TYPES:
                    rep_pick = rep_heur
                elif rep_heur == 'tau':
                    rep_pick = track.tau(anno_id=anno_id)['full_expand'].idxmax()
                elif rep_heur == 'tauhat':
                    pass

                if loc_heur in ssdm.AVAL_FEAT_TYPES:
                    loc_pick = loc_heur
                elif loc_heur == 'tau':
                    loc_pick = track.tau(anno_id=anno_id)['path_expand'].idxmax()
                elif loc_heur == 'tauhat':
                    pass

                if rep_heur == 'orc' and loc_heur == 'orc':
                    rep_pick = anno_l_scores.max(axis=1).idxmax()
                    loc_pick = anno_l_scores.max(axis=0).idxmax()
                elif rep_heur == 'orc':
                    rep_pick = anno_l_scores[loc_pick].idxmax()
                elif loc_heur == 'orc':
                    loc_pick = anno_l_scores.loc[rep_pick].idxmax()
                
                heur_score = anno_l_scores.loc[rep_pick, loc_pick]
                anno_scores[str(anno_id)] = heur_score

            if anno_merge_fn is None:
                index += [tid+f':{a}' for a in anno_scores]
                scores += [anno_scores[a] for a in anno_scores]
            else:
                index.append(tid)
                anno_scores = [anno_scores[a] for a in anno_scores]
                scores.append(anno_merge_fn(anno_scores, axis=0))

        return pd.Series(scores, index=index)
    
    else:
        raise ParameterError('bad heuristic')
    

def collate_tau(
    salami_split: str = 'working',
    anno_merge_fn: any = None,
) -> tuple:
    """
    """
    tau_rep_df = pd.DataFrame(columns=ssdm.AVAL_FEAT_TYPES, dtype='float')
    tau_loc_df = pd.DataFrame(columns=ssdm.AVAL_FEAT_TYPES, dtype='float')
    for tid in tqdm(get_ids(split=salami_split, out_type='list')):
        track = ssdm.Track(tid=tid)
        anno_tau_reps = []
        anno_tau_locs = []
        for anno_id in range(track.num_annos()):
            anno_tau_rep = track.tau(anno_id=anno_id)[f'full_expand']
            anno_tau_loc = track.tau(anno_id=anno_id)[f'path_expand']

            if anno_merge_fn is None:
                tau_rep_df.loc[f'{tid}:{anno_id}'] = anno_tau_rep
                tau_loc_df.loc[f'{tid}:{anno_id}'] = anno_tau_loc
            else:
                anno_tau_reps.append(anno_tau_rep)
                anno_tau_locs.append(anno_tau_loc)
        # return anno_tau_reps

        if anno_merge_fn is not None:
            tau_rep_df.loc[f'{tid}'] = anno_merge_fn(anno_tau_reps, axis=0)
            tau_rep_df.columns.name='tau_rep'
            tau_loc_df.loc[f'{tid}'] = anno_merge_fn(anno_tau_locs, axis=0)
            tau_loc_df.columns.name='tau_loc'
    return tau_rep_df.astype('float'), tau_loc_df.astype('float')


### Stand alone functions from Salami.py
def segmentation_to_meet(
    segmentation, 
    ts,
    num_layers = None,
) -> np.array:
    """
    """
    # initialize a 3d array to store all the meet matrices for each layer
    hier_anno = segmentation.annotations
    if num_layers:
        hier_anno = hier_anno[:num_layers]
    n_level = len(hier_anno)
    n_frames = len(ts)
    meet_mat_per_level = np.zeros((n_level, n_frames, n_frames))

    # put meet mat of each level of hierarchy in axis=0
    for level in range(n_level):
        layer_anno = hier_anno[level]
        label_samples = layer_anno.to_samples(ts)
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform([l[0] if len(l) > 0 else 'NL' for l in label_samples])
        meet_mat_per_level[level] = np.equal.outer(encoded_labels, encoded_labels).astype('float') * (level + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def segmentation_to_mireval(
    segmentation
) -> tuple:
    """
    """
    mir_eval_interval = []
    mir_eval_label = []
    for anno in segmentation.annotations:
        interval, value = anno.to_interval_values()
        mir_eval_interval.append(interval)
        mir_eval_label.append(value)

    return mir_eval_interval, mir_eval_label


def tau_ssm(
    ssm: np.array,
    segmentation: jams.JAMS,
    ts: np.array,
    quantize: str = None, # can be 'percentile' or 'kmeans' 
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat_flat = segmentation_to_meet(segmentation, ts).flatten()
    if quantize == 'percentile':
        bins = [np.percentile(ssm, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        ssm_flat = np.digitize(ssm.flatten(), bins=bins, right=False)
    if quantize == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        ssm_flat = kmeans_clusterer.fit_predict(ssm.flatten())
    else:
        ssm_flat = ssm.flatten()

    return stats.kendalltau(ssm_flat, meet_mat_flat)[0]


def tau_path(
    path_sim: np.array,
    segmentation: jams.JAMS,
    ts: np.array,
    quantize: str = 'kmeans',  # None, 'kmeans', and 'percentil'
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat = segmentation_to_meet(segmentation, ts)
    meet_diag = np.diag(meet_mat, k=1)
    if quantize == 'percentile':
        bins = [np.percentile(path_sim, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        path_sim = np.digitize(path_sim, bins=bins, right=False)
    elif quantize == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        path_sim = kmeans_clusterer.fit_predict(path_sim)
    return stats.kendalltau(path_sim, meet_diag)[0]


def compute_l(
    proposal: jams.JAMS, 
    annotation: jams.JAMS,
) -> np.array:
    """
    """
    anno_interval, anno_label = segmentation_to_mireval(annotation)
    proposal_interval, proposal_label = segmentation_to_mireval(proposal)

    # make last segment for estimation end at the same time as annotation
    end = max(anno_interval[-1][-1, 1], proposal_interval[-1][-1, 1])
    for i in range(len(proposal_interval)):
        proposal_interval[i][-1, 1] = end
    for i in range(len(anno_interval)):
        anno_interval[i][-1, 1] = end

    return mir_eval.hierarchy.lmeasure(
        anno_interval, anno_label, proposal_interval, proposal_label,
    )

