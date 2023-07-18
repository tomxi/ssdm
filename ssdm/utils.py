import pkg_resources
from functools import reduce
import json, os, glob
import xarray as xr

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats, sparse
from sklearn import cluster
from tqdm import tqdm

import jams
import mir_eval
from librosa import ParameterError

import ssdm
import musicsections as ms

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
# NOTE: STALE
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
                anno_scores[str(anno_id)] = track.lsd_l_feature_grid(anno_mode='expand', l_type=l_type, anno_id=anno_id)

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
                anno_l_scores = track.lsd_l_feature_grid(anno_mode='expand', l_type=l_type, anno_id=anno_id)
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
            anno_tau_rep = track.tau(anno_id=anno_id, quantize='kmeans')[f'full_expand']
            anno_tau_loc = track.tau(anno_id=anno_id, quantize='kmeans')[f'path_expand']

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
def anno_to_meet(
    anno: jams.Annotation, 
    ts: list,
    num_layers: int = None,
) -> np.array:
    """
    returns a square mat that's the meet matrix
    """
    #sample the annotation first and input cleaning:
    sampled_anno = anno.to_samples(ts)
    if num_layers is None:
        num_layers = len(sampled_anno[0])
    n_frames = len(ts)
    
    # initialize a 3d array to store all the meet matrices for each layer
    meet_mat_per_level = np.zeros((num_layers, n_frames, n_frames))

    # build hier samples list of list
    hier_labels = [[] for _ in range(num_layers)]

    # run through sampled observations in multi_seg anno
    for t, sample in enumerate(sampled_anno):
        for label_per_level in sample:
            lvl = label_per_level['level']
            label = label_per_level['label']
            if lvl < num_layers:
                # this is to ensure there's only 1 label per lvl per time point
                try:
                    hier_labels[lvl][t] = label
                except IndexError:
                    hier_labels[lvl].append(label)

    # clean labels
    le = preprocessing.LabelEncoder()
    hier_encoded_labels = []
    for lvl_label in hier_labels:
        hier_encoded_labels.append(
            le.fit_transform([l[0] if len(l) > 0 else 'NL' for l in lvl_label])
        )

    # put meet mat of each level of hierarchy in axis=0           
    for l in range(num_layers):
        meet_mat_per_level[l] = np.equal.outer(hier_encoded_labels[l], hier_encoded_labels[l]).astype('float') * (l + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def tau_ssm(
    ssm: np.array,
    segmentation: jams.JAMS,
    ts: np.array,
    quantize: str = 'kmeans', # can be 'percentile' or 'kmeans' or None
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat_flat = anno_to_meet(segmentation, ts).flatten()
    if quantize == 'percentile':
        bins = [np.percentile(ssm[ssm > 0], bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        print(bins)
        ssm_flat = np.digitize(ssm.flatten(), bins=bins, right=False)
    elif quantize == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        ssm_flat = kmeans_clusterer.fit_predict(ssm.flatten()[:, None])
    else:
        ssm_flat = ssm.flatten()

    # return ssm_flat, meet_mat_flat
    return stats.kendalltau(ssm_flat, meet_mat_flat)[0]


def tau_path(
    path_sim: np.array,
    segmentation: jams.JAMS,
    ts: np.array,
    quantize: str = 'kmeans',  # None, 'kmeans', and 'percentil'
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat = anno_to_meet(segmentation, ts)
    meet_diag = np.diag(meet_mat, k=1)
    if quantize == 'percentile':
        bins = [np.percentile(path_sim, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        print('tau_loc_bins:', bins)
        path_sim = np.digitize(path_sim, bins=bins, right=False)
    elif quantize == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        path_sim = kmeans_clusterer.fit_predict(path_sim[:, None])
    return stats.kendalltau(path_sim, meet_diag)[0]


def compute_l(
    proposal: jams.Annotation, 
    annotation: jams.Annotation,
    l_frame_size: float = 0.1
) -> np.array:
    anno_interval, anno_label = multiseg_to_mireval(annotation)
    proposal_interval, proposal_label = multiseg_to_mireval(proposal)

    # make last segment for estimation end at the same time as annotation
    end = max(anno_interval[-1][-1, 1], proposal_interval[-1][-1, 1])
    for i in range(len(proposal_interval)):
        proposal_interval[i][-1, 1] = end
    for i in range(len(anno_interval)):
        anno_interval[i][-1, 1] = end

    return mir_eval.hierarchy.lmeasure(
        anno_interval, anno_label, proposal_interval, proposal_label, 
        frame_size=l_frame_size
    )


# def mask_diag(sq_mat, width=13):
#     # carve out width from the full ssm
#     sq_mat_lil = sparse.lil_matrix(sq_mat)
#     for diag in range(-width + 1, width):
#         sq_mat_lil.setdiag(0, diag)
#     sq_mat = sq_mat_lil.toarray()



def multiseg_to_hier(anno)-> list:
    n_lvl_list = [obs.value['level'] for obs in anno]
    n_lvl = max(n_lvl_list) + 1
    hier = [[[],[]] for i in range(n_lvl)]
    for obs in anno:
        lvl = obs.value['level']
        label = obs.value['label']
        interval = [obs.time, obs.time+obs.duration]
        hier[lvl][0].append(interval)
        hier[lvl][1].append(f'{label}')
    return hier


def hier_to_multiseg(hier) -> jams.Annotation:
    anno = jams.Annotation(namespace='multi_segment')
    for layer, (intervals, labels) in enumerate(hier):
        for ival, label in zip(intervals, labels):
            anno.append(time=ival[0], 
                        duration=ival[1]-ival[0],
                        value={'label': str(label), 'level': layer})
    return anno

def hier_to_mireval(hier) -> tuple:
    intervals = []
    labels = []
    for itv, lbl in hier:
        intervals.append(np.asarray(itv))
        labels.append(lbl)

    return np.array(intervals, dtype=object), labels


def mireval_to_hier(itvls: np.ndarray, labels: list) -> list:
    hier = []
    n_lvl = len(labels)
    for lvl in range(n_lvl):
        lvl_anno = [itvls[lvl], labels[lvl]]
        hier.append(lvl_anno)
    return hier


def multiseg_to_mireval(anno) -> tuple:
    return hier_to_mireval(multiseg_to_hier(anno))


def mireval_to_multiseg(itvls: np.ndarray, labels: list) -> jams.Annotation:
    return hier_to_multiseg(mireval_to_hier(itvls, labels))


def clean_anno(anno, min_duration=8) -> jams.Annotation:
    """wrapper around adobe's clean_segments function"""
    hier = multiseg_to_hier(anno)
    levels = ms.core.reindex(hier)
    
    # If min_duration is set, apply multi-level SECTION FUSION 
    # to remove short sections
    fixed_levels = None
    if min_duration is None:
        fixed_levels = levels
    else:
        segs_list = []
        for i in range(1, len(levels) + 1):
            segs_list.append(ms.core.clean_segments(levels, 
                                                    min_duration=min_duration, 
                                                    fix_level=i, 
                                                    verbose=False))
        
        fixed_levels = ms.core.segments_to_levels(segs_list)
    
    return hier_to_multiseg(fixed_levels)


def openseg2multi(
    annos: list
) -> jams.Annotation:
    multi_anno = jams.Annotation(namespace='multi_segment')

    for lvl, openseg in enumerate(annos):
        for obs in openseg:
            multi_anno.append(time=obs.time,
                              duration=obs.duration,
                              value={'label': obs.value, 'level': lvl},
                             )
    
    return multi_anno


def init_empty_xr(grid_coords, name=None):
    shape = [len(grid_coords[option]) for option in grid_coords]
    empty_data = np.empty(shape)
    empty_data[:] = np.nan
    return xr.DataArray(empty_data,
                        dims=grid_coords.keys(),
                        coords=grid_coords,
                        name=name,
                        )

# collect jams for each track: 36(feat) * 9(dist) * 3(bw) combos for each track. Let's get the 36 * 9 feature distances collected in one jams.
# This is used once and should be DEPRE, but don't delete the code yet!
# def collect_lsd_jams(track, bandwidth_mode='med_k_scalar'):
#     # load jams if lsd_jams_path already exist
#     # create if not
#     lsd_root = '/scratch/qx244/data/salami/lsds/'
#     fname = f'{track.tid}_{bandwidth_mode}.jams'

    
#     lsd_jams_path = os.path.join(lsd_root, fname)
#     if os.path.exists(lsd_jams_path):
#         print('loading')
#         jam = jams.load(lsd_jams_path)
#         return jam
#     else:
#         print('collecting')
#         jam = jams.JAMS()
#         jam.file_metadata.duration = track.ts()[-1]

    
    
#     # a bunch of for statements
#     # loc_feat = 'crema' 
#     # l_dist = 'cityblock'
#     # rep_feat = 'chroma' 
#     # r_dist = 'cityblock'
#     rep_aff_mat_mode = 'sparse'
#     for loc_feat in ssdm.AVAL_FEAT_TYPES:
#         for l_dist in ssdm.AVAL_DIST_TYPES:
#             for rep_feat in ssdm.AVAL_FEAT_TYPES:
#                 for r_dist in ssdm.AVAL_DIST_TYPES:
#                     # pull from legacy_jams if exisit, compute if not
#                     legacy_fname = f'{track.tid}_{loc_feat}{l_dist}_{rep_feat}{r_dist}_{bandwidth_mode}_{rep_aff_mat_mode}.jams'
#                     legacy_fp = os.path.join(lsd_root, legacy_fname)
#                     try:
#                         legacy_jam = jams.load(legacy_fp)
#                     except:
#                         continue
#                     multi_lvl_anno = jams.Annotation('multiseg')
#                     multi_lvl_anno.sandbox = legacy_jam.sandbox
#                     # append to annotations
#                     for level, old_lvl_anno in enumerate(legacy_jam.annotations):
#                         for obs in old_lvl_anno:
#                             new_value = {'label': obs.value, 'level': level}
#                             multi_lvl_anno.append(time=obs.time, duration=obs.duration, value=new_value)

#                     jam.annotations.append(multi_lvl_anno)
    
#     # save to lsd_jams_path
#     jam.save(lsd_jams_path)
#     # (after all is debugged) delete legacy_jams
#     # for legacy_file in glob.glob(os.path.join(lsd_root, f'{track.tid}*{bandwidth_mode}*sparse*')):
#     #     os.remove(legacy_file)

#     return jam

