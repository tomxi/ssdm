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
import librosa
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
    hier_labels = [[''] * n_frames for _ in range(num_layers)]

    # run through sampled observations in multi_seg anno
    for t, sample in enumerate(sampled_anno):
        for label_per_level in sample:
            lvl = label_per_level['level']
            label = label_per_level['label']
            if lvl < num_layers:
                hier_labels[lvl][t] = label

    
    # clean labels
    le = preprocessing.LabelEncoder()
    hier_encoded_labels = []
    for lvl_label in hier_labels:
        hier_encoded_labels.append(
            le.fit_transform([l if len(l) > 0 else 'NL' for l in lvl_label])
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
    quantize: str = 'percentile', # can be 'percentile' or 'kmeans' or None
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat_flat = anno_to_meet(segmentation, ts).flatten()
    if quantize == 'percentile':
        bins = [np.percentile(ssm[ssm > 0], bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        # print(bins)
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
    quantize: str = 'percentil',  # None, 'kmeans', and 'percentil'
    quant_bins: int = 6, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    meet_mat = anno_to_meet(segmentation, ts)
    meet_diag = np.diag(meet_mat, k=1)
    if quantize == 'percentile':
        bins = [np.percentile(path_sim, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
        # print('tau_loc_bins:', bins)
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
        intervals.append(np.array(itv, dtype=float))
        labels.append(lbl)

    return intervals, labels


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


def clean_anno(
    anno, 
    min_duration=8
) -> jams.Annotation:
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


def get_lsd_scores(
    tids=[], 
    anno_col_fn=lambda stack: stack.mean(dim='anno_id'), 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = ssdm.Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.lsd_score(anno_id=anno_id, **lsd_score_kwargs))
        
        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


def get_adobe_scores(
    tids=[],
    anno_col_fn=lambda stack: stack.mean(dim='anno_id'),
    l_frame_size=0.1
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = ssdm.Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.adobe_l(anno_id=anno_id, l_frame_size=l_frame_size))
        
        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


def get_taus(
    tids=[], 
    anno_col_fn=lambda stack: stack.max(dim='anno_id'),
) -> xr.DataArray:
    tau_per_track = []
    for tid in tids:
        track = ssdm.Track(tid)
        tau_per_anno = []
        for anno_id in range(track.num_annos()):
            tau_per_anno.append(track.tau(anno_id=anno_id))
        
        anno_stack = xr.concat(tau_per_anno, pd.Index(range(len(tau_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        tau_per_track.append(track_flat)
    
    return xr.concat(tau_per_track, pd.Index(tids, name='tid')).rename()



def undone_lsd_tids(
    tids=[], 
    lsd_sel_dict=dict(rep_metric='cosine',bandwidth='med_k_scalar',rec_full=0,), 
    l_frame_size=0.1, 
    section_fusion_min_dur=None
):
    undone_ids = []
    for tid in tqdm(tids):
        track = ssdm.Track(tid)
        fusion_flag = f'_f{section_fusion_min_dur}' if section_fusion_min_dur else ''
        nc_path = os.path.join(track.salami_dir, f'ells/{track.tid}_{l_frame_size}{fusion_flag}.nc')
        try:
            lsd_score_da = xr.open_dataarray(nc_path)
        except FileNotFoundError:
            undone_ids.append(tid)
            continue

        if lsd_score_da.sel(lsd_sel_dict).isnull().any():
            undone_ids.append(tid)

    return undone_ids