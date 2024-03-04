import pkg_resources
import json
import xarray as xr

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, cluster

from tqdm import tqdm

import jams
import mir_eval

from ssdm import salami as slm

# import matplotlib
# import matplotlib.pyplot as plt

import ssdm
# import musicsections as ms

#DEPREE These belong in each dataset's code
def get_ids(
    split: str = 'working',
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    """ split can be ['audio', 'jams', 'excluded', 'new_val', 'new_test', 'new_train']
        Dicts sotred in id_path json file.
    """
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(id_path, 'r') as f:
            id_json = json.load(f)
    except FileNotFoundError:
        id_json = dict()
        id_json[split] = []
        with open(id_path, 'w') as f:
            json.dump(id_json, f)
    ids = id_json[split]
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


#DEPREE These belong in each dataset's code
def lucky_track(tids=get_ids(split='working'), announce=True):
    """Randomly pick a track from tids"""
    rand_idx = int(np.floor(np.random.rand() * len(tids)))
    tid = tids[rand_idx]
    if announce:
        print(f'track {tid} is the lucky track!')
    return slm.Track(tid)


# MOVE TO SALAMI
# add new splits to split_ids.json
def update_split_json(split_name='', split_idx=[]):
    # add new splits to split_id.json file at json_path
    # read from json and get dict
    json_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(json_path, 'r') as f:
            split_dict = json.load(f)
    except FileNotFoundError:
        split_dict = dict()
        split_dict[split_name] = split_idx
        with open(json_path, 'w') as f:
            json.dump(split_dict, f)
        return split_dict
    
    # add new split to dict
    split_dict[split_name] = split_idx
    
    # save json again
    with open(json_path, 'w') as f:
        json.dump(split_dict, f)
        
    with open(json_path, 'r') as f:
        return json.load(f)


# MOVE TO SALAMI    
def create_splits(arr, val_ratio=0.15, test_ratio=0.15, random_state=20230327):
    dev_set, test_set = train_test_split(arr, test_size = test_ratio, random_state = random_state)
    train_set, val_set = train_test_split(dev_set, test_size = val_ratio / (1 - test_ratio), random_state = random_state)
    train_set.sort()
    val_set.sort()
    test_set.sort()
    return train_set, val_set, test_set


### Stand alone functions
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


def meet_mat_no_diag(track, rec_mode='expand', diag_mode='refine', anno_id=0):
    diag_block = ssdm.anno_to_meet(track.ref(mode=diag_mode, anno_id=anno_id), ts=track.ts())
    full_rec = ssdm.anno_to_meet(track.ref(mode=rec_mode, anno_id=anno_id), ts=track.ts())
    return (diag_block == 0) * full_rec


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


### Formatting functions ###
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


def layer2mireval(multi_anno, layer=-1):
    all_itvls, all_labels = multiseg_to_mireval(multi_anno)
    return all_itvls[layer], all_labels[layer]


def layer2openseg(multi_anno, layer=-1):
    itvls, labels = layer2mireval(multi_anno, layer)
    anno = jams.Annotation(namespace='segment_open')
    for ival, label in zip(itvls, labels):
        anno.append(time=ival[0], 
                    duration=ival[1]-ival[0],
                    value=str(label))
    return anno


### END OF FORMATTING FUNCTIONS###

## Score collecting functions
def init_empty_xr(grid_coords, name=None):
    shape = [len(grid_coords[option]) for option in grid_coords]
    empty_data = np.empty(shape)
    empty_data[:] = np.nan
    return xr.DataArray(empty_data,
                        dims=grid_coords.keys(),
                        coords=grid_coords,
                        name=name,
                        )

# Move to SALAMI
def get_lsd_scores(
    tids=[], 
    anno_col_fn=lambda stack: stack.max(dim='anno_id'), 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = slm.Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.lsd_score(anno_id=anno_id, **lsd_score_kwargs))
        
        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()

# Move to SALAMI
def get_adobe_scores(
    tids=[],
    anno_col_fn=lambda stack: stack.max(dim='anno_id'),
    l_frame_size=0.1
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = slm.Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.adobe_l(anno_id=anno_id, l_frame_size=l_frame_size))
        
        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()

# Move to SALAMI
def get_taus(
    tids=[], 
    anno_col_fn=lambda stack: stack.max(dim='anno_id'),
    **tau_kwargs,
) -> xr.DataArray:
    tau_per_track = []
    for tid in tqdm(tids):
        track = slm.Track(tid)
        tau_per_anno = []
        for anno_id in range(track.num_annos()):
            tau_per_anno.append(track.tau(anno_id=anno_id, **tau_kwargs))
        
        anno_stack = xr.concat(tau_per_anno, pd.Index(range(len(tau_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        tau_per_track.append(track_flat)
    
    return xr.concat(tau_per_track, pd.Index(tids, name='tid')).rename()


def pick_by_taus(
    scores_grid: xr.DataArray, # tids * num_feat * num_feat
    rep_taus: xr.DataArray, # tids * num_feat
    loc_taus: xr.DataArray = None, # tids * num_feat
) -> pd.DataFrame: # tids * ['rep', 'loc', 'score', 'orc_rep', 'orc_loc', 'oracle']
    """pick the best rep and loc features according to taus from a scores_grid"""
    out = pd.DataFrame(index=scores_grid.tid, 
                       columns=['rep_pick', 'loc_pick', 'score', 'orc_rep_pick', 'orc_loc_pick', 'oracle']
                       )
    
    out.orc_rep_pick = scores_grid.max(dim='loc_ftype').idxmax(dim='rep_ftype').squeeze()
    out.orc_loc_pick = scores_grid.max(dim='rep_ftype').idxmax(dim='loc_ftype').squeeze()
    out.oracle = scores_grid.max(dim=['rep_ftype', 'loc_ftype']).squeeze()
    
    rep_pick = rep_taus.idxmax(dim='f_type')
    
    if loc_taus is None:
        loc_pick = out.orc_loc_pick.to_xarray()
    else:
        loc_pick = loc_taus.idxmax(dim='f_type').fillna('mfcc')

    out.rep_pick = rep_pick
    out.loc_pick = loc_pick
    out.score = scores_grid.sel(rep_ftype=rep_pick, loc_ftype=loc_pick)
    
    return out


def quantize(data, quantize_method='percentile', quant_bins=8):
    # method can me 'percentile' 'kmeans'. Everything else will be no quantize
    data_shape = data.shape
    if quantize_method == 'percentile':
        bins = [np.percentile(data[data > 0], bin * (100.0/(quant_bins - 1))) for bin in range(quant_bins)]
        # print(bins)
        quant_data_flat = np.digitize(data.flatten(), bins=bins, right=False)
    elif quantize_method == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        quantized_non_zeros = kmeans_clusterer.fit_predict(data[data>0][:, None])
        # make sure the kmeans group are sorted with asending centroid and relabel
        new_ccenter_order = np.argsort(kmeans_clusterer.cluster_centers_.flatten())
        nco = new_ccenter_order[new_ccenter_order]
        quantized_non_zeros = np.array([nco[g] for g in quantized_non_zeros], dtype=int)

        quant_data = np.zeros(data.shape)
        quant_data[data>0] = quantized_non_zeros + 1
        quant_data_flat = quant_data.flatten()
    elif quantize_method is None:
        quant_data_flat = data.flatten()
    else:
        assert('bad quantize method')

    return quant_data_flat.reshape(data_shape)

