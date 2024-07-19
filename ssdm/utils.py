import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch import nn

from sklearn import preprocessing, cluster
from sklearn.model_selection import train_test_split
from scipy import sparse
from tqdm import tqdm

import random, os
import jams, mir_eval

import ssdm
import ssdm.scluster as sc
import ssdm.scanner as scn


def noop(x):
    pass

def noop3(x, y, z):
    pass

def noop4(w, x, y, z):
    pass

mir_eval.hierarchy.validate_hier_intervals = noop
mir_eval.segment.validate_boundary = noop3
mir_eval.segment.validate_structure = noop4

def anno_to_meet(
    anno: jams.Annotation,  # multi-layer
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
    l_frame_size: float = 0.1,
    nlvl = None,
) -> np.array:
    anno_interval, anno_label = ssdm.multi2mireval(annotation)
    proposal_interval, proposal_label = ssdm.multi2mireval(proposal)
    if nlvl:
        assert nlvl > 0
        proposal_interval = proposal_interval[:nlvl]
        proposal_label = proposal_label[:nlvl]

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


def compute_flat(
    proposal: jams.Annotation, #Multi-seg
    annotation: jams.Annotation, #Multi-seg,
    a_layer = -1,
    frame_size = 0.1,
) -> np.array:
    ref_inter, ref_labels = ssdm.multi2mirevalflat(annotation, layer=a_layer) # Which anno layer?
    num_prop_layers = len(ssdm.multi2hier(proposal))

    # get empty dataarray for result
    results_dim = dict(
        m_type=['p', 'r', 'f'],
        # metric=['hr', 'hr3', 'pfc', 'v'],
        metric=['v'],
        layer=[x+1 for x in range(16)]
    )
    results = xr.DataArray(data=None, coords=results_dim, dims=list(results_dim.keys()))
    for p_layer in range(num_prop_layers):
        est_inter, est_labels = ssdm.multi2mirevalflat(proposal, layer=p_layer)
        # make last segment for estimation end at the same time as annotation
        end_time = max(ref_inter[-1, 1], est_inter[-1, 1])
        ref_inter[-1, 1] = end_time
        est_inter[-1, 1] = end_time
        
        layer_result_dict = dict(
            # hr=mir_eval.segment.detection(ref_inter, est_inter, window=.5, trim=True),
            # hr3=mir_eval.segment.detection(ref_inter, est_inter, window=3, trim=True),
            # pfc=mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels, frame_size=frame_size),
            v=mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels, frame_size=frame_size)
        )

        for metric in layer_result_dict:
            results.loc[dict(layer=p_layer+1, metric=metric)] = list(layer_result_dict[metric])
    return results


def load_net(model_path='', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #extract info from model_path
    model_basename = os.path.basename(model_path)
    # Check for cached inference results:
    # check all the available models in scanner.py to see if anyone fits
    net = None
    for model_id in scn.AVAL_MODELS:
        if model_basename.find(model_id + '_') >= 0:
            # print(model_basename)
            # print(model_id)
            # returns the first model found in the file base name
            net = scn.AVAL_MODELS[model_id]()
            continue
        elif model_basename.find(model_id + '-') >= 0:
            net = scn.AVAL_MODELS[model_id]()
            continue
    if net == None:
        raise IndexError('could not figure out which model architecutre to initialize.')      
    # load model and do inference
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict=state_dict)
    return net


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


def laplacian(rec_mat, normalization='random_walk'):
    degree_matrix = np.diag(np.sum(rec_mat, axis=1))
    unnormalized_laplacian = degree_matrix - rec_mat
    # Compute the Random Walk normalized Laplacian matrix
    if normalization == 'random_walk':
        degree_inv = np.linalg.inv(degree_matrix)
        return degree_inv @ unnormalized_laplacian       
    elif normalization == 'symmetrical':
        sqrt_degree_inv = np.linalg.inv(np.sqrt(degree_matrix))
        return sqrt_degree_inv @ unnormalized_laplacian @ sqrt_degree_inv
    elif normalization == None:
        return unnormalized_laplacian
    else:
        raise NotImplementedError(f'bad laplacian normalization mode: {normalization}')


# Create test train val splits on the fly, but with random seed.
def create_splits(arr, val_ratio=0.15, test_ratio=0.15, random_state=20230327):
    dev_set, test_set = train_test_split(arr, test_size = test_ratio, random_state = random_state)
    train_set, val_set = train_test_split(dev_set, test_size = val_ratio / (1 - test_ratio), random_state = random_state)
    train_set.sort()
    val_set.sort()
    test_set.sort()
    return dict(train=train_set, val=val_set, test=test_set)


def adjusted_best_layer(cube, tolerance=0):
    max_score_over_layer = cube.max('layer')
    # We can tolerate a bit of perforamcne, so let's take that off from the max
    thresh = max_score_over_layer - tolerance
    # The first layer over threshold
    over_thresh = cube >= thresh
    return over_thresh.idxmax('layer')


# HELPER functin to run laplacean spectral decomposition TODO this is where the rescaling happens
def run_lsd(
    track,
    config: dict,
    beat_sync: bool = False,
    recompute_ssm: bool = False,
    loc_sigma: float = 95
) -> jams.Annotation:
    def mask_diag(sq_mat, width=2):
        # carve out width from the full ssm
        sq_mat_lil = sparse.lil_matrix(sq_mat)
        for diag in range(-width + 1, width):
            sq_mat_lil.setdiag(0, diag)
        return sq_mat_lil.toarray()

    # Compute/get SSM mats
    rep_ssm = track.ssm(feature=config['rep_ftype'],
                        distance=config['rep_metric'],
                        width=config['rec_width'],
                        full=bool(config['rec_full']),
                        recompute=recompute_ssm,
                        beat_sync=beat_sync,
                        add_noise=config['add_noise'],
                        n_steps = config['n_steps'],
                        delay = config['delay']
                        )
    
    if config['rec_full']:
        rep_ssm = mask_diag(rep_ssm, width=config['rec_width'])

    # track.path_sim alwasy recomputes.
    path_sim = track.path_sim(feature=config['loc_ftype'],
                              distance=config['loc_metric'],
                              sigma_percentile=loc_sigma,
                              recompute=True,
                              beat_sync=beat_sync,
                              add_noise=config['add_noise'],
                              n_steps = config['n_steps'],
                              delay = config['delay'])

    # Spectral Clustering with Config
    est_bdry_idxs, est_sgmt_labels = sc.do_segmentation_ssm(rep_ssm, path_sim, config)
    # print(est_bdry_idxs, est_sgmt_labels)
    if beat_sync:
        ts = track.ts(mode='beat', pad=True)
    else:
        ts = track.ts(mode='frame')
    est_bdry_itvls = [sc.times_to_intervals(ts[lvl]) for lvl in est_bdry_idxs]
    return ssdm.mireval2multi(est_bdry_itvls, est_sgmt_labels)


def get_lsd_scores(
    ds,
    shuffle=False,
    anno_col_fn=lambda stack: stack.max(dim='anno_id'), # activated when there are more than 1 annotation for a track
    heir=False,
    beat_sync=True,
    **lsd_score_kwargs,
) -> xr.DataArray:    
    score_per_track = []
    tids = ds.tids
    if shuffle:
        random.shuffle(tids)

    for tid in tqdm(tids):
        track = ds.ds_module.Track(tid=tid)
        if track.num_annos() == 1:
            if heir:
                score_per_track.append(track.lsd_score_l(beat_sync=beat_sync, **lsd_score_kwargs))
            else:
                score_per_track.append(track.lsd_score_flat(beat_sync=beat_sync, **lsd_score_kwargs))
        else:
            score_per_anno = []
            for anno_id in range(track.num_annos()):
                if heir:
                    score_per_anno.append(track.lsd_score_l(anno_id=anno_id, beat_sync=beat_sync, **lsd_score_kwargs))
                else:
                    score_per_anno.append(track.lsd_score_flat(anno_id=anno_id, beat_sync=beat_sync, **lsd_score_kwargs))
            # print(score_per_anno)
            anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
            score_per_track.append(anno_col_fn(anno_stack))

    return xr.concat(score_per_track, pd.Index(tids, name='tid'), coords='minimal').rename().sortby('tid')


def net_pick_performance(ds, net, device='cuda', verbose=False, drop_feats=[]):
    score_da = ds.scores
    net_output = scn.net_infer_multi_loss(ds, net, device, verbose=verbose)
    if drop_feats:
        net_output = net_output.drop_sel(rep_ftype=drop_feats, loc_ftype=drop_feats)
    util_score = net_output.loc[:, :, :, 'util']
    nlvl_pick = net_output.loc[:, :, :, 'nlvl']
    best_util_idx = util_score.argmax(dim=['rep_ftype', 'loc_ftype'])
    net_layer_pick = nlvl_pick.isel(best_util_idx).astype(int)
    net_feat_pick_score = score_da.sel(rep_ftype=net_layer_pick.rep_ftype, loc_ftype=net_layer_pick.loc_ftype)

    # Get oracle feature pick, and see how net is doing on nlvl
    orc_feat_idx = score_da.max('layer').argmax(dim=['rep_ftype', 'loc_ftype'])
    net_layer_pick_orc_feat = nlvl_pick.isel(orc_feat_idx).astype(int)
    orc_feat_scores = score_da.sel(rep_ftype=net_layer_pick_orc_feat.rep_ftype, loc_ftype=net_layer_pick_orc_feat.loc_ftype)

    out = dict(net_pick = net_feat_pick_score.isel(layer=net_layer_pick),
               net_feat_orc_lvl = orc_feat_scores.isel(layer=net_layer_pick_orc_feat),
               orc_feat_net_lvl = net_feat_pick_score.max('layer'),
               orc = orc_feat_scores.max('layer')
               )
    return out

