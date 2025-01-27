import xarray as xr
import pandas as pd
import numpy as np


from sklearn import preprocessing, cluster
from sklearn.model_selection import train_test_split
from scipy import sparse, stats
from tqdm import tqdm

import random
import jams, mir_eval

import ssdm
import ssdm.scluster as sc


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
    bs: list,
    num_layers: int = None,
    mode: str = 'max',
    normalize: bool = False,
    density: bool = False,
) -> np.array:
    """
    returns a square mat that's the meet matrix
    The matrix sums to len(ts) * len(ts), ie it's entry mean is 1 if normalized.
    """
    ts = (bs[1:] + bs[:-1]) / 2 # Take mid-point of segment
    #sample the annotation first and input cleaning:
    sampled_anno = anno.to_samples(ts)
    if num_layers is None:
        num_layers = len(sampled_anno[0])
    n_frames = len(ts)
    
    # Create the area matrix as the outer product of seg_dur with itself, for normalization
    if density:
        seg_dur = bs[1:] - bs[:-1]
        area_matrix = np.outer(seg_dur, seg_dur)

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

    if mode == 'max':
        # put meet mat of each level of hierarchy in axis=0           
        for l in range(num_layers):
            meet_mat_per_level[l] = np.equal.outer(hier_encoded_labels[l], hier_encoded_labels[l]).astype('float') * (l + 1)
            if normalize:
                meet_mat_per_level[l] /= meet_mat_per_level[l].sum()
                if density:
                    meet_mat_per_level[l] /= area_matrix
        # get the deepest level matched
        return np.max(meet_mat_per_level, axis=0)
    elif mode == 'mean':
        for l in range(num_layers):
            meet_mat_per_level[l] = np.equal.outer(hier_encoded_labels[l], hier_encoded_labels[l]).astype('float')
            if normalize:
                meet_mat_per_level[l] /= meet_mat_per_level[l].sum()
                if density:
                    meet_mat_per_level[l] /= area_matrix
        return np.mean(meet_mat_per_level, axis=0)
    else:
        for l in range(num_layers):
            meet_mat_per_level[l] = np.equal.outer(hier_encoded_labels[l], hier_encoded_labels[l]).astype('float')
            if normalize:
                meet_mat_per_level[l] /= meet_mat_per_level[l].sum()
                if density:
                    meet_mat_per_level[l] /= area_matrix
        return meet_mat_per_level
       

def meet_mat_no_diag(track, rec_mode='expand', diag_mode='refine', anno_id=0):
    diag_block = ssdm.anno_to_meet(track.ref(mode=diag_mode, anno_id=anno_id), bs=track.ts())
    full_rec = ssdm.anno_to_meet(track.ref(mode=rec_mode, anno_id=anno_id), bs=track.ts())
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


def quantize(data, quantize_method='percentile', quant_bins=8):
    # method can me 'percentile' 'kmeans'. Everything else will be no quantize
    data_shape = data.shape
    if quantize_method == 'percentile':
        bins = [np.percentile(data[data > 0], bin * (100.0/(quant_bins - 1))) for bin in range(quant_bins)]
        # print(bins)
        quant_data_flat = np.digitize(data.flatten(), bins=bins, right=False)
    elif quantize_method == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins, n_init=50, max_iter=500)
        quantized_non_zeros = kmeans_clusterer.fit_predict(data[data>0][:, None])
        # make sure the kmeans group are sorted with asending centroid and relabel
        
        nco = stats.rankdata(kmeans_clusterer.cluster_centers_.flatten())
        # print(kmeans_clusterer.cluster_centers_, nco)
        quantized_non_zeros = np.array([nco[g] for g in quantized_non_zeros], dtype=int)

        quant_data = np.zeros(data.shape)
        quant_data[data>0] = quantized_non_zeros
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
    verbose=False,
    **lsd_score_kwargs,
) -> xr.DataArray:    
    score_per_track = []
    tids = ds.tids
    if shuffle:
        random.shuffle(tids)

    if verbose:
        tid_iterator = tqdm(tids)
    else:
        tid_iterator = tids
    for tid in tid_iterator:
        track = ds.ds_module.Track(tid=tid)
        if track.num_annos() == 1:
            score_per_track.append(track.lsd_score(**lsd_score_kwargs))
        else:
            score_per_anno = []
            for anno_id in range(track.num_annos()):
                score_per_anno.append(track.lsd_score(anno_id=anno_id, **lsd_score_kwargs))
            anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
            score_per_track.append(anno_col_fn(anno_stack))

    return xr.concat(score_per_track, pd.Index(tids, name='tid'), coords='minimal').rename().sortby('tid')

