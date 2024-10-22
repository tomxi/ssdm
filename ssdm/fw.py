import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from scipy import linalg, stats
import librosa, mir_eval
from .formatting import multi2mireval, mireval2multi
from .utils import quantize

def boundary_kde(boundaries, bw=0.25):
    if len(boundaries) == 1:
        return stats.norm(loc=boundaries[0], scale=bw).pdf
    else:
        kde_bw = bw / boundaries.std(ddof=1)
        kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
        return kde

# Building a KDE with boundaries, then sample the KDE 
# using either sr or using a set of specified sampling points
def boundary_salience(hier_intervals, trim=True, bw=0.25, sr=10, ts=None):
    mir_eval.hierarchy.validate_hier_intervals(hier_intervals)
    # Get the start and end time of the whole anno
    start = hier_intervals[0][0][0]
    end = hier_intervals[0][-1][-1]
    
    # Salience curves sampled from KDE at these positions
    if ts is None:
        # Time grid according to sr for the salience curve
        num_ticks = int(np.round((end-start) * sr)) + 1
        ts = np.linspace(start, end, num_ticks)

    # use KDE to build salience curves from flat segment boundaries
    salience_per_layer = []
    layer_labels = []
    for i, intervals in enumerate(hier_intervals):
        boundaries = mir_eval.util.intervals_to_boundaries(intervals, q=3)
        
        # ignore empty segmnetations
        if len(boundaries) > 2:    
            if trim:
                boundaries = boundaries[1:-1]
            salience_per_layer.append(boundary_kde(boundaries, bw=bw)(ts))
            layer_labels.append(i)

    boundary_salience_mat = np.array(salience_per_layer)
    return boundary_salience_mat, ts

def multi2bsm(multi_anno, trim=True, bw=0.25, sr=10, ts=None):
    itvls, lbls = multi2mireval(multi_anno)
    return boundary_salience(itvls, trim=trim, bw=bw, sr=sr, ts=ts)

def multi2lstars(multi_anno):
    itvls, lbls = multi2mireval(multi_anno)
    lstar_per_level = []
    for itvl, lbl in zip(itvls, lbls):
        bs = [np.round(b, decimals=2) for b, e in itvl]
        lstar_per_level.append(
            {b: l for b, l in zip(bs, lbl)}
        )
    return lstar_per_level
 

# Peak picking happens here
def pick_bsm2hb(boundary_salience_mat, ts, depth=None, pre_max=10, post_max=10, pre_avg=5, post_avg=5, delta=1e-3, wait=15):
    # Do peak picking on the row-wise average:
    # Get normalized novelty curve by summing the rows
    novelty = boundary_salience_mat.mean(axis=0)
    novelty /= novelty.max() + 1e-10
    # Make sure start and end are at the max:
    novelty[0] = novelty.max(); novelty[-1] = novelty.max()
    # pick peak
    boundaries = librosa.util.peak_pick(novelty, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
    # return boundaries, novelty
    # print(boundaries)

    # Quantize the ranks according to max_depth. Default to number of rows in bsm.
    if depth == None:
        depth = boundary_salience_mat.shape[0]
    # no more depth than num samples
    depth = min(depth, len(np.unique(novelty[boundaries])))

    # Quantize via KMeans
    boundary_salience = quantize(novelty[boundaries], quantize_method='kmeans', quant_bins=depth)
    rated_boundaries = {ts[b]: s for b, s in zip(boundaries, boundary_salience)}
    return rated_boundaries

# THIS ONE NEEDS SOME WORK with labeling.........
def hb2multi(rated_b, lstar_d=None):
    # Build mir_eval format hierarchical segmentation:
    intervals = []
    labels = []
    num_layers = int(max(list(rated_b.values())))
    for l in range(num_layers):
        salience_thresh = num_layers - l
        boundaries = [b for b in rated_b if rated_b[b] >= salience_thresh]
        # print(boundaries)
        intervals.append(mir_eval.util.boundaries_to_intervals(boundaries))
        # What do I do with labels?... For now let'st just put in its start frame position
        if lstar_d is None:
            labels.append(boundaries)
        else:
            labels.append([lstar_d[b] for b in boundaries[:-1]])
    return mireval2multi(intervals, labels)

# Let's do match event first, and get indicator vector of whether each ref bs of any depth is recalled at all:
def hb_recall(ref_rated_boundaries, est_rated_boundaries, window=0.5):
    # Retrun recall for flat segmentation, and then for pairs of ranking violated, and then total relationships = hit + pair
    ref_bs = list(ref_rated_boundaries)
    ref_bs_salience = [ref_rated_boundaries[t] for t in ref_rated_boundaries]
    est_bs = list(est_rated_boundaries)
    hits = mir_eval.util.match_events(ref_bs, est_bs, window=window)
    
    recalled_bs_salience = [0] * len(ref_rated_boundaries)
    for pair in hits:
        # if it's a hit, get the salience of that boundary in est_salience 
        est_salience = est_rated_boundaries[list(est_bs)[pair[1]]]
        recalled_bs_salience[pair[0]] = est_salience + 1
    # print(recalled_bs_salience)

    invertion_count, pair_count = mir_eval.hierarchy._compare_frame_rankings(
        np.array(ref_bs_salience), np.array(recalled_bs_salience), transitive=True
    )
    correct_pair_count = pair_count - invertion_count
    print('recalled and total ranked pairs:', correct_pair_count, pair_count)
    print('recall of all boundaries:', len(hits), len(ref_bs))
    fb_recall = len(hits) / len(ref_bs)
    if pair_count == 0:
        pair_recall = 1
        hb_recall = fb_recall
    else:
        pair_recall = correct_pair_count / pair_count
        hb_recall = (correct_pair_count + len(hits)) /  (pair_count + len(ref_bs))

    return fb_recall, pair_recall, hb_recall

## Wrappers for better interfacing
def hbmeasure(ref_multi_anno, est_multi_anno, window=0.5, ref_depth=None, est_depth=None):
    ref_b, ref_l = multi2mireval(ref_multi_anno)
    est_b, est_l = multi2mireval(est_multi_anno)
    # make last segment for estimation end at the same time as annotation
    end = max(ref_b[-1][-1, 1], est_b[-1][-1, 1])
    for i in range(len(est_b)):
        est_b[i][-1, 1] = end
    for i in range(len(ref_b)):
        ref_b[i][-1, 1] = end
        
    ref_bsm, ts = boundary_salience(ref_b)
    est_bsm, ts = boundary_salience(est_b)

    ref_hier_bdy = pick_bsm2hb(ref_bsm, ts, depth=ref_depth)
    est_hier_bdy = pick_bsm2hb(est_bsm, ts, depth=est_depth)
    return hb_recall(est_hier_bdy, ref_hier_bdy, window=window), hb_recall(ref_hier_bdy, est_hier_bdy, window=window)

def multi2hb(multi_anno, trim=True, bw=0.25, sr=10, **pickbsm_kwargs):
    bsm, ts = multi2bsm(multi_anno, trim=trim, bw=bw, sr=sr)
    return pick_bsm2hb(
        bsm, ts, **pickbsm_kwargs
    )


def hb2bsm(rated_b, trim=True, bw=0.25, sr=10, ts=None):
    return multi2bsm(hb2multi(rated_b), trim=trim, bw=bw, sr=sr, ts=ts)


def bsm2multi(boundary_salience_mat, ts, **pick_bsm_kwargs):
    hb = pick_bsm2hb(boundary_salience_mat, ts, **pick_bsm_kwargs)
    return hb2multi(hb)


## Label stuff
def agg_by_boundaries(mat, ts, bounday_times, agg_func=np.mean):
    # Do it twice on both boundaries
    mat = aggregate_columns_by_boundaries(mat, ts, bounday_times, agg_func).T
    return aggregate_columns_by_boundaries(mat, ts, bounday_times, agg_func).T


def aggregate_columns_by_boundaries(mat, ts, boundary_times, agg_func=np.mean):
    """
    Aggregates columns of the matrix `mat` based on time boundaries defined by `ts` and `boundary_times`.

    Parameters:
    ts (list or np.array): Time points corresponding to columns of the matrix.
    boundary_times (list or np.array): Boundaries to segment time points.
    mat (np.array): Matrix to be aggregated, with columns corresponding to `ts`.
    agg_func (function): Aggregation function, e.g., np.sum or np.mean.

    Returns:
    np.array: Aggregated values for each segment.
    """
    # Ensure all boundary times are included, including the first and last time points
    boundary_times = np.unique(np.concatenate(([ts[0]], boundary_times, [ts[-1]])))
    
    # Find segment indices for each time point in ts
    segment_indices = np.searchsorted(boundary_times, ts, side='right') - 1

    # Initialize an array to store the results
    segment_aggregates = np.zeros((mat.shape[0], len(boundary_times) - 1))
    
    # Aggregate columns for each segment
    for i in range(len(boundary_times) - 1):
        mask = segment_indices == i
        if np.any(mask):
            segment_aggregates[:, i] = agg_func(mat[:, mask], axis=1)
    
    return segment_aggregates

