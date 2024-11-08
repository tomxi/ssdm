import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import linalg, stats
from collections import Counter
import librosa, mir_eval
from .formatting import multi2mireval, mireval2multi
from .utils import quantize, laplacian, anno_to_meet

multi2lmm = anno_to_meet

def hb2betas(hb, trim=False):
    itvls = hb2intervals(hb)
    if trim:
        return [mir_eval.util.intervals_to_boundaries(i)[1:-1] for i in itvls]
    else:
        return [mir_eval.util.intervals_to_boundaries(i) for i in itvls]


def sc_eg(M, k=None, min_k=1, verbose=False):
    # scluster with k groups. default is eigen gap.
    L = laplacian(M, normalization='random_walk')
    # Assuming L_rw is your random walk normalized Laplacian matrix
    evals, evecs = linalg.eig(L)
    evals = evals.real; evecs = evecs.real
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:,idx]
    egaps = np.diff(evals)
    # print(egaps)
    # print(evals)
    T = len(evals)

    # Determine number of clusters using eigen gap heuristic if k is not provided
    if k is None:
        if min_k >= T or max(evals) < 0.1:
            k = T  # Allow singleton group when all eigenvalues are tiny
        else:
            k = np.argmax(egaps[min_k - 1:]) + min_k

    
    membership = evecs[:, :k]
    KM = KMeans(n_clusters=k, n_init=50, max_iter=500)
    return KM.fit_predict(membership), k


def boundary_kde(boundaries, bw=0.25):
    if len(boundaries) == 1:
        return stats.norm(loc=boundaries[0], scale=bw).pdf
    else:
        kde_bw = bw / boundaries.std(ddof=1)
        kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
        return kde

# Building a KDE with boundaries, then sample the KDE 
# using either sr or using a set of specified sampling points
def boundary_salience(hier_intervals, trim=False, bw=0.25, sr=10, ts=None):
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


def multi2bsm(multi_anno, trim=False, bw=0.25, sr=10, ts=None):
    itvls, lbls = multi2mireval(multi_anno)
    return boundary_salience(itvls, trim=trim, bw=bw, sr=sr, ts=ts)


def multi2lstars(multi_anno):
    itvls, lbls = multi2mireval(multi_anno)
    lstar_per_level = []
    for itvl, lbl in zip(itvls, lbls):
        bs = [np.round(b, decimals=1) for b, e in itvl]
        lstar_per_level.append(
            {b: l for b, l in zip(bs, lbl)}
        )
    T = itvls[-1][-1][-1]
    return lstar_per_level, T
 

def lstars2hb(lstars, T=None):
    boundary_count = Counter()
    bts = [{b:1 for b in lstar} for lstar in lstars]
    for bt in bts:
        if T is None:
            del bt[0.0]
        else:
            T = np.round(T, decimals=1)
            bt[T] = 1

        boundary_count.update(bt)
    return dict(sorted(boundary_count.items()))

# Peak picking happens here
def pick_bsm2hb(boundary_salience_mat, ts, depth=None, pre_max=8, post_max=8, pre_avg=3, post_avg=3, delta=1e-3, wait=10):
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
    rated_boundaries = {np.round(ts[b], decimals=1): s for b, s in zip(boundaries, boundary_salience)}
    return rated_boundaries


# Decoding and labeling happens here
def hb2intervals(rated_b):
    intervals = []
    num_layers = int(max(list(rated_b.values())))
    for l in range(num_layers):
        salience_thresh = num_layers - l
        boundaries = [np.round(b, decimals=1) for b in rated_b if rated_b[b] >= salience_thresh]
        # print(boundaries)
        intervals.append(mir_eval.util.boundaries_to_intervals(boundaries))
    return intervals




def hb2lstars(hb, labs):
    beta_stars = [[p[0] for p in intervals] for intervals in hb2intervals(hb)]
    lstars = []
    for lab_list, beta_star in zip(labs, beta_stars):
        lstars.append({b:l for b, l in zip(beta_star, lab_list)})
    return lstars

# These are code to get labels given boundary and label meet mat estimates (mixed) for each desegth of the hierarchy
def lmms2lstars(lmms, lmm_ts, rated_b):
    lstars = []
    intervals = hb2intervals(rated_b)
    for layer_interval, lmm in zip(intervals, lmms):
        b_times = mir_eval.util.intervals_to_boundaries(layer_interval)
        lstars.append(lmm2lstar(lmm, lmm_ts, b_times))
    return lstars


def lmm2lstar(lmm, lmm_ts, b_times):
    # spectral clustering on lmm with Eigen-gap, aggregated by b_times
    seg_sim_mat = agg_by_boundaries(lmm, lmm_ts, b_times)
    seg_ids = sc_eg(seg_sim_mat)
    return {str(np.round(b, decimals=1)): l for b, l in zip(b_times, seg_ids)}


# With hb and lstar we have defined hierarcies.
def hb2multi(rated_b, lstar_d=None, monotonic_label=False):
    intervals = hb2intervals(rated_b)
    start_times = []
    for interval in intervals:
        start_times.append([np.round(pair[0], decimals=1) for pair in interval])
    if lstar_d is None:
        labels = start_times
    elif not monotonic_label:
        labels = [list(d.values()) for d in lstar_d]
    else:
        labels = []
        for s_list in start_times:
            labels.append([lstar_d[-1][s] for s in s_list])
    return mireval2multi(intervals, labels)


# More formatting
def multi2hb_bsm(multi_anno, trim=False, bw=0.25, sr=10, **pickbsm_kwargs):
    bsm, ts = multi2bsm(multi_anno, trim=trim, bw=bw, sr=sr)
    full_bs = pick_bsm2hb(
        bsm, ts, **pickbsm_kwargs
    )
    if trim:
        del full_bs[0.0]
        del full_bs[list(full_bs)[-1]]
    return full_bs


def multi2hb(multi_anno, trim=False):
    if trim:
        return lstars2hb(multi2lstars(multi_anno)[0])
    else:
        return lstars2hb(*multi2lstars(multi_anno))


def hb2bsm(rated_b, trim=False, bw=0.25, sr=10, ts=None):
    return multi2bsm(hb2multi(rated_b), trim=trim, bw=bw, sr=sr, ts=ts)


def bsm2multi(boundary_salience_mat, ts, lstar_d=None, **pick_bsm_kwargs):
    hb = pick_bsm2hb(boundary_salience_mat, ts, **pick_bsm_kwargs)
    return hb2multi(hb, lstar_d=lstar_d)


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
    # Convert inputs to numpy arrays
    ts = np.asarray(ts, dtype=float)
    boundary_times = np.asarray(boundary_times, dtype=float)

    # Ensure all boundary times are included, including the first and last time points
    boundary_times = np.concatenate(([ts[0]], boundary_times, [ts[-1]]))
    boundary_times = np.unique(boundary_times)

    # Find segment indices for each time point in ts
    segment_indices = np.searchsorted(boundary_times, ts[:-1], side='right') - 1

    # Initialize array to store aggregated values
    segment_aggregates = []

    # Aggregate columns for each segment
    for i in range(len(boundary_times) - 1):
        mask = (segment_indices == i)
        if np.any(mask):
            aggregated_values = agg_func(mat[:, mask], axis=1)
            segment_aggregates.append(aggregated_values)

    # Convert result list to a numpy array
    segment_aggregates = np.column_stack(segment_aggregates)

    return segment_aggregates


## New label stuff
class LabelMapping(object):
    def __init__(self, l_star, T):
        # l_star is a dictionary mapping segment starting times to labels, T is the last boundary, the end of the piece.
        # Ensure boundaries start from 0
        self.l_star, self.T = l_star, T
        self.beta_star = list(l_star)
        self.beta_full = np.array(list(l_star) + [T], dtype=float)
        
    def __call__(self, x):
        # Find the correct interval for x and return the corresponding label
        for i in range(len(self.l_star)):
            if self.beta_full[i] <= x < self.beta_full[i + 1]:
                # seg_start = str(self.beta_full[i])
                return self.l_star[self.beta_full[i]]
        return self.l_star[self.beta_star[-1]]  # For the edge case where x == max(boundary)

    def segsim(self, boundaries=None, density=False):
        # Integration boundaries
        if boundaries is None:
            boundaries = self.beta_full
        boundaries = np.array(boundaries, dtype=float)
        p = len(boundaries) - 1
        joint_pmf = np.zeros((p, p))

        # Compute joint PMF using the new integration boundaries
        for i in range(p):
            for j in range(p):
                u1, u2 = boundaries[i], boundaries[i + 1]
                v1, v2 = boundaries[j], boundaries[j + 1]

                # Find the intersections with self.beta_full (original segment boundaries)
                u_segments = self.beta_full[(self.beta_full > u1) & (self.beta_full < u2)]
                v_segments = self.beta_full[(self.beta_full > v1) & (self.beta_full < v2)]

                # Combine the original boundaries with the internal segments and sort them
                u_segments = np.concatenate(([u1], u_segments, [u2]))
                v_segments = np.concatenate(([v1], v_segments, [v2]))

                # Calculate the contribution to the joint PMF from each subrectangle
                for k in range(len(u_segments) - 1):
                    for l in range(len(v_segments) - 1):
                        u_start, u_end = u_segments[k], u_segments[k + 1]
                        v_start, v_end = v_segments[l], v_segments[l + 1]

                        # Use the label from the midpoint of each subrectangle to determine if they match
                        if self((u_start + u_end) / 2) == self((v_start + v_end) / 2):
                            overlap_area = (u_end - u_start) * (v_end - v_start)
                            joint_pmf[i, j] += overlap_area

        seg_dur = boundaries[1:] - boundaries[:-1]
        
        if density:
            total_mass = joint_pmf.sum()
            if total_mass != 0:
                joint_pmf /= total_mass
            area_matrix = np.outer(seg_dur, seg_dur)
            joint_pmf /= area_matrix
            return joint_pmf

        else:
            # Normalize by each segment by their marginal probability of being picked
            # joint_pmf /= seg_dur
            # Normalize joint PMF to sum to 1
            total_mass = joint_pmf.sum()
            if total_mass != 0:
                joint_pmf /= total_mass

            return joint_pmf


class A_H(object):
    def __init__(self, lams=[]):
        self.lams = lams
    
    def segsim(self, boundaries=None, density=False):
        if boundaries is None:
            boundaries = self.lams[-1].beta_full
        return np.asarray([lam.segsim(boundaries, density) for lam in self.lams])



### Metric stuff
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

