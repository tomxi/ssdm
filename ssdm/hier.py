from .formatting import mireval2multi
from .viz import multi_seg


from scipy import stats
import numpy as np
# from .utils import quantize, laplacian, anno_to_meet

class S(object):
    """ A flat segmentation, labeled intervals.
    """
    def __init__(self, itvls, labels, sr=10, Bhat_bw=0.25):
        # build 1 lvl multi annotation
        self.itvls = itvls
        self.labels = labels
        self.anno = mireval2multi([itvls], [labels])

        # build Lstar and T
        self.Lstar = {b: l for (b, e), l in zip(itvls, labels)}
        self.beta = np.array(sorted(set(self.Lstar.keys()).union([itvls[-1][-1]])))
        self.T0, self.T = self.beta[0], self.beta[-1]

        # Build BSC and ticks
        self.sr = None; self.ticks = None
        self.update_sr(sr)
        self._Bhat = None; self.Bhat_bw = None
        self.update_bw(Bhat_bw)

        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.seg_dur_area_mat = np.outer(self.seg_dur, self.seg_dur)
        self.total_label_agreement_area = np.sum(self.seg_dur_area_mat * self.A(bs=self.beta))


    def update_bw(self, bw):
        if self.Bhat_bw != bw:
            self.Bhat_bw = bw
            boundaries = self.beta[1:-1]
            if len(boundaries) == 0:
                self._Bhat = lambda _: 0
            elif len(boundaries) == 1:
                self._Bhat = stats.norm(loc=boundaries[0], scale=bw).pdf
            else:
                kde_bw = bw / boundaries.std(ddof=1)
                kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
                self._Bhat = kde


    def update_sr(self, sr):
        if self.sr != sr:
            self.sr = sr
            self.ticks = np.linspace(
                self.T0, self.T, 
                int(np.round((self.T - self.T0) * self.sr)) + 1
            )


    def L(self, x):
        if x > self.T or x < self.T0:
            raise IndexError(f'RANGE: {x} outside the range of this segmentation!')
        # Find the correct interval for x and return the corresponding label
        for i in range(len(self.Lstar)):
            if self.beta[i] <= x < self.beta[i + 1]:
                return self.Lstar[self.beta[i]]
        return self.Lstar[self.beta[-2]]  # For the edge case where x == max(boundary)


    def B(self, x):
        return int(x in self.beta)


    def Bhat(self, ts=None):
        if ts is None:
            ts = self.ticks
        return self._Bhat(ts)
    

    def A(self, bs=None):
        if bs is None:
            bs = self.ticks
        ts = (bs[1:] + bs[:-1]) / 2 # Sample label from mid-points of each frame
        sampled_anno = self.anno.to_samples(ts)
        sample_labels = [obs[0]['label'] for obs in sampled_anno]
        return np.equal.outer(sample_labels, sample_labels).astype('float')


    def Ahat(self, bs=None):
        return self.A(bs) / self.total_label_agreement_area


    def plot(self, **kwargs):
        """plotting kwargs:
        figsize=(8, 3.2), reindex=True, legend_ncol=6, title=None, text=False
        """
        new_kwargs = dict(text=True, legend_ncol=0, figsize=(8, 1.5))
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)


class H(object):
    def __init__(self, itvls, labels, sr=10, Bhat_bw=0.25):
        # Make sure the S have the same time support! S.beta[0], S.sr, and S.T should all be the same
        self.levels = [S(i, l, sr=sr, Bhat_bw=Bhat_bw) for i, l in zip(itvls, labels)]
        self.anno = mireval2multi(itvls, labels)
        self.d = len(self.levels)
        self.sr = self.levels[0].sr
        self.T0, self.T = self.levels[0].T0, self.levels[0].T
        self.beta = np.unique(np.concatenate([seg.beta for seg in self.levels]))

    def Ahats(self, bs=None):
        return np.asarray([lvl.Ahat(bs) for lvl in self.levels])

    def Bhats(self, ts=None):
        return np.asarray([lvl.Bhat(ts) for lvl in self.levels])

    def Ahat(self, bs=None, weights=None):
        if weights is None:
            weights = np.ones(self.d) / self.d
        weighted = np.array(weights).reshape(-1, 1, 1) * self.Ahats(bs)
        return weighted.sum(axis=0)
        
    def Bhat(self, ts=None, weights=None):
        if weights is None:
            weights = np.ones(self.d) / self.d
        weighted = np.array(weights).reshape(-1, 1) * self.Bhats(ts)
        return weighted.sum(axis=0)
    
    def A(self, bs=None):
        """Number of layers that the labels agree on."""
        if bs is None:
            bs = self.beta  # Default to the union of all boundaries
        return sum(level.A(bs=bs) for level in self.levels)  # Element-wise sum across all levels


    def B(self, ts=None):
        """Number of layers each boundary exists in."""
        if ts is None:
            ts = self.beta  # Default to self.beta
        return sum(np.array([level.B(t) for t in ts]) for level in self.levels)


    def Astar(self, bs=None):
        """The deepest level of the hierarchy where the labels are labeled identically."""
        if bs is None:
            bs = self.beta  # Default to the union of all boundaries

        Ahats = self.Ahats(bs=bs)  # Get Ahat for each level, shape (d, len(bs)-1, len(bs)-1)

        # Convert Ahats to binary and multiply each level's agreement by its level index (1-based)
        indexed_Ahats = [(level + 1) * (Ahats[level] > 0).astype(int) for level in range(self.d)]

        # Take the maximum over the level dimension (axis=0) to get the deepest level of agreement
        return np.max(indexed_Ahats, axis=0)

    def has_monoA(self):
        """Check if labels are monotonic across levels, i.e., if H.A == H.Astar."""
        # Compute A and Astar matrices using self.beta
        A_matrix = self.A(bs=self.beta)
        Astar_matrix = self.Astar(bs=self.beta)

        # Check if A and Astar are identical
        return np.allclose(A_matrix, Astar_matrix)
    
    def has_monoB(self):
        """Check if boundaries are monotonic across levels, i.e., earlier level boundaries are subsets of later levels."""
        for i in range(1, self.d):
            # Check if all boundaries of the previous level are in the current level
            if not set(self.levels[i - 1].beta).issubset(self.levels[i].beta):
                return False
        return True


    def M(self, bs=None, **Ahat_kwargs):
        # Cut Ahat up using bs that's the union of all boundaries we are working with here.
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        all_bs = np.array(sorted(set(self.beta).union(bs)))
        
        # get area mat and agreement indicator using all_bs, then sum the right ones to get to M
        seg_dur = all_bs[1:] - all_bs[:-1]
        seg_agreement_area = np.outer(seg_dur, seg_dur) 
        return _resample_matrix(seg_agreement_area * self.Ahat(bs=all_bs, **Ahat_kwargs), all_bs, bs)


    def Mhat(self, bs=None, **Ahat_kwargs):
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        seg_dur = bs[1:] - bs[:-1]
        return self.M(bs, **Ahat_kwargs) / np.outer(seg_dur, seg_dur)

 
    def plot(self, **kwargs):
        """plotting kwargs:
        figsize=(8, 3.2), reindex=True, legend_ncol=6, title=None, text=False
        """
        new_kwargs = dict()
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)


def _resample_matrix(matrix, old_bounds, new_bounds):
    # Find indices of new boundaries within the old boundaries
    indices = np.searchsorted(old_bounds, new_bounds)
    new_size = len(new_bounds) - 1
    new_matrix = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            # Use indices to define the region and sum over it
            top, bottom = indices[i], indices[i + 1]
            left, right = indices[j], indices[j + 1]
            new_matrix[i, j] = np.sum(matrix[top:bottom, left:right])

    return new_matrix

