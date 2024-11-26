from .formatting import mireval2multi
from .viz import multi_seg

import numpy as np
from scipy import stats

class S:
    """A flat segmentation, labeled intervals."""
    def __init__(self, itvls, labels, sr=10, Bhat_bw=0.25):
        """Initialize the flat segmentation."""
        self.itvls = itvls
        self.labels = labels
        self.anno = mireval2multi([itvls], [labels])

        # Build Lstar and T
        self.Lstar = {b: l for (b, e), l in zip(itvls, labels)}
        self.beta = np.array(sorted(set(self.Lstar.keys()).union([itvls[-1][-1]])))
        self.T0, self.T = self.beta[0], self.beta[-1]

        # Build BSC and ticks
        self.update_sr(sr)
        self.update_bw(Bhat_bw)

        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.seg_dur_area_mat = np.outer(self.seg_dur, self.seg_dur)
        self.total_label_agreement_area = np.sum(self.seg_dur_area_mat * self.A(bs=self.beta))

    def update_bw(self, bw):
        """Update bandwidth for Bhat calculation."""
        if hasattr(self, 'Bhat_bw') and self.Bhat_bw == bw:
            return
        self.Bhat_bw = bw
        boundaries = self.beta[1:-1]
        if len(boundaries) == 0:
            self._Bhat = lambda _: 0
        elif len(boundaries) == 1:
            self._Bhat = stats.norm(loc=boundaries[0], scale=bw).pdf
        else:
            kde_bw = bw / boundaries.std(ddof=1) if boundaries.std(ddof=1) != 0 else bw
            kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
            self._Bhat = kde

    def update_sr(self, sr):
        """Update sampling rate and ticks."""
        if hasattr(self, 'sr') and self.sr == sr:
            return
        self.sr = sr
        self.ticks = np.linspace(self.T0, self.T, int(np.round((self.T - self.T0) * self.sr)) + 1)

    def L(self, x):
        """Return the label for a given time x."""
        if not (self.T0 <= x <= self.T):
            raise IndexError(f'RANGE: {x} outside the range of this segmentation!')
        idx = np.searchsorted(self.beta, x, side='right') - 1
        return self.Lstar[self.beta[idx]]

    def B(self, x):
        """Return whether x is a boundary."""
        return int(x in self.beta)

    def Bhat(self, ts=None):
        """Return the boundary salience curve at given time steps."""
        if ts is None:
            ts = self.ticks
        return self._Bhat(ts)

    def A(self, bs=None):
        """Return the label agreement indicator for given boundaries."""
        if bs is None:
            bs = self.ticks
        ts = (bs[1:] + bs[:-1]) / 2  # Sample label from mid-points of each frame
        sampled_anno = self.anno.to_samples(ts)
        sample_labels = [obs[0]['label'] for obs in sampled_anno]
        return np.equal.outer(sample_labels, sample_labels).astype(float)

    def Ahat(self, bs=None):
        """Return the label agreement matrix."""
        return self.A(bs) / self.total_label_agreement_area

    def plot(self, **kwargs):
        """Plot the segmentation."""
        new_kwargs = dict(text=True, legend_ncol=0, figsize=(8, 1.5))
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)


class H:
    """A hierarchical segmentation composed of multiple flat segmentations."""
    def __init__(self, itvls, labels, sr=10, Bhat_bw=0.25):
        """Initialize the hierarchical segmentation."""
        self.levels = [S(i, l, sr=sr, Bhat_bw=Bhat_bw) for i, l in zip(itvls, labels)]
        self.anno = mireval2multi(itvls, labels)
        self.d = len(self.levels)
        self.sr = self.levels[0].sr
        self.T0, self.T = self.levels[0].T0, self.levels[0].T
        self.beta = np.unique(np.concatenate([seg.beta for seg in self.levels]))

    def Ahats(self, bs=None):
        """Return the normalized label agreement matrices for all levels."""
        return np.asarray([lvl.Ahat(bs) for lvl in self.levels])

    def Bhats(self, ts=None):
        """Return the smoothed boundary strengths for all levels."""
        return np.asarray([lvl.Bhat(ts) for lvl in self.levels])

    def Ahat(self, bs=None, weights=None):
        """Return the weighted normalized label agreement matrix."""
        if weights is None:
            weights = np.ones(self.d) / self.d
        weighted = np.array(weights).reshape(-1, 1, 1) * self.Ahats(bs)
        return np.sum(weighted, axis=0)

    def Bhat(self, ts=None, weights=None):
        """Return the weighted smoothed boundary strength."""
        if ts is None:
            ts = self.beta
        if weights is None:
            weights = np.ones(self.d) / self.d
        weighted = np.array(weights).reshape(-1, 1) * self.Bhats(ts)
        return np.sum(weighted, axis=0)

    def A(self, bs=None):
        """Return the label agreement matrix for all levels."""
        if bs is None:
            bs = self.beta
        return sum(level.A(bs=bs) for level in self.levels)

    def B(self, ts=None):
        """Return the boundary count across all levels."""
        if ts is None:
            ts = self.beta
        return sum(level.B(t) for t in ts for level in self.levels)

    def Astar(self, bs=None):
        """Return the deepest level where labels are identical."""
        if bs is None:
            bs = self.beta
        Ahats = self.Ahats(bs=bs)
        indexed_Ahats = np.array([(level + 1) * (Ahats[level] > 0).astype(int) for level in range(self.d)])
        return np.max(indexed_Ahats, axis=0)

    def has_monoA(self):
        """Check if labels are monotonic across levels."""
        return np.allclose(self.A(bs=self.beta), self.Astar(bs=self.beta))

    def has_monoB(self):
        """Check if boundaries are monotonic across levels."""
        return all(set(self.levels[i - 1].beta).issubset(self.levels[i].beta) for i in range(1, self.d))

    def M(self, bs=None, **Ahat_kwargs):
        """Return the resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        all_bs = np.array(sorted(set(self.beta).union(bs)))
        seg_dur = all_bs[1:] - all_bs[:-1]
        seg_agreement_area = np.outer(seg_dur, seg_dur)
        return _resample_matrix(seg_agreement_area * self.Ahat(bs=all_bs, **Ahat_kwargs), all_bs, bs)

    def Mhat(self, bs=None, **Ahat_kwargs):
        """Return the normalized resampled agreement area matrix."""
        if bs is None:
            bs = self.beta
        bs = np.array(sorted(set(bs).union([self.T, self.T0])))
        seg_dur = bs[1:] - bs[:-1]
        return self.M(bs, **Ahat_kwargs) / np.outer(seg_dur, seg_dur)

    def plot(self, **kwargs):
        """Plot the hierarchical segmentation."""
        new_kwargs = dict()
        new_kwargs.update(kwargs)
        return multi_seg(self.anno, **new_kwargs)


def _resample_matrix(matrix, old_bounds, new_bounds):
    """Resample the given matrix based on new boundaries."""
    indices = np.searchsorted(old_bounds, new_bounds)
    new_size = len(new_bounds) - 1
    new_matrix = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            top, bottom = indices[i], indices[i + 1]
            left, right = indices[j], indices[j + 1]
            new_matrix[i, j] = np.sum(matrix[top:bottom, left:right])

    return new_matrix
