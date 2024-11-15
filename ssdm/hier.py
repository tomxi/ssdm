from .formatting import multi2mireval, mireval2multi
from .viz import multi_seg


from scipy import stats
import numpy as np
# from .utils import quantize, laplacian, anno_to_meet

class S(object):
    """ A flat segmentation, labeled intervals.
    """
    def __init__(self, itvls, labels, sr=10, Bhat_bw=0.25):
        self.itvls = itvls
        self.labels=labels
        self.anno = mireval2multi([itvls], [labels])
        
        self.Lstar = {b: l for (b, e), l in zip(self.itvls, self.labels)}
        self.beta = list(set(self.Lstar.keys()).union([itvls[-1][-1]]))
        self.beta.sort()
        self.T = self.beta[-1]

        self.seg_dur = self.beta[1:] - self.beta[:-1]
        self.seg_dur_area_mat = np.outer(self.seg_dur, self.seg_dur)
        self.label_agreement_area = np.sum(self.seg_dur_area_mat * self.A(bs=self.beta))

        self.update_ticks(sr)
        self.update_Bhat(Bhat_bw)

    def L(self, x):
        # Find the correct interval for x and return the corresponding label
        for i in range(len(self.l_star)):
            if self.beta[i] <= x < self.beta[i + 1]:
                return self.Lstar[self.beta[i]]
        return self.Lstar[self.beta[-2]]  # For the edge case where x == max(boundary)

    def B(self, ts=None):
        if ts is None:
            ts = self.ticks
        try:
            return int(ts in self.beta)
        except:
            return [int(t in self.beta) for t in ts]
    
    def Bhat(self, ts=None):
        if ts is None:
            ts = self.ticks
        return self._Bhat(ts)        

    def A(self, bs=None):
        if bs is None:
            bs = self.ticks
        ts = (bs[1:] + bs[:-1]) / 2 # Take mid-point of segment
        sampled_anno = self.anno.to_samples(ts)        
        return np.equal.outer(sampled_anno, sampled_anno).astype('float')

    def Ahat(self, bs=None):
        if bs is None:
            bs = self.ticks
        return self.A(bs) / self.label_agreement_area

    def M(self):
        pass

    def Mhat(self):
        pass

    def plot(self, **kwargs):
        """plotting kwargs:
        figsize=(8, 3.2), reindex=True, legend_ncol=6, title=None, text=False
        """
        return multi_seg(self.anno, **kwargs)

    def update_Bhat(self, bw):
        if self.Bhat_bw != bw:
            self.Bhat_bw = bw
            boundaries = self.beta[1:-1]
            if len(boundaries) == 1:
                self._Bhat = stats.norm(loc=boundaries[0], scale=bw).pdf
            else:
                kde_bw = bw / boundaries.std(ddof=1)
                kde = stats.gaussian_kde(boundaries, bw_method=kde_bw)
                self._Bhat = kde

    def update_ticks(self, sr):
        if self.sr != sr:
            self.sr = sr
            self.ticks = np.linspace(
                self.beta[0], self.beta[1], 
                int(np.round((self.beta[1] - self.beta[0]) * self.sr)) + 1
            )