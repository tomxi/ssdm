import ssdm
import ssdm.utils
from ssdm import scluster
from . import feature
from .expand_hier import expand_hierarchy
import librosa

import jams
import json, itertools, os

import torch
from torch.utils.data import Dataset

import numpy as np
import xarray as xr
from scipy import spatial, stats
from scipy.linalg import eig, eigh

from librosa.segment import recurrence_matrix
from librosa.feature import stack_memory
from librosa import frames_to_time

np.int = int
np.float = float


class Track(object):
    def __init__(
        self,
        tid: str = '',
        dataset_dir: str = '', 
        output_dir: str = '',
        feature_dir: str = '',
    ):
        self.tid = tid
        self.dataset_dir = dataset_dir # where annotations are
        self.output_dir = output_dir
        self.feature_dir = feature_dir

        self.title = tid
        self.audio_path = None
        
        self._sr = 22050 # This is fixed
        self._hop_len = 4096 # This is fixed
        self._y = None
        self._jam = None # populate when .jam() is called.
        self._track_ts = None # populate when .ts() is called.


    def ts(self, mode='frame', pad=False) -> np.array: # mode can also be beat
        if self._track_ts is None:
            num_frames = np.asarray(
                [self.representation(feat, use_track_ts=False, beat_sync=False).shape[-1] for feat in ssdm.AVAL_FEAT_TYPES]
            )
            self._track_ts = frames_to_time(
                list(range(np.min(num_frames))), 
                hop_length=feature._HOP_LEN, 
                sr=feature._AUDIO_SR)
            
        if mode == 'frame':
            return self._track_ts
        elif mode == 'beat':
            beats = self._madmom_beats()
            beats = np.array([b for b in beats if b <= self.ts()[-1]])
            if pad:
                return beats
            else:
                return beats[:-1]


    def combined_rec_mat(
        self, 
        config_update: dict = dict(),
        recompute: bool = False,
        beat_sync: bool = False,
    ):
        config = ssdm.DEFAULT_LSD_CONFIG.copy()
        if beat_sync:
            config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
        config.update(config_update)
        # calculated combined rec mat
        rep_ssm = self.ssm(feature=config['rep_ftype'], 
                            distance=config['rep_metric'],
                            width=config['rec_width'],
                            full=config['rec_full'],
                            recompute=recompute,
                            beat_sync=beat_sync,
                            add_noise=config['add_noise'],
                            n_steps = config['n_steps'],
                            delay=config['delay']
                            )
        path_sim = self.path_sim(feature=config['loc_ftype'], 
                                 distance=config['loc_metric'],
                                 beat_sync=beat_sync,
                                 add_noise=config['add_noise'],
                                 n_steps = config['n_steps'],
                                 delay=config['delay'])
        
        if path_sim.shape[0] != rep_ssm.shape[0] - 1:
            path_sim = path_sim[:rep_ssm.shape[0] - 1]
        return scluster.combine_ssms(rep_ssm, path_sim, rec_smooth=config['rec_smooth'])

    
    def embedded_rec_mat(self, feat_combo=dict(), lap_norm='random_walk', beat_sync=False, recompute=False):
        beat_suffix = {"_bsync" if beat_sync else ""}
        save_path = os.path.join(self.output_dir, f'evecs/{self.tid}_rep{feat_combo["rep_ftype"]}_loc{feat_combo["loc_ftype"]}_{lap_norm}{beat_suffix}.npy')
        if not recompute:
            try:
                return np.load(save_path)
            except:
                print(save_path)
                raise NotImplementedError
                recompute = True
        
        rec_mat = self.combined_rec_mat(config_update=feat_combo, beat_sync=beat_sync)
        degree_matrix = np.diag(np.sum(rec_mat, axis=1))
        unnormalized_laplacian = degree_matrix - rec_mat
        # Compute the Random Walk normalized Laplacian matrix
        if lap_norm == 'random_walk':
            degree_inv = np.linalg.inv(degree_matrix)
            normalized_laplacian = degree_inv @ unnormalized_laplacian
            evals, evecs = eig(normalized_laplacian)
            sort_indices = np.argsort(evals.real)
            # Reorder the eigenvectors matrix columns using the sort indices of evals
            sorted_eigenvectors = evecs[:, sort_indices]
            first_evecs = sorted_eigenvectors.real[:, :20]
        elif lap_norm == 'symmetrical':
            sqrt_degree_inv = np.linalg.inv(np.sqrt(degree_matrix))
            normalized_laplacian = sqrt_degree_inv @ unnormalized_laplacian @ sqrt_degree_inv
            evals, evecs = eigh(normalized_laplacian)
            sort_indices = np.argsort(evals)
            # Reorder the eigenvectors matrix columns using the sort indices of evals
            sorted_eigenvectors = evecs[:, sort_indices]
            first_evecs = sorted_eigenvectors[:, :20]
        else:
            print('lap_norm can only be random_walk or symmetrical')

        np.save(save_path, first_evecs)
        return first_evecs


    def num_dist_segs(self):
        num_seg_per_anno = []
        for aid in range(self.num_annos()):
            ref_anno = ssdm.multi2openseg(self.ref(mode='normal'))
            segs = []
            for obs in ref_anno:
                segs.append(obs.value)
            num_seg_per_anno.append(len(set(segs)))
        return max(num_seg_per_anno)


class DS(Dataset):
    def __init__(
        self, 
        mode='both',
        infer = True,
        transform = None,
        lap_norm = 'random_walk',
        beat_sync = True,
        sample_select_fn=None
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if mode not in ('rep', 'loc', 'both'):
            raise AssertionError('bad dataset mode, can only be rep or loc both')
        self.mode = mode
        self.infer = infer
        self.sample_select_fn = sample_select_fn
        self.lap_norm = lap_norm
        self.beat_sync = beat_sync

        # sample building
        if self.infer:
            if self.mode == 'both':
                self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES, ssdm.AVAL_FEAT_TYPES))
            else:
                self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))
        else:
            self.labels = sample_select_fn(self.split)
            self.samples = list(self.labels.keys())
        self.samples.sort()
        self.transform=transform
        self.output_dir = self.track_obj().output_dir
        

    def __len__(self):
        return len(self.samples)


    def __repr__(self):
        return f'{self.name}{self.mode}{self.split}'


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tid, *feats = self.samples[idx]
        config = ssdm.DEFAULT_LSD_CONFIG.copy()
        if self.beat_sync:
            config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
        s_info = (tid, *feats, self.mode)
        

        first_evecs = self.track_obj(tid=tid).embedded_rec_mat(
            feat_combo=dict(rep_ftype=feats[0], loc_ftype=feats[1]), 
            lap_norm=self.lap_norm, beat_sync=self.beat_sync,
            recompute=False
        )
        data = torch.tensor(first_evecs, dtype=torch.float32, device=self.device)
        if self.mode != 'both':
            assert KeyError('bad mode: can onpy be both')
        
        nlvl_save_path = os.path.join(self.output_dir, 'evecs/'+s_info[0]+'_nlvl.npy')

        best_nlvl = np.load(nlvl_save_path)


        datum = {'data': data[None, None, :],
                 'info': s_info,
                 'uniq_segs': torch.tensor(best_nlvl, dtype=torch.long, device=self.device)}

        if not self.infer:
            label, nlvl = self.labels[self.samples[idx]]
            datum['label'] = torch.tensor([label], dtype=torch.float32, device=self.device)[None, :]
            datum['best_layer'] = torch.tensor([nlvl], dtype=torch.float32, device=self.device)[None, :]
        
        if self.transform:
            datum = self.transform(datum)
        
        return datum

    def track_obj(self):
        raise NotImplementedError


def delay_embed(
    feat_mat,
    add_noise: bool = False,
    n_steps: int = 1, # param for time_delay_emb
    delay: int = 1, # param for time_delay_emb
) -> np.ndarray:
    if add_noise:
        rng = np.random.default_rng()
        noise = rng.random(feat_mat.shape) * (1e-9)
        feat_mat = feat_mat + noise

    return stack_memory(
        feat_mat, 
        mode='edge', 
        n_steps=n_steps, 
        delay=delay
    )