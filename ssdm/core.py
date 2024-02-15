import os, json
import numpy as np
import librosa
import jams
import pandas as pd
import xarray as xr
from scipy import spatial, sparse, stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import ssdm
from ssdm.utils import *
import ssdm.scluster as sc

# import matplotlib
# import matplotlib.pyplot as plt

AVAL_FEAT_TYPES = ['crema', 'tempogram', 'mfcc', 'yamnet', 'openl3']
AVAL_DIST_TYPES = ['cosine', 'sqeuclidean']
AVAL_BW_TYPES = ['med_k_scalar', 'gmean_k_avg']
DEFAULT_LSD_CONFIG = {
    'rec_width': 5,
    'rec_smooth': 20, 
    'evec_smooth': 20,
    'rec_full': 0,
    'rep_ftype': 'crema', # grid 5
    'loc_ftype': 'mfcc', # grid 5
    'rep_metric': 'cosine',
    'loc_metric': 'sqeuclidean',
    'bandwidth': 'med_k_scalar',
    'hier': True,
    'num_layers': 10
}

REP_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'crema': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'tempogram': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'mfcc': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'yamnet': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'openl3': {'add_noise': True, 'n_steps': 6, 'delay': 2},
}

LOC_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'crema': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'tempogram': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'yamnet': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 3, 'delay': 1},
}

LSD_SEARCH_GRID = dict(rep_ftype=AVAL_FEAT_TYPES, 
                       loc_ftype=AVAL_FEAT_TYPES, 
                      )


# HELPER functin to run laplacean spectral decomposition TODO this is where the rescaling happens
def run_lsd(
    track, 
    config: dict, 
    recompute_ssm: bool = False,
    loc_sigma: float = 95
) -> jams.Annotation:
    def mask_diag(sq_mat, width=13):
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
                        **REP_FEAT_CONFIG[config['rep_ftype']]
                        )
    if config['rec_full']:
        rep_ssm = mask_diag(rep_ssm, width=config['rec_width'])

    # track.path_sim alwasy recomputes.
    path_sim = track.path_sim(feature=config['loc_ftype'],
                              distance=config['loc_metric'],
                              sigma_percentile=loc_sigma,
                              **LOC_FEAT_CONFIG[config['loc_ftype']]
                             )
    
    # Spectral Clustering with Config
    est_bdry_idxs, est_sgmt_labels = sc.do_segmentation_ssm(rep_ssm, path_sim, config)
    est_bdry_itvls = [sc.times_to_intervals(track.ts()[lvl]) for lvl in est_bdry_idxs]
    return mireval_to_multiseg(est_bdry_itvls, est_sgmt_labels)
