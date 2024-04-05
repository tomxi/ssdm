from ssdm.utils import *

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
    'num_layers': 10,
    'add_noise': True, 
    'n_steps': 3, 
    'delay': 2,
}

BEAT_SYNC_CONFIG_PATCH = {
    "evec_smooth": 8,
    "rec_smooth" : 8,
    "rec_width"  : 2, 
    "num_layers" : 16,
    'add_noise': True, 
    'n_steps': 3, 
    'delay': 1,
}

REP_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'crema': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'tempogram': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'yamnet': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 3, 'delay': 1},
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
