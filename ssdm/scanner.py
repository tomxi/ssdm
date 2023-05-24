import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import librosa
import ssdm

from tqdm import tqdm
import numpy as np
import pandas as pd
        

class SalamiDataset(Dataset):
    def __init__(self, 
                 split='working',
                 rep_thresh=[0.07137794553805293, 0.1495658948917135],
                 loc_thresh=[0.00449933048453631, 0.07604734484361576],
                ):
        self.rep, self.loc = ssdm.collate_tau(salami_split=split, anno_merge_fn=np.max)
        self.track_ids = ssdm.get_ids(split=split, out_type='list')
        if rep_thresh is None:
            self.rep_thresh = [np.percentile(self.rep, 25), np.percentile(self.rep, 75)]
        else:
            self.rep_thresh = rep_thresh
        if loc_thresh is None:
            self.loc_thresh = [np.percentile(self.loc, 25), np.percentile(self.loc, 75)]
        else:
            self.loc_thresh = loc_thresh

        self.rl_labels = {}
        for tid in self.track_ids:
            for feat in ssdm.AVAL_FEAT_TYPES:
                # generate labels 0 for bad, 1 for good, 0.5 for meh.
                rep_neg = self.rep.loc[tid][feat] < self.rep_thresh[0]
                rep_pos = self.rep.loc[tid][feat] > self.rep_thresh[1]
                loc_neg = self.loc.loc[tid][feat] < self.loc_thresh[0]
                loc_pos = self.loc.loc[tid][feat] > self.loc_thresh[1]

                rep_label = (1 - int(rep_neg) + int(rep_pos)) / 2
                loc_label = (1 - int(loc_neg) + int(loc_pos)) / 2

                # only append to list of samples when both rep and loc labels are not zero.
                if not (rep_label == 0.5 and loc_label == 0.5):
                    self.rl_labels[f'{tid}_{feat}'] = (np.asarray([rep_label]), np.asarray([loc_label]))
        
        # store keys as a list for indexing
        self.rl_samples = list(self.rl_labels.keys())


    def __len__(self):
        return len(self.rl_samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_key = self.rl_samples[idx]
        tid, feat = sample_key.split('_')
        r_lab, l_lab = self.rl_labels[sample_key]

        track = ssdm.Track(tid)
        config = ssdm.DEFAULT_LSD_CONFIG
        rep_ssm = track.ssm(feature=feat, 
                            distance=config['rep_metric'],
                            width=config['rec_width'],
                            full=config['rec_full'],
                            **ssdm.REP_FEAT_CONFIG[feat]
                            )
        path_sim = track.path_sim(feature=feat,
                                  distance=config['loc_metric'],
                                  **ssdm.LOC_FEAT_CONFIG[feat]
                                 )

        return {
            'ssm': torch.tensor(rep_ssm[None, None, :], dtype=torch.float32),
            'path': torch.tensor(path_sim[None, None, :], dtype=torch.float32),
            'rep_label': torch.tensor(r_lab[None, :], dtype=torch.float32),
            'loc_label': torch.tensor(l_lab[None, :], dtype=torch.float32),
            'info': sample_key,
        }