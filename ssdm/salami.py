import ssdm
from ssdm import base

import torch
from torch.utils.data import Dataset

import os, json, pkg_resources
import librosa, jams
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

from scipy import stats

from ssdm.expand_hier import expand_hierarchy


class Track(base.Track):
    def __init__(
            self, 
            tid: str = '384', 
            dataset_dir: str = '/scratch/qx244/data/salami/', 
            output_dir: str = '/vast/qx244/salami/',
            feature_dir: str = '/scratch/qx244/data/salami/features'
        ):
        super().__init__(tid, dataset_dir=dataset_dir, output_dir=output_dir, feature_dir=feature_dir)
        self.audio_path = os.path.join(dataset_dir, f'audio/{tid}/audio.mp3')


    def num_annos(
        self,
    ) -> int:
        return len(self.jam().search(namespace='segment_salami_upper'))


    def ref(
        self,
        mode: str = 'expand', # {'normal', 'expand', 'refine', 'coarse'},
        anno_id: int = 0,
    ) -> jams.Annotation: 
        """
        A list of `jams.Annotation`s with two modes: {'normal', 'expand'}
        """
        upper_annos = self.jam().search(namespace='segment_salami_upper')
        lower_annos = self.jam().search(namespace='segment_salami_lower')
        if mode == 'normal':

            out_anno = ssdm.openseg2multi([upper_annos[anno_id], lower_annos[anno_id]])
            # multi_anno = jams.Annotation(namespace='multi_segment')
        else:
            upper_expanded = expand_hierarchy(upper_annos[anno_id])
            lower_expanded = expand_hierarchy(lower_annos[anno_id])
            
            if mode == 'expand':
                out_anno = ssdm.openseg2multi(upper_expanded + lower_expanded)
            elif mode == 'refine':
                upper_refined = upper_expanded[-1]
                lower_refined = lower_expanded[-1]
                out_anno = ssdm.openseg2multi([upper_refined, lower_refined])
            elif mode == 'coarse':
                upper_coarsened = upper_expanded[0]
                lower_coarsened = lower_expanded[0]
                out_anno = ssdm.openseg2multi([upper_coarsened, lower_coarsened])
            else:
                raise librosa.ParameterError("mode can only be one of 'normal', 'expand', 'refine', or 'coarse'.")
        return out_anno


    def adobe(
        self,
    ) -> jams.Annotation:
        result_dir = '/scratch/qx244/data/ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/'
        filename = f'{self.tid}.mp3.msdclasscsnmagic.json'

        with open(os.path.join(result_dir, filename), 'rb') as f:
            adobe_hier = json.load(f)

        anno = ssdm.hier_to_multiseg(adobe_hier)
        anno.sandbox.update(mu=0.1, gamma=0.1)
        return anno


    def adobe_l(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        anno_mode: str = 'expand', #see `segmentation_anno`'s `mode`.
        # l_type: str = 'l', # can also be 'lr' and 'lp' for recall and precision.
        l_frame_size = 0.1
    ) -> xr.DataArray:
        record_path = os.path.join(self.output_dir, f'ells/{self.tid}_a{anno_id}_{anno_mode}{l_frame_size}_adobe.pkl')
        # test if record_path exist, if no, set recompute to true.
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            l_score = pd.Series(index=['lp', 'lr', 'l'])
            l_score[:]= ssdm.compute_l(
                self.adobe(), 
                self.ref(mode=anno_mode, anno_id=anno_id),
                l_frame_size=l_frame_size
            )
            # save to record_path
            l_score.to_pickle(record_path)

        # Read from record_path
        pd_series = pd.read_pickle(record_path).astype('float')
        pd_series.index.name = 'l_type'
        return pd_series.to_xarray()


def get_lsd_scores(
    tids=[], 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        track_score = track.lsd_score(**lsd_score_kwargs)
        score_per_track.append(track_score)
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


def get_ids(
    split: str = 'working',
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    """ split can be ['audio', 'jams', 'excluded', 'new_val', 'new_test', 'new_train']
        Dicts sotred in id_path json file.
    """
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(id_path, 'r') as f:
            id_json = json.load(f)
    except FileNotFoundError:
        id_json = dict()
        id_json[split] = []
        with open(id_path, 'w') as f:
            json.dump(id_json, f)
    ids = id_json[split]
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


class DS(Dataset):
    """ split='train',
        mode='rep', # {'rep', 'loc'}
        infer=False,
        drop_features=[],
        precomputed_tau_fp = '/home/qx244/scanning-ssm/ssdm/taus_1107.nc'
    """
    def __init__(self, 
                 split='train',
                 mode='rep', # {'rep', 'loc'}
                 infer=False,
                 drop_features=[],
                 precomputed_tau_fp = '/home/qx244/scanning-ssm/ssdm/taus_1107.nc',
                 transform = None
                ):
        self.transform = transform

        if mode not in ('rep', 'loc', 'both'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.split = split
        # load precomputed taus, and drop feature and select tau type
        taus_full = xr.open_dataarray(precomputed_tau_fp)

        
        self.tau_scores = taus_full.drop_sel(f_type=drop_features).sel(tau_type=mode)
        if mode == 'both':
            pass # TODO, get both scores

        # Get the threshold for upper and lower quartiles, 
        # and use them as positive and negative traning examples respectively
        tau_percentile_flat = stats.percentileofscore(self.tau_scores.values.flatten(), self.tau_scores.values.flatten())
        self.tau_percentile = self.tau_scores.copy(data=tau_percentile_flat.reshape(self.tau_scores.shape))
        self.tau_thresh = [np.percentile(self.tau_scores, 25), np.percentile(self.tau_scores, 75)]
        
        tau_series = self.tau_scores.to_series()
        split_ids = get_ids(split, out_type='list')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.infer = infer
        if infer:
            # just get all the combos, returns the percentile of the sample tau value [0~1]
            self.samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in tau_series.index.to_flat_index().values \
                            if pair[0] in split_ids}
        else:
            # use threshold to prepare training data
            # calculate threshold from all taus
            neg_tau_series = tau_series[tau_series < self.tau_thresh[0]]
            self.all_neg_samples = neg_tau_series.index.to_flat_index().values
            pos_tau_series = tau_series[tau_series > self.tau_thresh[1]]
            self.all_pos_samples = pos_tau_series.index.to_flat_index().values

            # use set to select only the ones in the split
            neg_samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in self.all_neg_samples if pair[0] in split_ids}
            pos_samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in self.all_pos_samples if pair[0] in split_ids}

            self.samples = pos_samples.copy()
            self.samples.update(neg_samples)
        self.ordered_keys = list(self.samples.keys())
        self.tids = list(split_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, feat = self.ordered_keys[idx]
        track = Track(tid)
        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        if self.mode == 'rep':
            rep_ssm = track.ssm(feature=feat, 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feat]
                                )
            tau_percent = self.samples[(tid, feat)]
            sample = {'data': torch.tensor(rep_ssm[None, None, :], dtype=torch.float32, device=self.device),
                    'label': torch.tensor([tau_percent > 0.5], dtype=torch.float32, device=self.device)[None, :],
                    'tau_percent': torch.tensor(tau_percent, dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
        
        elif self.mode == 'loc':
            path_sim = track.path_sim(feature=feat, 
                                      distance=config['loc_metric'],
                                      **ssdm.LOC_FEAT_CONFIG[feat])

            tau_percent = self.samples[(tid, feat)]
            sample = {'data': torch.tensor(path_sim[None, None, :], dtype=torch.float32, device=self.device),
                    'label': torch.tensor([tau_percent > 0.5], dtype=torch.float32)[None, :],
                    'tau_percent': torch.tensor(tau_percent, dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
       
        else:
            assert KeyError

        if self.transform:
            sample = self.transform(sample)

        return sample

