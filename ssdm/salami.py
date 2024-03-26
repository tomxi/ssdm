import ssdm
from ssdm import base
from ssdm import scluster

import torch
from torch.utils.data import Dataset

import os, json, pkg_resources, itertools
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

        anno = ssdm.hier2multi(adobe_hier)
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


# def get_lsd_scores(
#     tids=[], 
#     **lsd_score_kwargs
# ) -> xr.DataArray:
#     score_per_track = []
#     for tid in tqdm(tids):
#         track = Track(tid)
#         track_score = track.lsd_score(**lsd_score_kwargs)
#         score_per_track.append(track_score)
    
#     return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


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
    """ split='working',
        mode='rep', # {'rep', 'loc'}
        infer=False,
        drop_features=[],
        precomputed_tau_fp = '/home/qx244/scanning-ssm/ssdm/taus_1107.nc'
    """
    def __init__(self, 
                 split='working',
                 tids=[],
                 mode='rep', # {'rep', 'loc', 'both'}
                 infer=True,
                #  drop_features=[],
                #  precomputed_tau_fp='/home/qx244/scanning-ssm/ssdm/taus_1107.nc',
                 transform=None,
                 sample_select_fn=ssdm.utils.select_samples_using_outstanding_l_score,
                ):
        self.transform = transform
        if mode not in ('rep', 'loc', 'both'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.split = split
        if not tids:
            self.tids = get_ids(self.split, out_type='list')
        else:
            self.tids=tids

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.infer = infer
        if infer:
            # just get all the combos, returns the percentile of the sample tau value [0~1]
            if self.mode != 'both':
                self.samples = {pair: 0.5 \
                                for pair in list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))}
            else:
                self.samples = {pair: 0.5 \
                                for pair in list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES, ssdm.AVAL_FEAT_TYPES))}
        else:
            self.samples = sample_select_fn(self)
        self.ordered_keys = list(self.samples.keys())

    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)
    
    def __repr__(self):
        return 'salami' + self.mode + self.split
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, *feats = self.ordered_keys[idx]
        s_info = (tid, *feats, self.mode)
        track = Track(tid)
        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        if self.mode == 'rep':
            rep_ssm = track.ssm(feature=feats[0], 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feats[0]]
                                )
            data = torch.tensor(rep_ssm, dtype=torch.float32, device=self.device)
            
        
        elif self.mode == 'loc':
            path_sim = track.path_sim(feature=feats[0], 
                                      distance=config['loc_metric'],
                                      **ssdm.LOC_FEAT_CONFIG[feats[0]])
            data = torch.tensor(path_sim, dtype=torch.float32, device=self.device)
            
        elif self.mode == 'both':
            save_path = os.path.join(self.track_obj().output_dir, 'evecs/'+'_'.join(s_info)+'.pt')
            # Try to see if it's already calculated. if so load:
            try:
                first_evecs = torch.load(save_path, map_location=self.device) # load
            # else: calculate
            except:
                # print(save_path)
                lsd_config = dict(rep_ftype=feats[0], loc_ftype=feats[1])
                rec_mat = torch.tensor(track.combined_rec_mat(config_update=lsd_config), dtype=torch.float32, device=self.device)
                # compute normalized laplacian
                with torch.no_grad():
                    rec_mat += 1e-30 # makes inverses nice...
                    # Compute the degree matrix
                    degree_matrix = torch.diag(torch.sum(rec_mat, dim=1))
                    unnormalized_laplacian = degree_matrix - rec_mat
                    # Compute the normalized Laplacian matrix
                    degree_inv = torch.inverse(degree_matrix)
                    normalized_laplacian = degree_inv @ unnormalized_laplacian

                    evals, evecs = torch.linalg.eig(normalized_laplacian)
                    first_evecs = evecs.real[:, :20].to(torch.float32)
                    torch.save(first_evecs, save_path)
            data = first_evecs.to(torch.float32).to(self.device)
        
        label = self.samples[(tid, *feats)]
        best_nlvl = min(track.num_dist_segs() - 1, 10)
        sample = {'data': data[None, None, :],
                  'label': torch.tensor([label], dtype=torch.float32, device=self.device)[None, :],
                  'info': s_info,
                  'uniq_segs': torch.tensor([best_nlvl], dtype=torch.long, device=self.device),
                 }
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_adobe_scores(
    tids=[],
    anno_col_fn=lambda stack: stack.max(dim='anno_id'),
    l_frame_size=0.1
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.adobe_l(anno_id=anno_id, l_frame_size=l_frame_size))

        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)

    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


# MOVE TO SALAMI
# add new splits to split_ids.json
def update_split_json(split_name='', split_idx=[]):
    # add new splits to split_id.json file at json_path
    # read from json and get dict
    json_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(json_path, 'r') as f:
            split_dict = json.load(f)
    except FileNotFoundError:
        split_dict = dict()
        split_dict[split_name] = split_idx
        with open(json_path, 'w') as f:
            json.dump(split_dict, f)
        return split_dict

    # add new split to dict
    split_dict[split_name] = split_idx

    # save json again
    with open(json_path, 'w') as f:
        json.dump(split_dict, f)

    with open(json_path, 'r') as f:
        return json.load(f)


class NewDS(base.DS):
    def __init__(self, mode='rep', infer=False, 
                 split='train', tids=None, transform=None,
                 sample_select_fn=ssdm.select_samples_using_outstanding_l_score):
        self.name = 'slm'

        if tids is None:
            self.tids = get_ids(split=split, out_type='list')
            self.split = split
        else:
            self.tids = tids
            self.split = f'custom{len(tids)}'
        
        super().__init__(mode=mode, infer=infer, sample_select_fn=sample_select_fn, transform=transform)
    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)