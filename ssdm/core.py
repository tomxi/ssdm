import os, json
import numpy as np
import librosa
import jams
import pandas as pd
import xarray as xr
from scipy import spatial, sparse
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
    'rec_width': 4,
    'rec_smooth': 20, 
    'evec_smooth': 20,
    'rec_full': 0,
    'rep_ftype': 'crema', # grid 5
    'loc_ftype': 'mfcc', # grid 5
    'rep_metric': 'cosine',
    'loc_metric': 'sqeuclidean', # grid 2
    'bandwidth': 'med_k_scalar',
    'hier': True,
    'num_layers': 10
}

REP_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'crema': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'tempogram': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'yamnet': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 3, 'delay': 1},
}

LOC_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'crema': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'tempogram': {'add_noise': True, 'n_steps': 2, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'yamnet': {'add_noise': True, 'n_steps': 2, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 2, 'delay': 1},
}

LSD_SEARCH_GRID = dict(rep_ftype=AVAL_FEAT_TYPES, 
                       loc_ftype=AVAL_FEAT_TYPES, 
                      )

class Track:
    def __init__(
        self, 
        tid: str = '384', 
        salami_dir: str = '/scratch/qx244/data/salami/',
        feature_sr: float = None,
    ):
        self.tid = tid
        self.salami_dir = salami_dir
        self.audio_path = os.path.join(salami_dir, f'audio/{tid}/audio.mp3')
        self._feature_sr = 4096/22050 if feature_sr is None else feature_sr

        self._y = None # populate when .audio() is called.
        self._sr = None # changed when .audio() is called.
        self._jam = None # populate when .jam() is called.
        self._common_ts = None # populate when .ts() is called.


    def audio(
        self, 
        sr: float = 22050
    ) -> tuple:
        if self._sr != sr:
            self._y, self._sr = librosa.load(self.audio_path, sr=sr)
        return self._y, self._sr
   

    def jam(
        self, 
    ) -> jams.JAMS:
        if self._jam is None:
            jams_path = os.path.join(self.salami_dir, f'jams/{self.tid}.jams')
            self._jam = jams.load(jams_path, validate=False)
        return self._jam


    def num_annos(
        self,
    ) -> int:
        return len(self.jam().search(namespace='segment_salami_upper'))


    def ts(
        self
    ) -> np.array:
        if self._common_ts is None:
            num_frames = np.asarray([
                len(ssdm.feature.openl3(self)['ts']),
                len(ssdm.feature.yamnet(self)['ts']),
                len(ssdm.feature.chroma(self)['ts']),
                len(ssdm.feature.crema(self)['ts']),
                len(ssdm.feature.tempogram(self)['ts']),
                len(ssdm.feature.mfcc(self)['ts'])
            ])
            self._common_ts = ssdm.feature.crema(self)['ts'][:np.min(num_frames)]
        return self._common_ts


    def representation(
        self,
        feat_type: str = 'openl3',
        add_noise: bool = False,
        n_steps: int = 1, # param for time_delay_emb
        delay: int = 1, # param for time_delay_emb
    ) -> np.array:
        # THIS HAS SR FIXED AT 22050
        if feat_type == 'openl3':
            feat_npz = ssdm.feature.openl3(self)
        elif feat_type == 'yamnet':
            feat_npz = ssdm.feature.yamnet(self)
        elif feat_type == 'mfcc':
            feat_npz = ssdm.feature.mfcc(self)
        elif feat_type == 'tempogram':
            feat_npz = ssdm.feature.tempogram(self)
        elif feat_type == 'chroma':
            feat_npz = ssdm.feature.chroma(self)
        elif feat_type == 'crema':
            feat_npz = ssdm.feature.crema(self)
        else:
            raise librosa.ParameterError('bad representation name')

        feat_mat = feat_npz['feature'][:len(self.ts())]
        if add_noise:
            rng = np.random.default_rng()
            noise = rng.random(feat_mat.shape) * (1e-9)
            feat_mat = feat_mat + noise

        feat_mat = librosa.feature.stack_memory(
            feat_mat, mode='edge', n_steps=n_steps, delay=delay
        )
        return feat_mat[:, :len(self.ts())]


    def sdm(
        self,
        feature: str = 'mfcc',
        distance: str = 'cosine',
        recompute: bool = False,
        **kwargs, # add_noise, n_steps, delay; for preping features
    ) -> np.array:
        # npy path
        sdm_path = os.path.join(self.salami_dir, f'sdms/{self.tid}_{feature}_{distance}.npy')
        
        # see if desired sdm is already computed
        if not os.path.exists(sdm_path) or recompute:
            # compute sdm
            feat_mat = self.representation(feat_type=feature, **kwargs)
            dmat = spatial.distance.squareform(
                spatial.distance.pdist(feat_mat.T, distance)
            )
            # store sdm
            with open(sdm_path, 'wb') as f:
                np.save(f, dmat)
        
        # read npy file
        with open(sdm_path, 'rb') as f:
            dmat = np.load(sdm_path, allow_pickle=True)
        return dmat
    

    def ssm(
        self,
        feature: str = 'mfcc',
        distance: str = 'cosine',
        width = 4, # width param for librosa.segment.rec_mat <= 1
        bw: str = 'med_k_scalar', # one of {'med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair'}
        full: bool = False,
        add_noise: bool = False, # padding representation with a little noise to avoid divide by 0 somewhere...
        n_steps: int = 1, # Param for time delay embedding of representation
        delay: int = 1, # Param for time delay embedding of representation
        recompute: bool = False,
    ) -> np.array:
        full = bool(full)
        # npy path
        ssm_info_str = f'n{feature}_{distance}{f"_fw{width}"}{f"_f" if full else "_s"}'
        feat_info_str = f's{n_steps}xd{delay}{"_n" if add_noise else ""}'
        ssm_path = os.path.join(
            self.salami_dir, 
            f'ssms/{self.tid}_{ssm_info_str}_{feat_info_str}_{bw}.npy'
        )
        # print(ssm_path)
        
        # see if desired ssm is already computed
        if not os.path.exists(ssm_path):
            recompute = True

        try:
            with open(ssm_path, 'rb') as f:
                ssm = np.load(ssm_path, allow_pickle=True)
        except:
            recompute=True

        if recompute:
            # compute ssm
            feat_mat = self.representation(
                feat_type=feature, 
                add_noise=add_noise,
                n_steps=n_steps,
                delay=delay,
            )

            # fix width with short tracks
            feat_max_width = ((feat_mat.shape[-1] - 1) // 2) - 5
            if width >= feat_max_width:
                width=feat_max_width
            try:
                ssm = librosa.segment.recurrence_matrix(
                    feat_mat, 
                    mode='affinity',
                    width=width,
                    sym=True,
                    full=full,
                    bandwidth=bw,
                )
            except:
                ssm = librosa.segment.recurrence_matrix(
                    feat_mat, 
                    mode='affinity',
                    width=width,
                    sym=False,
                    full=full,
                    bandwidth=bw,
                )

            # store ssm
            with open(ssm_path, 'wb') as f:
                np.save(f, ssm)
        
        # read npy file
        with open(ssm_path, 'rb') as f:
            ssm = np.load(ssm_path, allow_pickle=True)

        return ssm


    def path_sim(
            self,
            feature: str = 'mfcc',
            distance: str = 'sqeuclidean',
            add_noise: bool = False, # padding representation with a little noise to avoid divide by 0 somewhere...
            n_steps: int = 1, # Param for time delay embedding of representation
            delay: int = 1, # Param for time delay embedding of representation
            aff_kernel_sigma_percentile = 85, 
            recompute = False,
            width: int = 1, # TODO Nice to have if things still doesn't work.
        ) -> np.array:
        feat_mat = self.representation(
            feat_type=feature, 
            add_noise=add_noise,
            n_steps=n_steps,
            delay=delay,
        )


        path_sim_info_str = f'pathsim_{feature}_{distance}'
        feat_info_str = f'ns{n_steps}xd{delay}xap{aff_kernel_sigma_percentile}{"_n" if add_noise else ""}'
        pathsim_path = os.path.join(
            self.salami_dir, 
            f'ssms/{self.tid}_{path_sim_info_str}_{feat_info_str}.npy'
        )

        if not os.path.exists(pathsim_path):
            recompute = True

        if recompute:
            frames = feat_mat.shape[1]
            path_dist = []
            for i in range(frames-1):
                pair_dist = spatial.distance.pdist(feat_mat[:, i:i + 2].T, metric=distance)
                path_dist.append(pair_dist[0])
            path_dist = np.array(path_dist)
            sigma = np.percentile(path_dist, aff_kernel_sigma_percentile)
            pathsim = np.exp(-path_dist / sigma)

            # store pathsim
            with open(pathsim_path, 'wb') as f:
                np.save(f, pathsim)
        
        # read npy file
        with open(pathsim_path, 'rb') as f:
            pathsim = np.load(pathsim_path, allow_pickle=True)
        return pathsim
        
    # TODO add anno mode
    def tau(
        self,
        tau_sel_dict: dict = dict(),
        anno_id: int = 0,
        anno_mode: str = 'expand',
        recompute: bool = False,
        quantize: str = 'percentile', # None, 'kmeans' or 'percentile'
        quant_bins: int = 8, # used for loc tau quant schemes
        quant_bins_loc: int = 8,
        aff_kernel_sigma_percentile=85, # used for self.path_sim
        rec_width: int = 16,
    ) -> xr.DataArray:
        # test if record_path exist, if no, set recompute to true.
        suffix = f'_{quantize}{quant_bins}_{quant_bins_loc}' if quantize is not None else ''
        am_str = f'{anno_mode}' if anno_mode != 'expand' else ''
        record_path = os.path.join(self.salami_dir, f'taus/{self.tid}_rw{rec_width}a{anno_id}{suffix}{am_str}.nc')
        if not os.path.exists(record_path):
            recompute = True
        else:
            tau = xr.open_dataarray(record_path)

        if recompute or tau.sel(**tau_sel_dict).isnull().any():
            grid_coords = dict(f_type=AVAL_FEAT_TYPES, 
                               tau_type=['rep', 'loc'], 
                               )
            tau = init_empty_xr(grid_coords)

            # build lsd_configs from tau_sel_dict
            config_midx = tau.sel(**tau_sel_dict).coords.to_index()

            meet_mat = anno_to_meet(self.ref(mode=anno_mode, anno_id=anno_id), self.ts())

            # print(config_midx)
            for f_type, tau_type in config_midx:
                if tau_type == 'rep':
                    ssm = self.ssm(
                        feature=f_type, 
                        distance='cosine', 
                        width=rec_width,
                        recompute=recompute,
                        **REP_FEAT_CONFIG[f_type]
                    )

                    quant_sim = ssdm.quantize(ssm, quantize_method=quantize, quant_bins=quant_bins)
                    tau.loc[dict(f_type=f_type, tau_type=tau_type)] = stats.kendalltau(
                        quant_sim.flatten(), meet_mat.flatten()
                    )[0]

                else: #path_sim
                    path_sim = self.path_sim(feature=f_type, 
                                             distance='sqeuclidean',
                                             aff_kernel_sigma_percentile=aff_kernel_sigma_percentile,
                                             recompute=recompute,
                                             **LOC_FEAT_CONFIG[f_type])

                    quant_sim_diag = ssdm.quantize(path_sim, quantize_method=quantize, quant_bins=quant_bins)
                    meet_mat_diag = np.diag(meet_mat, k=1)
                    
                    tau.loc[dict(f_type=f_type, tau_type=tau_type)] = stats.kendalltau(
                        quant_sim_diag, meet_mat_diag
                    )[0]
                    
                tau.to_netcdf(record_path)
        # return tau
        return tau.sel(**tau_sel_dict)


    ############ RETURES JAMS.ANNOTATIONS BELOW
    ## TODO add option for fast and loose? 3 times less res, and add flag for record_path
    def lsd(
            self,
            config_update: dict = dict(),
            recompute: bool  = False,
            print_config: bool = False,
            path_sim_sigma_percentile: float = 85,
    ) -> jams.Annotation:
        # load lsd jams and file/log handeling...

        config = DEFAULT_LSD_CONFIG.copy()
        config.update(config_update)

        if print_config:
            print(config)

        record_path = os.path.join('/vast/qx244/lsd/', 
                                   f'{self.tid}_{config["rep_ftype"]}_{config["loc_metric"]}_{config["rec_width"]}.jams'
                                  )
        
        if not os.path.exists(record_path):
            # create and save jams file
            print('creating new jams file')
            empty_jam = jams.JAMS(file_metadata=self.jam().file_metadata)
            empty_jam.save(record_path)

        # print('loading lsd_jam')
        try:
            lsd_jam = jams.load(record_path)
        except json.decoder.JSONDecodeError:
            lsd_jam = jams.JAMS(file_metadata=self.jam().file_metadata)
            lsd_jam.save(record_path)
        # print('done')
        
        # search if config already stored in lsd_jam
        config_sb = jams.Sandbox()
        config_sb.update(**config)
        # print('trying to find...')
        lsd_annos = lsd_jam.search(sandbox=config_sb)

        # check if multi_anno with config exist, if no, recompute = True
        if len(lsd_annos) == 0:
            # print('multi_anno not found')
            recompute = True
        else:
            # print(f'found {len(lsd_annos)} multi_annos!')
            lsd_anno = lsd_annos[0]
    
        if recompute:
            print('recomputing! -- lsd')
            # genearte new multi_segment annotation and store/overwrite it in the original jams file.
            lsd_anno = _run_lsd(self, config=config, recompute_ssm=recompute, loc_sigma=path_sim_sigma_percentile)
            print('Done! -- lsd')
            # update jams file
            # print('updating jams! -- lsd')
            lsd_anno.sandbox=config_sb
            for old_anno in lsd_annos:
                lsd_jam.annotations.remove(old_anno)
            lsd_jam.annotations.append(lsd_anno)
            lsd_jam.save(record_path)
            # print('Done! -- lsd')
        
        return lsd_anno


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

            out_anno = openseg2multi([upper_annos[anno_id], lower_annos[anno_id]])
            # multi_anno = jams.Annotation(namespace='multi_segment')
        else:
            upper_expanded = ssdm.expand_hierarchy(upper_annos[anno_id])
            lower_expanded = ssdm.expand_hierarchy(lower_annos[anno_id])
            
            if mode == 'expand':
                out_anno = openseg2multi(upper_expanded + lower_expanded)
            elif mode == 'refine':
                upper_refined = upper_expanded[-1]
                lower_refined = lower_expanded[-1]
                out_anno = openseg2multi([upper_refined, lower_refined])
            elif mode == 'coarse':
                upper_coarsened = upper_expanded[0]
                lower_coarsened = lower_expanded[0]
                out_anno = openseg2multi([upper_coarsened, lower_coarsened])
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

        anno = hier_to_multiseg(adobe_hier)
        anno.sandbox.update(mu=0.1, gamma=0.1)
        return anno


    ############# RETURNS l-scores in xr.DataArray
    def lsd_score(
        self,
        lsd_sel_dict: dict = dict(),
        anno_id: int = 0,
        recompute: bool = False,
        l_frame_size: float = 0.1,
        path_sim_sigma_percent: float = 85,
    ) -> xr.DataArray:
        #initialize file if doesn't exist or load the xarray nc file

        # TODO this line... make this a fast and loose flag
        nc_path = os.path.join(self.salami_dir, f'ells/{self.tid}_{l_frame_size}p{path_sim_sigma_percent}rw4.nc')
        if not os.path.exists(nc_path):
            init_grid = LSD_SEARCH_GRID.copy()
            init_grid.update(anno_id=range(self.num_annos()), l_type=['lp', 'lr', 'lm'])
            lsd_score_da = init_empty_xr(init_grid, name=self.tid)
            lsd_score_da.to_netcdf(nc_path)

        lsd_score_da = xr.open_dataarray(nc_path)
        
        # build lsd_configs from lsd_sel_dict
        configs = []
        config_midx = lsd_score_da.sel(**lsd_sel_dict).coords.to_index()
        for option_values in config_midx:
            exp_config = DEFAULT_LSD_CONFIG.copy()
            for opt in lsd_sel_dict:
                if opt in DEFAULT_LSD_CONFIG:
                    exp_config[opt] = lsd_sel_dict[opt]
            try:
                for opt, value in zip(config_midx.names, option_values):
                    if opt in DEFAULT_LSD_CONFIG:
                        exp_config[opt] = value
            except TypeError:
                pass
            if exp_config not in configs:
                configs.append(exp_config)

        # run through the configs list and populate / readout scores
        for lsd_conf in configs:
            # first build the index dict
            coord_idx = lsd_conf.copy()
            coord_idx.update(anno_id=anno_id, l_type=['lp', 'lr', 'lm'])
            for k in coord_idx.copy():
                if k not in lsd_score_da.coords:
                    del coord_idx[k]
            
            exp_slice = lsd_score_da.sel(coord_idx)
            if recompute or exp_slice.isnull().any():
                # Only compute if value doesn't exist:
                ###
                # print('recomputing l score')
                proposal = self.lsd(lsd_conf, print_config=False, path_sim_sigma_percentile=path_sim_sigma_percent, recompute=recompute)
                # cleaned_proposal = clean_anno(proposal, section_fusion_min_dur)
                annotation = self.ref(anno_id=anno_id)
                # search l_score from old places first?
                l_score = compute_l(proposal, annotation, l_frame_size=l_frame_size)
                lsd_score_da.loc[coord_idx] = list(l_score)
                # update the file with 10% chance:
                # if np.random.rand() > 0.1:
                #     lsd_score_da.to_netcdf(nc_path)

        lsd_score_da.to_netcdf(nc_path)
        # return lsd_score_da
        return lsd_score_da.sel(anno_id=anno_id, **lsd_sel_dict)


    def adobe_l(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        anno_mode: str = 'expand', #see `segmentation_anno`'s `mode`.
        # l_type: str = 'l', # can also be 'lr' and 'lp' for recall and precision.
        l_frame_size = 0.1
    ) -> xr.DataArray:
        record_path = os.path.join(self.salami_dir, f'ells/{self.tid}_a{anno_id}_{anno_mode}{l_frame_size}_adobe.pkl')
        # test if record_path exist, if no, set recompute to true.
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            l_score = pd.Series(index=['lp', 'lr', 'l'])
            l_score[:]= compute_l(
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



# HELPER functin to run laplacean spectral decomposition TODO this is where the rescaling happens
def _run_lsd(
    track, 
    config: dict, 
    recompute_ssm: bool = False,
    loc_sigma: float = 85
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
                              aff_kernel_sigma_percentile=loc_sigma,
                              **LOC_FEAT_CONFIG[config['loc_ftype']]
                             )
    
    # Spectral Clustering with Config
    est_bdry_idxs, est_sgmt_labels = sc.do_segmentation_ssm(rep_ssm, path_sim, config)
    est_bdry_itvls = [sc.times_to_intervals(track.ts()[lvl]) for lvl in est_bdry_idxs]
    return mireval_to_multiseg(est_bdry_itvls, est_sgmt_labels)