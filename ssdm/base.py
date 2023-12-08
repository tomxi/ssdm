import ssdm
from . import feature
from .expand_hier import expand_hierarchy
import librosa

import jams
import json
import numpy as np
import xarray as xr
from scipy import spatial, stats
from sklearn.metrics import roc_auc_score

import os
from librosa.segment import recurrence_matrix
from librosa.feature import stack_memory
from librosa import frames_to_time


class Track(object):
    def __init__(
        self,
        tid: str = '',
        dataset_dir: str = '', 
        output_dir: str = '',
        feature_dir: str = '',
    ):
        self.tid = tid
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.feature_dir = feature_dir

        self.title = tid
        self.audio_path = None
        
        self._sr = 22050 # This is fixed
        self._hop_len = 4096 # This is fixed
        self._y = None
        self._jam = None # populate when .jam() is called.
        self._track_ts = None # populate when .ts() is called.


    def jam(self) -> jams.JAMS:
        if self._jam is None:
            jams_path = os.path.join(self.dataset_dir, f'jams/{self.title}.jams')
            self._jam = jams.load(jams_path, validate=False)
        return self._jam

    
    def audio(
        self, 
        sr: float = 22050
    ) -> tuple:
        if self._sr != sr or self._y == None:
            self._y, self._sr = librosa.load(self.audio_path, sr=sr)
        return self._y, self._sr
    
    
    def num_annos(self) -> int:
        return 1
    

    def representation(self, feat_type='mfcc', use_track_ts=True, recompute=False, **delay_emb_kwargs):
        """delay_emb_kwargs: add_noise, n_steps, delay"""
        delay_emb_config = dict(add_noise=False, n_steps=1, delay=1)
        delay_emb_config.update(delay_emb_kwargs)

        # Try loading 
        feature_path = os.path.join(self.feature_dir, f'{self.title}_{feat_type}.npz')
        if recompute or not os.path.exists(feature_path):
            if self.audio_path:
                feature.FEAT_MAP[feat_type](self.audio_path, feature_path)
            else:
                raise LookupError("Track does not have audio!")
        
        npz = np.load(feature_path)

        try:
            fmat = npz['feature']
        except KeyError:
            fmat = npz['pitch']

        if use_track_ts:
            fmat = fmat[:, :len(self.ts())]

        return delay_embed(fmat, **delay_emb_config)


    def ts(self) -> np.array:
        if self._track_ts is None:
            num_frames = np.asarray(
                [self.representation(feat, use_track_ts=False).shape[-1] for feat in ssdm.AVAL_FEAT_TYPES]
            )
            self._track_ts = frames_to_time(
                list(range(np.min(num_frames))), 
                hop_length=feature._HOP_LEN, 
                sr=feature._AUDIO_SR)
        return self._track_ts


    def ref(self, mode='expand', **anno_id_kwarg):
        seg_anns = self.jam().search(namespace='segment_open')

        def fill_out_anno(anno, ts):
            anno_start_time = anno.data[0].time
            anno_end_time = anno.data[-1].time + anno.data[-1].duration

            last_frame_time = ts[-1]
            if anno_start_time != 0:
                anno.append(value='NL', time=0, 
                            duration=anno_start_time, confidence=1
                        )
            if anno_end_time < last_frame_time:
                anno.append(value='NL', time=anno_end_time, 
                            duration=last_frame_time - anno_start_time, confidence=1
                        )
            return anno
        
        anno = fill_out_anno(seg_anns[0], self.ts())
        if mode == 'expand':
            return ssdm.openseg2multi(expand_hierarchy(anno))
        elif mode == 'normal':
            return ssdm.openseg2multi([anno])


    def path_ref(
            self,
            mode: str = 'expand', # {'normal', 'expand'}
            binarize: bool = True,
            **anno_id_kwarg
        ):
        # Get reference annotation
        ref_anno = self.ref(mode, **anno_id_kwarg)
        # Get annotation meet matrix
        anno_meet = ssdm.anno_to_meet(ref_anno, self.ts())
        # Pull out diagonal
        anno_diag = anno_meet.diagonal(1)
        if binarize:
            anno_diag = anno_diag == np.max(anno_diag)
        return anno_diag.astype(int)
    

    def ssm(self, feature='mfcc', distance='cosine', full=False, add_noise=True, n_steps=6, delay=2, width=30, recompute=False): 
        # save
        # npy path
        ssm_info_str = f'n{feature}_{distance}{f"_fw{width}"}_s'
        feat_info_str = f's{n_steps}xd{delay}{"_n" if add_noise else ""}'
        ssm_path = os.path.join(
            self.output_dir, 
            f'ssms/{self.tid}_{ssm_info_str}_{feat_info_str}.npy'
        )

        ssm_dir = os.path.join(self.output_dir, 'ssms')
        if not os.path.exists(ssm_dir):
            os.system(f'mkdir {ssm_dir}')

        # try loading, if can't then set recompute to True
        try:
            with open(ssm_path, 'rb') as f:
                ssm = np.load(ssm_path, allow_pickle=True)
        except:
            recompute=True
        
        # compute
        if recompute:
            # prepare feature matrix
            feat_mat = self.representation(feature, add_noise=add_noise, n_steps=n_steps, delay=delay)
            
            # fix width with short tracks
            feat_max_width = ((feat_mat.shape[-1] - 1) // 2) - 5
            if width >= feat_max_width:
                width=feat_max_width
            
            # sometimes sym=True will give an error related to empty rows or something...
            try:   
                ssm = recurrence_matrix(
                    feat_mat, width=width, metric=distance, mode='affinity', sym=True, full=full,
                )
            except:
                print('setting rec mat sym to False for the following SSM: \n \t' + ssm_path)
                ssm = recurrence_matrix(
                    feat_mat, width=width, metric=distance, mode='affinity', sym=False, full=full,
                )

            # store ssm
            with open(ssm_path, 'wb') as f:
                np.save(f, ssm)

        return ssm
    
    
    def path_sim(self, feature='mfcc', distance='sqeuclidean', add_noise=True, n_steps=6, delay=2, sigma_percentile=95, recompute=False): 
        feat_mat = delay_embed(self.representation(feature), add_noise, n_steps, delay)

        path_sim_info_str = f'pathsim_{feature}_{distance}'
        feat_info_str = f'ns{n_steps}xd{delay}xap{sigma_percentile}{"_n" if add_noise else ""}'
        pathsim_path = os.path.join(
            self.output_dir, 
            f'ssms/{self.tid}_{path_sim_info_str}_{feat_info_str}.npy'
        )

        try:
            with open(pathsim_path, 'rb') as f:
                pathsim = np.load(pathsim_path, allow_pickle=True)

        except:
            recompute = True

        if recompute:
            frames = feat_mat.shape[1]
            path_dist = []
            for i in range(frames-1):
                pair_dist = spatial.distance.pdist(feat_mat[:, i:i + 2].T, metric=distance)
                path_dist.append(pair_dist[0])
            path_dist = np.array(path_dist)
            sigma = np.percentile(path_dist, sigma_percentile)
            pathsim = np.exp(-path_dist / sigma)

            # store pathsim
            with open(pathsim_path, 'wb') as f:
                np.save(f, pathsim)
        
        # read npy file
        with open(pathsim_path, 'rb') as f:
            pathsim = np.load(pathsim_path, allow_pickle=True)
        return pathsim

    
    def lsd(
            self,
            config_update: dict = dict(),
            recompute: bool  = False,
            print_config: bool = False,
            path_sim_sigma_percentile: float = 95,
    ) -> jams.Annotation:
        # load lsd jams and file/log handeling...

        config = ssdm.DEFAULT_LSD_CONFIG.copy()
        config.update(config_update)

        if print_config:
            print(config)

        record_path = os.path.join(self.output_dir, 
                                f'lsd/{self.tid}_{config["rep_ftype"]}_{config["loc_metric"]}_{config["rec_width"]}_{path_sim_sigma_percentile}.jams'
                                )
        
        if not os.path.exists(record_path):
            # create and save jams file
            print('creating new jams file')
            empty_jam = jams.JAMS(file_metadata=self.jam().file_metadata)
            try: 
                empty_jam.save(record_path)
            except FileNotFoundError:
                os.system(f'mkdir {os.path.join(self.output_dir, "lsd")}')

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
            print('computing! -- lsd')
            # genearte new multi_segment annotation and store/overwrite it in the original jams file.
            lsd_anno = ssdm.run_lsd(self, config=config, recompute_ssm=recompute, loc_sigma=path_sim_sigma_percentile)
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
    

    def lsd_score(
        self,
        lsd_sel_dict: dict = dict(),
        recompute: bool = False,
        l_frame_size: float = 0.1,
        path_sim_sigma_percent: float = 95,
        anno_id: int = 0,
    ) -> xr.DataArray:
        #initialize file if doesn't exist or load the xarray nc file
        nc_path = os.path.join(self.output_dir, f'ells/{self.tid}_{l_frame_size}p{path_sim_sigma_percent}rw5.nc')
        if not os.path.exists(nc_path):
            init_grid = ssdm.LSD_SEARCH_GRID.copy()
            init_grid.update(anno_id=range(self.num_annos()), l_type=['lp', 'lr', 'lm'])
            lsd_score_da = ssdm.init_empty_xr(init_grid, name=str(self.tid))
            lsd_score_da.to_netcdf(nc_path)
        else:
            with xr.open_dataarray(nc_path) as lsd_score_da:
                lsd_score_da.load()
        
        # build lsd_configs from lsd_sel_dict
        configs = []
        config_midx = lsd_score_da.sel(**lsd_sel_dict).coords.to_index()
        for option_values in config_midx:
            exp_config = ssdm.DEFAULT_LSD_CONFIG.copy()
            for opt in lsd_sel_dict:
                if opt in ssdm.DEFAULT_LSD_CONFIG:
                    exp_config[opt] = lsd_sel_dict[opt]
            try:
                for opt, value in zip(config_midx.names, option_values):
                    if opt in ssdm.DEFAULT_LSD_CONFIG:
                        exp_config[opt] = value
            except TypeError:
                pass
            if exp_config not in configs:
                configs.append(exp_config)

        # run through the configs list and populate / readout scores
        for lsd_conf in configs:
            # first build the index dict
            coord_idx = lsd_conf.copy()
            coord_idx.update(l_type=['lp', 'lr', 'lm'])
            for k in coord_idx.copy():
                if k not in lsd_score_da.coords:
                    del coord_idx[k]
            
            exp_slice = lsd_score_da.sel(coord_idx)
            if recompute or exp_slice.isnull().any():
                # Only compute if value doesn't exist:
                ###
                # print('recomputing l score')
                proposal = self.lsd(lsd_conf, print_config=False, path_sim_sigma_percentile=path_sim_sigma_percent, recompute=recompute)
                annotation = self.ref(anno_id=anno_id)
                # search l_score from old places first?
                l_score = ssdm.compute_l(proposal, annotation, l_frame_size=l_frame_size)
                with xr.open_dataarray(nc_path) as lsd_score_da:
                    lsd_score_da.load()
                lsd_score_da.loc[coord_idx] = list(l_score)
                lsd_score_da.to_netcdf(nc_path)
        # return lsd_score_da
        return lsd_score_da.sel(**lsd_sel_dict)


    def tau(
        self,
        tau_sel_dict: dict = dict(),
        anno_mode: str = 'expand',
        recompute: bool = False,
        quantize: str = 'percentile', # None, 'kmeans' or 'percentile'
        quant_bins: int = 7, # used for loc tau quant schemes
        rec_width: int = 30,
        eigen_block: bool = True,
        **anno_id_kwarg
    ) -> xr.DataArray:
        # test if record_path exist, if no, set recompute to true.
        suffix = f'_{quantize}{quant_bins}' if quantize is not None else ''
        am_str = f'{anno_mode}' if anno_mode != 'expand' else ''
        eigen_block_txt = f'blocky' if eigen_block else ''
        record_path = os.path.join(self.output_dir, f'taus/{self.tid}_rw{rec_width}{suffix}{am_str}{eigen_block_txt}.nc')
        if not os.path.exists(record_path):
            grid_coords = dict(f_type=ssdm.AVAL_FEAT_TYPES, 
                               tau_type=['rep', 'loc'], 
                               )
            tau = ssdm.init_empty_xr(grid_coords)
            tau.to_netcdf(record_path)
        else:
            with xr.open_dataarray(record_path) as tau:
                tau.load()

        if recompute or tau.sel(**tau_sel_dict).isnull().any():

            # build lsd_configs from tau_sel_dict

            config_midx = tau.sel(**tau_sel_dict).coords.to_index()

            meet_mat = ssdm.anno_to_meet(self.ref(mode=anno_mode, **anno_id_kwarg), self.ts())
            meet_mat_diag = np.diag(meet_mat, k=1)
            # print(config_midx)
            for f_type, tau_type in config_midx:
                if tau_type == 'rep':
                    ssm = self.ssm(
                        feature=f_type, 
                        distance='cosine', 
                        width=rec_width,
                        recompute=recompute,
                        **ssdm.REP_FEAT_CONFIG[f_type]
                    )

                    if eigen_block:
                        combined_graph = ssdm.scluster.combine_ssms(ssm, meet_mat_diag)
                        _, evecs = ssdm.scluster.embed_ssms(combined_graph, evec_smooth=13)
                        first_evecs = evecs[:, :10]
                        quant_block = ssdm.quantize(np.matmul(first_evecs, first_evecs.T), quant_bins=quant_bins)
                        tau.loc[dict(f_type=f_type, tau_type=tau_type)] = stats.kendalltau(
                            quant_block.flatten(), meet_mat.flatten()
                        )[0]

                    else:
                        quant_sim = ssdm.quantize(ssm, quantize_method=quantize, quant_bins=quant_bins)
                        tau.loc[dict(f_type=f_type, tau_type=tau_type)] = stats.kendalltau(
                            quant_sim.flatten(), meet_mat.flatten()
                        )[0]

                else: #path_sim AUC
                    path_sim = self.path_sim(feature=f_type, 
                                             distance='sqeuclidean',
                                             recompute=recompute,
                                             **ssdm.LOC_FEAT_CONFIG[f_type])
                    
                    tau.loc[dict(f_type=f_type, tau_type=tau_type)] = roc_auc_score(self.path_ref(**anno_id_kwarg), path_sim)        
                tau.to_netcdf(record_path)
        # return tau
        return tau.sel(**tau_sel_dict)


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