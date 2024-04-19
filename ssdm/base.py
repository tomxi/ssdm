import ssdm
# import ssdm.utils
from ssdm import scluster
from . import feature
from .expand_hier import expand_hierarchy
import librosa
import madmom

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
    

    def _madmom_beats(self, recompute=False):
        """
        https://github.com/morgan76/Triplet_Mining/blob/189f897a55b0ae14d8fc9417accad27266d6c94e/features.py#L41C1-L50C17
        """
        # add caching
        save_path = os.path.join(self.feature_dir, f'{self.tid}_madmom_beat.npy')
        if not recompute:
            try:
                return np.load(save_path)
            except:
                recompute = True
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(self.audio_path)
        beat_times = np.asarray(proc(act))
        if beat_times[0] > 0:
            beat_times = np.insert(beat_times, 0, 0)
        if beat_times[-1] < self.ts(mode='frame')[-1]:
            beat_times = np.append(beat_times, self.ts(mode='frame')[-1])
        np.save(save_path, beat_times)
        return beat_times



    def representation(self, feat_type='mfcc', use_track_ts=True, recompute=False, beat_sync=False, **delay_emb_kwargs):
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

        if not beat_sync and use_track_ts:
            fmat = fmat[:, :len(self.ts())]

        if beat_sync:
            beat_frames = librosa.time_to_frames(self.ts(mode='beat', pad=True), sr=1 / self.ts(mode='frame')[1], hop_length=1)
            fmat = librosa.util.sync(fmat, beat_frames, aggregate=np.mean)
            if use_track_ts:
                fmat = fmat[:, :len(self.ts('beat'))]

        return delay_embed(fmat, **delay_emb_config)


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


    def ref(self, mode='expand', ts_mode='frame', **anno_id_kwarg):
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
        
        anno = fill_out_anno(seg_anns[0], self.ts(mode=ts_mode))
        if mode == 'expand':
            return ssdm.openseg2multi(expand_hierarchy(anno))
        elif mode == 'normal':
            return ssdm.openseg2multi([anno])


    def path_ref(
            self,
            mode: str = 'expand', # {'normal', 'expand'}
            binarize: bool = True,
            ts_mode:str = 'frame',
            **anno_id_kwarg
        ):
        # Get reference annotation
        ref_anno = self.ref(mode, **anno_id_kwarg)
        # Get annotation meet matrix
        anno_meet = ssdm.anno_to_meet(ref_anno, self.ts(mode=ts_mode))
        # Pull out diagonal
        anno_diag = anno_meet.diagonal(1)
        if binarize:
            anno_diag = anno_diag == np.max(anno_diag)
        return anno_diag.astype(int)
    

    def ssm(self, feature='mfcc', distance='cosine', full=False, add_noise=True, n_steps=6, delay=2, width=30, recompute=False, beat_sync=False): 
        # save
        # npy path
        ssm_info_str = f'n{feature}_{distance}{f"_fw{width}"}_s'
        feat_info_str = f's{n_steps}xd{delay}{"_n" if add_noise else ""}{"_bsync" if beat_sync else ""}'
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
            recompute = True
        
        # compute
        if recompute:
            # prepare feature matrix
            feat_mat = self.representation(feature, beat_sync=beat_sync, add_noise=add_noise, n_steps=n_steps, delay=delay)
            # fix width with short tracks
            feat_max_width = ((feat_mat.shape[-1] - 1) // 2) - 2
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
    
    
    def path_sim(self, feature='mfcc', distance='sqeuclidean', add_noise=True, n_steps=6, delay=2, sigma_percentile=95, recompute=False, beat_sync=False): 
        path_sim_info_str = f'pathsim_{feature}_{distance}'
        feat_info_str = f'ns{n_steps}xd{delay}xap{sigma_percentile}{"_n" if add_noise else ""}{"_bsync" if beat_sync else ""}'
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
            feat_mat = delay_embed(self.representation(feature, beat_sync=beat_sync), add_noise, n_steps, delay)
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

    
    def lsd(
        self,
        config_update: dict = dict(),
        recompute: bool  = False,
        print_config: bool = False,
        beat_sync: bool = False,
        path_sim_sigma_percentile: float = 95,
    ) -> jams.Annotation:
        # load lsd jams and file/log handeling...
        config = ssdm.DEFAULT_LSD_CONFIG.copy()
        if beat_sync:
            config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
        config.update(config_update)

        if print_config:
            print(config)

        record_path = os.path.join(self.output_dir, 
                                f'lsd/{self.tid}_{config["rep_ftype"]}_{config["loc_metric"]}_{config["rec_width"]}_{path_sim_sigma_percentile}{"_bsync" if beat_sync else ""}.jams'
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
            print('computing! -- lsd', self.tid)
            # genearte new multi_segment annotation and store/overwrite it in the original jams file.
            lsd_anno = ssdm.utils.run_lsd(self, config=config, recompute_ssm=recompute, beat_sync=beat_sync, loc_sigma=path_sim_sigma_percentile)
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
    

    def new_lsd_score(self, anno_id=0, l_frame_size=0.5, beat_sync=True, path_sim_bw=95, recompute=False):
        # save 
        if anno_id != 0:
            anno_flag = f'a{anno_id}'
        else:
            anno_flag = ''
        nc_path = os.path.join(
            self.output_dir, 
            f'ells/{self.tid}_{anno_flag}_{l_frame_size}p{path_sim_bw}{"_bsync" if beat_sync else ""}.nc'
        )

        if not recompute:
            try:
                return xr.open_dataarray(nc_path)
            except:
                recompute = True
        
        # build da to store
        score_dims = dict(
            rep_ftype=ssdm.AVAL_FEAT_TYPES,
            loc_ftype=ssdm.AVAL_FEAT_TYPES,
            layer=[x+1 for x in range(16)],
            m_type=['p', 'r', 'f'],
        )

        score_da = xr.DataArray(
            data=np.nan,  # Initialize the data with NaNs
            coords=score_dims,
            dims=list(score_dims.keys()),
            name=str(self.tid)
        )

        # build all the lsd configs:
        indexer = itertools.product(score_dims['rep_ftype'], score_dims['loc_ftype'])

        for rep_ftype, loc_ftype in indexer:
            feature_pair = dict(rep_ftype=rep_ftype, loc_ftype=loc_ftype)
            lsd_config = ssdm.DEFAULT_LSD_CONFIG.copy()
            lsd_config.update(feature_pair)
            if beat_sync:
                lsd_config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)

            for layer in score_dims['layer']:
                score_da.loc[feature_pair].loc[layer] = list(ssdm.compute_l(
                    self.lsd(lsd_config, beat_sync=beat_sync, recompute=False), 
                    self.ref(mode='normal', anno_id=anno_id),
                    l_frame_size=l_frame_size, nlvl=layer
                ))
        
        try:
            score_da.to_netcdf(nc_path)
        except PermissionError:
            os.system(f'rm {nc_path}')
            score_da.to_netcdf(nc_path)
        return score_da


    # def lsd_score(
    #     self,
    #     lsd_sel_dict: dict = dict(),
    #     recompute: bool = False,
    #     l_frame_size: float = 0.1,
    #     beat_sync: bool = False,
    #     path_sim_sigma_percent: float = 95,
    #     anno_id: int = 0,
    # ) -> xr.DataArray:
    #     #initialize file if doesn't exist or load the xarray nc file
    #     nc_path = os.path.join(self.output_dir, f'ells/{self.tid}_{l_frame_size}p{path_sim_sigma_percent}{"_bsync2" if beat_sync else ""}.nc')
    #     if not os.path.exists(nc_path):
    #         init_grid = ssdm.LSD_SEARCH_GRID.copy()
    #         init_grid.update(anno_id=range(self.num_annos()), l_type=['lp', 'lr', 'lm'])
    #         lsd_score_da = xr.DataArray(np.nan, dims=init_grid.keys(), coords=init_grid, name=str(self.tid))
    #         lsd_score_da.to_netcdf(nc_path)
    #     else:
    #         with xr.open_dataarray(nc_path) as lsd_score_da:
    #             lsd_score_da.load()
        
    #     # build lsd_configs from lsd_sel_dict
    #     configs = []
    #     config_midx = lsd_score_da.sel(**lsd_sel_dict).coords.to_index()
    #     for option_values in config_midx:
    #         exp_config = ssdm.DEFAULT_LSD_CONFIG.copy()
    #         if beat_sync:
    #             exp_config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
    #         for opt in lsd_sel_dict:
    #             if opt in ssdm.DEFAULT_LSD_CONFIG:
    #                 exp_config[opt] = lsd_sel_dict[opt]
    #         try:
    #             for opt, value in zip(config_midx.names, option_values):
    #                 if opt in ssdm.DEFAULT_LSD_CONFIG:
    #                     exp_config[opt] = value
    #         except TypeError:
    #             pass
    #         if exp_config not in configs:
    #             configs.append(exp_config)

    #     # run through the configs list and populate / readout scores
    #     for lsd_conf in configs:
    #         # first build the index dict
    #         coord_idx = lsd_conf.copy()
    #         coord_idx.update(l_type=['lp', 'lr', 'lm'])
    #         for k in coord_idx.copy():
    #             if k not in lsd_score_da.coords:
    #                 del coord_idx[k]
            
    #         exp_slice = lsd_score_da.sel(coord_idx)
    #         if recompute or exp_slice.isnull().any():
    #             # Only compute if value doesn't exist:
    #             # print('recomputing l score')
    #             proposal = self.lsd(lsd_conf, print_config=False, beat_sync=beat_sync, path_sim_sigma_percentile=path_sim_sigma_percent, recompute=False)
    #             annotation = self.ref(anno_id=anno_id)
    #             # search l_score from old places first?
    #             l_score = ssdm.compute_l(proposal, annotation, l_frame_size=l_frame_size)
    #             with xr.open_dataarray(nc_path) as lsd_score_da:
    #                 lsd_score_da.load()
    #             lsd_score_da.loc[coord_idx] = list(l_score)
    #             try:
    #                 lsd_score_da.to_netcdf(nc_path)
    #             except PermissionError:
    #                 os.system(f'rm {nc_path}')
    #                 # Sometimes saving can fail due to multiple workers
    #                 pass
    #     # return lsd_score_da
    #     out = lsd_score_da.sel(**lsd_sel_dict)
    #     try:
    #         out = out.sel(anno_id=anno_id)
    #     except:
    #         pass
    #     return out


    def lsd_score_flat(self, anno_id=0, beat_sync=False, recompute=False) -> xr.DataArray:
        # save 
        if anno_id != 0:
            anno_flag = f'a{anno_id}'
        else:
            anno_flag = ''
        nc_path = os.path.join(self.output_dir, f'ells/{self.tid}_{anno_flag}flat{"_bsync2" if beat_sync else ""}.nc')

        if not os.path.exists(nc_path):
            # build da to store
            score_dims = dict(
                rep_ftype=ssdm.AVAL_FEAT_TYPES,
                loc_ftype=ssdm.AVAL_FEAT_TYPES,
                m_type=['p', 'r', 'f'],
                metric=['hr', 'hr3', 'pfc', 'nce'],
                layer=[x+1 for x in range(16)],
            )

            score_da = xr.DataArray(
                data=np.nan,  # Initialize the data with NaNs
                coords=score_dims,
                dims=list(score_dims.keys()),
                name=str(self.tid)
            )

            try:
                score_da.to_netcdf(nc_path)
            except PermissionError:
                os.system(f'rm {nc_path}')
            recompute = True
        else:
            with xr.open_dataarray(nc_path) as score_da:
                score_da.load()

        if recompute:
            # build all the lsd configs:
            for rep_ftype in ssdm.AVAL_FEAT_TYPES:
                for loc_ftype in ssdm.AVAL_FEAT_TYPES:
                    feature_pair = dict(rep_ftype=rep_ftype, loc_ftype=loc_ftype)
                    lsd_config = ssdm.DEFAULT_LSD_CONFIG.copy()
                    if beat_sync:
                        lsd_config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
                    lsd_config.update(feature_pair)
                    score_da.loc[feature_pair] = ssdm.compute_flat(self.lsd(lsd_config, beat_sync=beat_sync, recompute=False), self.ref(mode='normal', anno_id=anno_id))
            
            try:
                score_da.to_netcdf(nc_path)
            except PermissionError:
                os.system(f'rm {nc_path}')

        return score_da


    def tau_both(
        self,
        anno_mode: str = 'normal',
        recompute: bool = False,
        lap_norm: str = 'random_walk', # symmetrical or random_walk
        quantize: str = 'percentile', # None, 'kmeans' or 'percentile'
        quant_bins: int = 8, # used for loc tau quant schemes
        beat_sync: bool = True,
        verbose: bool = False,
        **anno_id_kwarg
    ) -> xr.DataArray:
        # test if record_path exist, if no, set recompute to true.
        quantize_suffix = f'_{quantize}{quant_bins}' if quantize is not None else ''
        beat_suffix = "_bsync3" if beat_sync else ""
        record_path = os.path.join(self.output_dir, f'taus/{self.tid}_{anno_mode}{quantize_suffix}{lap_norm}{beat_suffix}.nc')
        
        if not recompute:
            try:
                with xr.open_dataarray(record_path) as tau:
                    return tau.load().sel(**anno_id_kwarg)
            except:
                recompute = True

        grid_coords = dict(
            anno_id=list(range(self.num_annos())), 
            rep_ftype=ssdm.AVAL_FEAT_TYPES, 
            loc_ftype=ssdm.AVAL_FEAT_TYPES
        )
        tau = xr.DataArray(np.nan, coords=grid_coords, dims=grid_coords.keys())
        
        for anno_id, rep_ftype, loc_ftype in tau.coords.to_index():
            if verbose:
                print(anno_id, rep_ftype, loc_ftype)
            # Compute the combined rec mat
            feat_combo = dict(rep_ftype=rep_ftype, loc_ftype=loc_ftype)
            # Combined rec -> evecs function function (with caching, and normalization param lap_norm)
            evecs = self.embedded_rec_mat(feat_combo=feat_combo, beat_sync=beat_sync, lap_norm=lap_norm, recompute=recompute)
            # print(evecs.shape, beat_sync)
            # Calculate the LRA of the normalized laplacian, and use that to compute corr with anno_meet_mat
            lap_low_rank_approx = evecs[:, :10] @ evecs[:, :10].T
            
            if quantize:
                lap_low_rank_approx = ssdm.quantize(lap_low_rank_approx, quant_bins=quant_bins)
            
            if beat_sync:
                ts_mode = 'beat'
            else:
                ts_mode = 'frame'
            meet_mat = ssdm.anno_to_meet(self.ref(mode=anno_mode, anno_id=anno_id), self.ts(mode=ts_mode))
            least_beats = min(lap_low_rank_approx.shape[0], meet_mat.shape[0])
            lap_low_rank_approx = lap_low_rank_approx[:least_beats, :least_beats]
            meet_mat = meet_mat[:least_beats, :least_beats]
            tau.loc[anno_id, rep_ftype, loc_ftype] = stats.kendalltau(
                lap_low_rank_approx.flatten(), meet_mat.flatten()
                )[0]
        try:
            tau.to_netcdf(record_path)
        except:
            pass
        return tau.sel(**anno_id_kwarg)


    def embedded_rec_mat(self, feat_combo=dict(), lap_norm='random_walk', beat_sync=True, recompute=False):
        beat_suffix = {"_bsync" if beat_sync else ""}
        save_path = os.path.join(self.output_dir, f'evecs/{self.tid}_rep{feat_combo["rep_ftype"]}_loc{feat_combo["loc_ftype"]}_{lap_norm}{beat_suffix}.npy')
        if not recompute:
            try:
                return np.load(save_path)
            except:
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


class MyTrack(Track):
    def __init__(self,
                 audio_dir='/vast/qx244/my_tracks/audio', 
                 output_dir='/vast/qx244/my_tracks', 
                 title='gun_test.mp3',
                ):
        
        super().__init__(tid=title, output_dir=output_dir)
        
        self.audio_dir = audio_dir
        self.audio_path = os.path.join(audio_dir, title)
        self.feature_dir = os.path.join(output_dir, 'features')
        
    def jam(self):
        if self._jam is None:
            j = jams.JAMS()
            j.file_metadata.duration = librosa.get_duration(path=self.audio_path)
            j.file_metadata.title = self.title
            self._jam = j
            
        return self._jam


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
            self.labels = sample_select_fn(self)
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
            assert KeyError('bad mode: can only be both')
        
        nlvl_save_path = os.path.join(self.output_dir, 'evecs/'+s_info[0]+'_nlvl.npy')
        try:
            best_nlvl = np.load(nlvl_save_path)
        except:
            track = self.track_obj(tid=tid)
            best_nlvl = np.array([min(track.num_dist_segs() - 1, 10)])
            np.save(nlvl_save_path, best_nlvl)

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