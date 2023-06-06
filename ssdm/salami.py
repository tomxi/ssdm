import os, json
import numpy as np
import librosa
import jams
import pandas as pd
from scipy import spatial, sparse

import ssdm
import ssdm.scluster as sc

AVAL_FEAT_TYPES = ['chroma', 'crema', 'tempogram', 'mfcc', 'yamnet', 'openl3']
AVAL_DIST_TYPES = ['cosine', 'sqeuclidean', 'cityblock']
DEFAULT_LSD_CONFIG = {
    'rec_width': 13,
    'rec_smooth': 7,
    'rec_full': False,
    'evec_smooth': 13,
    'rep_ftype': 'chroma', # grid
    'loc_ftype': 'mfcc', # grid
    'rep_metric': 'cosine', # grid
    'loc_metric': 'cosine', # grid
    'hier': True,
    'num_layers': 12
}

REP_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'crema': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'tempogram': {'add_noise': True, 'n_steps': 1, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 6, 'delay': 2},
    'yamnet': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 3, 'delay': 1},
}

LOC_FEAT_CONFIG = {
    'chroma': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'crema': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'tempogram': {'add_noise': True, 'n_steps': 1, 'delay': 1},
    'mfcc': {'add_noise': True, 'n_steps': 3, 'delay': 1},
    'yamnet': {'add_noise': True, 'n_steps': 1, 'delay': 1},
    'openl3': {'add_noise': True, 'n_steps': 1, 'delay': 1},
}

class Track:
    def __init__(
        self, 
        tid: str = '384', 
        salami_dir: str = '/scratch/qx244/data/salami/',
    ):
        self.tid = tid
        self.salami_dir = salami_dir
        self.audio_path = os.path.join(salami_dir, f'audio/{tid}/audio.mp3')

        self._y = None # populate when .audio() is called.
        self._sr = None # populate when .audio() is called.
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
            self._jam = jams.load(os.path.join(self.salami_dir, f'jams/{self.tid}.jams'))
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
        width = 13, # ~2.5 sec width param for librosa.segment.rec_mat <= 1
        bw: str = 'med_k_scalar', # one of {'med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair'}
        full: bool = False,
        add_noise: bool = False, # padding representation with a little noise to avoid divide by 0 somewhere...
        n_steps: int = 1, # Param for time delay embedding of representation
        delay: int = 1, # Param for time delay embedding of representation
        recompute: bool = False,
    ) -> np.array:
        # npy path
        ssm_info_str = f'{feature}_{distance}{f"_fw{width}" if full else ""}'
        feat_info_str = f's{n_steps}xd{delay}{"_n" if add_noise else ""}'
        ssm_path = os.path.join(
            self.salami_dir, 
            f'ssms/{self.tid}_{ssm_info_str}_{feat_info_str}_{bw}.npy'
        )
        # print(ssm_path)
        
        # see if desired ssm is already computed
        if not os.path.exists(ssm_path):
            recompute = True

        if recompute:
            # compute ssm
            feat_mat = self.representation(
                feat_type=feature, 
                add_noise=add_noise,
                n_steps=n_steps,
                delay=delay,
            )
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
            distance: str = 'cosine',
            add_noise: bool = False, # padding representation with a little noise to avoid divide by 0 somewhere...
            n_steps: int = 1, # Param for time delay embedding of representation
            delay: int = 1, # Param for time delay embedding of representation
            recompute: bool = False,
        ) -> np.array:
        # npy path
        path_info_str = f'path_{feature}_{distance}'
        feat_info_str = f's{n_steps}xd{delay}{"_n" if add_noise else ""}'
        path_sim_path = os.path.join(
            self.salami_dir, 
            f'ssms/{self.tid}_{path_info_str}_{feat_info_str}.npy'
        )
        # print(path_sim_path)
        
        # see if desired ssm is already computed
        if not os.path.exists(path_sim_path):
            recompute = True

        if recompute:
            feat_mat = self.representation(
                feat_type=feature, 
                add_noise=add_noise,
                n_steps=n_steps,
                delay=delay,
            )

            frames = feat_mat.shape[1]
            path_dist = []
            for i in range(frames-1):
                pair_dist = spatial.distance.pdist(feat_mat[:, i:i + 2].T, metric=distance)
                path_dist.append(pair_dist[0])
            path_dist = np.array(path_dist)
            sigma = np.median(path_dist)
            path_sim = np.exp(-path_dist / sigma)

            # store path_sim
            with open(path_sim_path, 'wb') as f:
                np.save(f, path_sim)

        return np.load(path_sim_path, allow_pickle=True)


    def segmentation_annotation(
        self,
        mode: str = 'normal', # {'normal', 'expand', 'refine', 'coarse'},
        anno_id: int = 0,
    ) -> jams.JAMS: 
        """
        A list of `jams.Annotation`s with two modes: {'normal', 'expand'}
        """
        upper_annos = self.jam().search(namespace='segment_salami_upper')
        lower_annos = self.jam().search(namespace='segment_salami_lower')
        if mode == 'normal':
            out_jam = jams.JAMS(annotations=[upper_annos[anno_id], lower_annos[anno_id]])
        else:
            upper_expanded = ssdm.expand_hierarchy(upper_annos[anno_id])
            lower_expanded = ssdm.expand_hierarchy(lower_annos[anno_id])
            
            if mode == 'expand':
                out_jam = jams.JAMS(annotations=upper_expanded + lower_expanded)
            elif mode == 'refine':
                upper_refined = upper_expanded[-1]
                lower_refined = lower_expanded[-1]
                out_jam = jams.JAMS(annotations=[upper_refined, lower_refined])
            elif mode == 'coarse':
                upper_coarsened = upper_expanded[0]
                lower_coarsened = lower_expanded[0]
                out_jam = jams.JAMS(annotations=[upper_coarsened, lower_coarsened])
            else:
                raise librosa.ParameterError("mode can only be one of 'normal', 'expand', 'refine', or 'coarse'.")
        out_jam.file_metadata.duration = self.ts()[-1]
        return out_jam


    def segmentation_lsd(
        self,
        config = None,
        recompute = True,
    ) -> jams.JAMS:
        """
        A list of `jams.Annotation`s segmented with config
        """
        if config is None:
            config = DEFAULT_LSD_CONFIG

        full_flag = 'full' if config['rec_full'] else 'sparse'
        record_path = os.path.join(self.salami_dir, 
                                   f'lsds/{self.tid}_{config["rep_ftype"]}_{config["loc_ftype"]}_{full_flag}.jams'
                                  )
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            # Compute/get SSM mats
            rep_ssm = self.ssm(feature=config['rep_ftype'], 
                               distance=config['rep_metric'],
                               width=config['rec_width'],
                               full=config['rec_full'],
                               **REP_FEAT_CONFIG[config['rep_ftype']]
                              )
            if config['rec_full']:
                rep_ssm = ssdm.mask_diag(rep_ssm, width=config['rec_width'])

            
            path_sim = self.path_sim(feature=config['loc_ftype'],
                                     distance=config['loc_metric'],
                                     **LOC_FEAT_CONFIG[config['loc_ftype']]
                                    )
            # Spectral Clustering with Config
            est_bdry_idxs, est_sgmt_labels = sc.do_segmentation_ssm(rep_ssm, path_sim, config)

            # create jams annotation
            hier_anno = []
            for layer_bdry_idx, layer_labels in zip(est_bdry_idxs, est_sgmt_labels):
                anno = jams.Annotation(namespace='segment_open')
                for i, segment_label in enumerate(layer_labels):
                    start_time = self.ts()[layer_bdry_idx[i]]
                    end_time = self.ts()[layer_bdry_idx[i+1]]
                    anno.append(
                        time=start_time, 
                        duration=end_time - start_time,
                        value=str(segment_label),
                    )
                hier_anno.append(anno)

            lsd_jam = jams.JAMS(annotations=hier_anno, sandbox=config)
            lsd_jam.file_metadata.duration = self.ts()[-1]
            lsd_jam.save(record_path)
        return jams.load(record_path)


    def segmentation_adobe(
        self,
    ) -> jams.JAMS:
        result_dir = '/scratch/qx244/data/ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/'
        filename = f'{self.tid}.mp3.msdclasscsnmagic.json'

        with open(os.path.join(result_dir, filename), 'rb') as f:
            justin_result = json.load(f)

        # Build jams
        hier_anno = []
        for layer, (bdry, labels) in enumerate(justin_result):
            ann = jams.Annotation(namespace='segment_open')
            for ival, label in zip(bdry, labels):
                ann.append(time=ival[0], duration=ival[1]-ival[0], value=label)
            hier_anno.append(ann)
            
        justin_jam = jams.JAMS(annotations=hier_anno, sandbox={'mu': 0.1, 'gamma': 0.1})
        justin_jam.file_metadata.duration=self.ts()[-1]
        return justin_jam


    def tau(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        quantize: str = 'kmeans', # None, 'kemans' or 'percentile'
        quant_bins: int = 6, # used for both quantization schemes
        lsd_config: dict = DEFAULT_LSD_CONFIG,
        combined: bool = False, # make a 6*6 square that's the gmean of tau_rep and loc
    ) -> pd.DataFrame:
        """  
        """
        # test if record_path exist, if no, set recompute to true.
        suffix = f'_{quantize}{quant_bins}' if quantize is not None else ''

        record_path = os.path.join(self.salami_dir, f'taus/{self.tid}_a{anno_id}{suffix}.pkl')
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            # creating a dataframe to store all different variations of kendall taus for each feature
            tau = pd.DataFrame(
                index=AVAL_FEAT_TYPES, 
                columns=['full_expand', 'path_expand'],
                dtype='float'
            )
            for feat in AVAL_FEAT_TYPES:
                ssm = self.ssm(feature=feat, 
                               distance=lsd_config['rep_metric'],
                               width=lsd_config['rec_width'],
                               full=lsd_config['rec_full'],
                               **REP_FEAT_CONFIG[feat]
                              )
                path_sim = self.path_sim(feature=feat,
                                         distance=lsd_config['loc_metric'],
                                         **LOC_FEAT_CONFIG[feat]
                                        )
                
                tau.loc[feat][f'full_expand'] = ssdm.tau_ssm(
                    ssm, 
                    self.segmentation_annotation(mode='expand', anno_id=anno_id),
                    self.ts(),
                    quantize = quantize,
                    quant_bins = quant_bins
                )
                tau.loc[feat][f'path_expand'] = ssdm.tau_path(
                    path_sim, 
                    self.segmentation_annotation(mode='expand', anno_id=anno_id),
                    self.ts(),
                    quantize = quantize,
                    quant_bins = quant_bins
                )
                
            # Save to record_path
            tau.to_pickle(record_path)
        
        # Read from record_path
        tau = pd.read_pickle(record_path)
        if combined:
            tau -= tau.to_numpy().flatten().min()
            tau += 1e-8

            combined_score = pd.DataFrame(
                np.outer(tau['full_expand'], tau['path_expand'])**0.5, 
                index = list(tau['full_expand'].index), 
                columns = tau['path_expand'].index
            )
            combined_score.index.name='tau_rep'
            combined_score.columns.name='tau_loc'
            return combined_score
        else:
            return tau
    

## TO CHANGE
    def lsd_l(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        anno_mode: str = 'expand', #see `segmentation_anno`'s `mode`.
        l_type: str = 'l', # can also be 'lr' and 'lp' for recall and precision.
        lsd_config: dict = DEFAULT_LSD_CONFIG
    ) -> pd.DataFrame:
        record_path = os.path.join(self.salami_dir, f'ells/{self.tid}_a{anno_id}_{anno_mode}.pkl')
        # test if record_path exist, if no, set recompute to true.
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            l_score = pd.DataFrame(
                index=AVAL_FEAT_TYPES,
                columns=pd.MultiIndex.from_product([['lp', 'lr', 'l'], AVAL_FEAT_TYPES])
            )
            for r_feat in AVAL_FEAT_TYPES:
                lsd_config['rep_ftype'] = r_feat
                for l_feat in AVAL_FEAT_TYPES:
                    lsd_config['loc_ftype'] = l_feat
                    l_score.loc[r_feat, (slice(None), l_feat)] = ssdm.compute_l(
                        self.segmentation_lsd(lsd_config, recompute=recompute), 
                        self.segmentation_annotation(mode=anno_mode, anno_id=anno_id)
                    )
            # save to record_path
            l_score.to_pickle(record_path)

        # Read from record_path
        l_df = pd.read_pickle(record_path).astype('float')
        l_sub_square = l_df.loc[slice(None), (l_type, slice(None))]
        l_sub_square.index.name = 'rep_feat'
        l_sub_square.columns = l_sub_square.columns.droplevel(0)
        l_sub_square.columns.name = 'loc_feat'
        return l_sub_square
    

    def adobe_l(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        anno_mode: str = 'expand', #see `segmentation_anno`'s `mode`.
        l_type: str = 'l' # can also be 'lr' and 'lp' for recall and precision.
    ) -> pd.DataFrame:
        record_path = os.path.join(self.salami_dir, f'ells/{self.tid}_a{anno_id}_{anno_mode}_adobe.pkl')
        # test if record_path exist, if no, set recompute to true.
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            l_score = pd.Series(index=['lp', 'lr', 'l'])
            l_score[:]= ssdm.compute_l(
                self.segmentation_adobe(), 
                self.segmentation_annotation(mode=anno_mode, anno_id=anno_id)
            )
            # save to record_path
            l_score.to_pickle(record_path)

        # Read from record_path
        return pd.read_pickle(record_path).astype('float')[l_type]
