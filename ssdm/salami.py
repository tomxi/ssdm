import os, json
import pkg_resources
import numpy as np
import librosa
import jams
import mir_eval
import pandas as pd
from sklearn import preprocessing
from scipy import spatial, sparse, stats

import ssdm
import ssdm.scluster as sc



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
        time_delay_emb: bool = False,
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

        if time_delay_emb:
            feat_mat = librosa.feature.stack_memory(
                feat_mat, mode='edge', n_steps=6, delay=2
            )
        return feat_mat


    def sdm(
        self,
        feature: str = 'mfcc',
        distance: str = 'cosine',
        recompute: bool = False,
        **kwargs, # add_noise, time_delay_emb; for preping features
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
        hier_annos = []
        if mode == 'normal':
            return jams.JAMS(annotations=[upper_annos[anno_id], lower_annos[anno_id]])
        else:
            upper_expanded = ssdm.expand_hierarchy(upper_annos[anno_id])
            lower_expanded = ssdm.expand_hierarchy(lower_annos[anno_id])
            
            if mode == 'expand':
                return  jams.JAMS(annotations=upper_expanded + lower_expanded)
            elif mode == 'refine':
                upper_refined = upper_expanded[-1]
                lower_refined = lower_expanded[-1]
                return  jams.JAMS(annotations=[upper_refined, lower_refined])
            elif mode == 'coarse':
                upper_coarsened = upper_expanded[0]
                lower_coarsened = lower_expanded[0]
                return  jams.JAMS(annotations=[upper_coarsened, lower_coarsened])
            else:
                raise librosa.ParameterError("mode can only be one of 'normal', 'expand', 'refine', or 'coarse'.")


    def segmentation_lsd(
        self,
        config = None,
    ) -> jams.JAMS:
        """
        A list of `jams.Annotation`s segmented with config
        """
        if config is None:
            config = ssdm.DEFAULT_LSD_CONFIG
        
        rep_kwarg = ssdm.REPRESENTATION_KWARGS[config['rep_ftype']]
        loc_kwarg = ssdm.REPRESENTATION_KWARGS[config['loc_ftype']]
        rep_f = self.representation(feat_type=config['rep_ftype'], **rep_kwarg)[:, :len(self.ts())]
        loc_f = self.representation(feat_type=config['loc_ftype'], **loc_kwarg)[:, :len(self.ts())]

        # Spectral Clustering with Config
        est_bdry_idxs, est_sgmt_labels = sc.do_segmentation(rep_f, loc_f, config)

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
                    value=segment_label
                )
            hier_anno.append(anno)

        lsd_jam = jams.JAMS(annotations=hier_anno, sandbox=config)
        return lsd_jam


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
        return justin_jam


    def report_tau(
        self,
        anno_id: int = 0,
        recompute: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        kwargs for compute_tau function: 
            region: str = 'full', #{'full', 'path'}
            quantize: bool = False, 
            quant_bins: int = 16, # number of quantization bins, ignored whtn quantize is Flase
        """
        # test if record_path exist, if no, set recompute to true.
        record_path = os.path.join(self.salami_dir, f'taus/{self.tid}_a{anno_id}.pkl')
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            # creating a dataframe to store all different variations of kendall taus for each feature
            tau_score = pd.DataFrame(
                index=ssdm.AVAL_FEAT_TYPES, 
                columns=['full_expand', 'full_normal', 'full_refine', 'path_expand', 'path_normal', 'path_refine']
            )
            for feat in ssdm.AVAL_FEAT_TYPES:
                sdm = self.sdm(feature=feat, **ssdm.REPRESENTATION_KWARGS[feat])
                tau_score.loc[feat]['full_expand'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='expand', anno_id=anno_id),
                    self.ts(),
                    region='full',
                    **kwargs
                )
                tau_score.loc[feat]['full_normal'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='normal', anno_id=anno_id),
                    self.ts(),
                    region='full',
                    **kwargs
                )
                tau_score.loc[feat]['full_refine'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='refine', anno_id=anno_id),
                    self.ts(),
                    region='full',
                    **kwargs
                )
                tau_score.loc[feat]['path_refine'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='refine', anno_id=anno_id),
                    self.ts(),
                    region='path',
                    **kwargs
                )
                tau_score.loc[feat]['path_normal'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='normal', anno_id=anno_id),
                    self.ts(),
                    region='path',
                    **kwargs
                )
                tau_score.loc[feat]['path_expand'] = compute_tau(
                    sdm, 
                    self.segmentation_annotation(mode='expand', anno_id=anno_id),
                    self.ts(),
                    region='path',
                    **kwargs
                )
            # Save to record_path
            tau_score.to_pickle(record_path)
        
        # Read from record_path
        return pd.read_pickle(record_path)


    def report_lsd_l(
        self,
        anno_id: int = 0,
        recompute: bool = False,
    ) -> pd.DataFrame:
        record_path = os.path.join(self.salami_dir, f'ells/{self.tid}_a{anno_id}.pkl')
        # test if record_path exist, if no, set recompute to true.
        if not os.path.exists(record_path):
            recompute = True

        if recompute:
            l_score = pd.DataFrame(
                index=ssdm.AVAL_FEAT_TYPES,
                columns=pd.MultiIndex.from_product([['lp', 'lr', 'l'], ssdm.AVAL_FEAT_TYPES])
            )
            lsd_config = ssdm.DEFAULT_LSD_CONFIG
            for r_feat in ssdm.AVAL_FEAT_TYPES:
                lsd_config['rep_ftype'] = r_feat
                for l_feat in ssdm.AVAL_FEAT_TYPES:
                    lsd_config['loc_ftype'] = l_feat
                    l_score.loc[r_feat, (slice(None), l_feat)] = compute_l(
                        self.segmentation_lsd(lsd_config), 
                        self.segmentation_annotation(anno_id=anno_id)
                    )
            # save to record_path
            l_score.to_pickle(record_path)

        # Read from record_path
        return pd.read_pickle(record_path)


### Stand alone functions
def get_ids(
    split: str = 'dev',
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    """ split can be ['audio', 'jams', 'excluded', 'new_val', 'new_test', 'new_train']
        Dicts sotred in id_path json file.
    """
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    with open(id_path, 'r') as f:
        id_json = json.load(f)
        
    ids = id_json[split]
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


def segmentation_to_meet(
    segmentation, 
    ts,
) -> np.array:
    """
    """
    # initialize a 3d array to store all the meet matrices for each layer
    hier_anno = segmentation.annotations
    n_level = len(hier_anno)
    n_frames = len(ts)
    meet_mat_per_level = np.zeros((n_level, n_frames, n_frames))

    # put meet mat of each level of hierarchy in axis=0
    for level in range(n_level):
        layer_anno = hier_anno[level]
        label_samples = layer_anno.to_samples(ts)
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform([l[0] if len(l) > 0 else 'NL' for l in label_samples])
        meet_mat_per_level[level] = np.equal.outer(encoded_labels, encoded_labels).astype('float') * (level + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def segmentation_to_mireval(
    segmentation
) -> tuple:
    """
    """
    mir_eval_interval = []
    mir_eval_label = []
    for anno in segmentation.annotations:
        interval, value = anno.to_interval_values()
        mir_eval_interval.append(interval)
        mir_eval_label.append(value)

    return mir_eval_interval, mir_eval_label


def compute_tau(
    sdm: np.array, 
    segmentation: jams.JAMS,
    ts: np.array, 
    region: str = 'full', #{'full', 'path'}
    quantize: bool = False, 
    quant_bins: int = 16, # number of quantization bins, ignored whtn quantize is Flase
) -> float:
    """
    """
    meet_mat = segmentation_to_meet(segmentation, ts)
    
    if region == 'full':
        if quantize:
            normalized_sdm = sdm / (sdm.max() + 1e-9)
            bins = np.arange(0, 1 + 1e-9, 1 / quant_bins)
            sdm = np.digitize(normalized_sdm, bins=bins, right=False)

        return -stats.kendalltau(sdm.flatten(), meet_mat.flatten())[0]
    elif region == 'path':
        meet_diag = np.diag(meet_mat, k=1)
        sdm_diag = np.diag(sdm, k=1)
        if quantize:
            normalized_sdm_diag = sdm_diag / (sdm_diag.max() + 1e-9)
            bins = np.arange(0, 1 + 1e-9, 1 / quant_bins)
            sdm_diag = np.digitize(normalized_sdm_diag, bins=bins, right=False)
        return -stats.kendalltau(sdm_diag, meet_diag)[0]
    else:
        raise librosa.ParameterError('region can only be "full" or "path"')


def compute_l(
    proposal: jams.JAMS, 
    annotation: jams.JAMS,
) -> np.array:
    """
    """
    anno_interval, anno_label = segmentation_to_mireval(annotation)
    proposal_interval, proposal_label = segmentation_to_mireval(proposal)

    # make last segment for estimation end at the same time as annotation
    end = max(anno_interval[-1][-1, 1], proposal_interval[-1][-1, 1])
    for i in range(len(proposal_interval)):
        proposal_interval[i][-1, 1] = end
    for i in range(len(anno_interval)):
        anno_interval[i][-1, 1] = end

    return mir_eval.hierarchy.lmeasure(
        anno_interval, anno_label, proposal_interval, proposal_label,
    )

