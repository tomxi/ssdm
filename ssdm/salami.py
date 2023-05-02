import os, json
import numpy as np
import librosa
import jams
import mir_eval
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
        mode: str = 'normal', # {'normal', 'expand'},
        anno_id: int = 0,
    ) -> list: 
        """
        A list of `jams.Annotation`s with two modes: {'normal', 'expand'}
        """
        upper_annos = self.jam().search(namespace='segment_salami_upper')
        lower_annos = self.jam().search(namespace='segment_salami_lower')
        hier_annos = []
        if mode == 'normal':
            return [upper_annos[anno_id], lower_annos[anno_id]]
        elif mode == 'expand':
            upper_expanded = ssdm.expand_hierarchy(upper_annos[anno_id])
            lower_expanded = ssdm.expand_hierarchy(lower_annos[anno_id])
            return  upper_expanded + lower_expanded


    def segmentation_lsd(
        self,
        config = None,
    ) -> list:
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
        return hier_anno


    def segmentation_adobe(
        self,
    ) -> list:
        # TODO
        pass


### Stand alone functions
def compute_tau(
    sdm, 
    segmentation,
    ts, 
    region: str = 'full', #{'full', 'path'}
    quantize: bool = True,
) -> float:
    """
    """
    pass


def compute_l(
    proposal, 
    annotation,
) -> float:
    """
    """
    pass


def segmentation_to_meet(
    segmentation, 
    ts,
) -> np.array:
    """
    """
    pass


def segmentation_to_mireval(
    segmentation
) -> tuple:
    """
    """
    pass

    hier_intervals, hier_values = ([], [])
    return hier_intervals, hier_values


def get_ids(
    split: str = 'dev',
    id_paths: str = '/home/qx244/scanning-ssm/revive/split_ids.json',
    out_type: str = 'list' # {'set', 'list'}
) -> list:
    """ split can be ['audio', 'jams', 'excluded', 'new_val', 'new_test', 'new_train']
        Dicts sotred in id_paths json file.
    """
    with open(id_paths, 'r') as f:
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
    

# Optional do last.
def mask_sdm(
    sdm, 
    options
) -> np.ndarray:
    """
    """
    pass