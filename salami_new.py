import os, json
import numpy as np
import librosa
import jams
import mir_eval
from sklearn import preprocessing
from scipy import spatial, sparse, stats

import spliter
import feature
import scluster as sc


class Track:
    def __init__(
        self, 
        tid: str = '2', 
        salami_dir: str = '/scratch/qx244/data/salami/',
    ):
        self.tid = tid
        self.salami_dir = salami_dir
        self.audio_path = os.path.join(salami_dir, f'audio/{tid}/audio.mp3')

        self._y = None # populate when .audio() is called.
        self._sr = None # populate when .audio() is called.
        self._num_annos = None # populate when .jam() is called.
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
        jam = jams.load(os.path.join(self.salami_dir, f'jams/{self.tid}.jams'))
        if self._num_annos is None:
            self._num_annos = len(jam.search(namespace='segment_salami_upper'))
        return jam
    
    
    def segmentation_annotation(
        self,
        mode: str = 'normal', # {'normal', 'expanded'},
        anno_id: int = 0,
    ) -> list: # hier annotaion format (standardize)
        # TODO
        upper_annos = self.jam.search(namespace='segment_salami_upper')
        lower_annos = self.jam.search(namespace='segment_salami_lower')
        hier_annos = []
        if mode == 'normal':
            return [upper_annos[anno_id], lower_annos[anno_id]]
        elif mode == 'expanded':
            # TODO
            pass

            


    def segmentation_lsd(
        self, 
    ) -> list: # hier annotaion format (standardize)
        # TODO
        pass


    def segmentation_adobe(
        self,
    ) -> list:
        # TODO
        pass
    
    
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
            feat_mat = self.feature(feat_type=feature, **kwargs)
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


    def ts(
        self
    ) -> np.array:
        if self._common_ts is None:
            num_frames = np.asarray([
                len(feature.openl3(self)['ts']),
                len(feature.yamnet(self)['ts']),
                len(feature.chroma(self)['ts']),
                len(feature.crema(self)['ts']),
                len(feature.tempogram(self)['ts']),
                len(feature.mfcc(self)['ts'])
            ])

            self._common_ts = feature.crema(self)['ts'][:np.min(num_frames)]
        
        return self._common_ts


    def feature(
        self,
        feat_type: str = 'openl3',
        add_noise: bool = False,
        time_delay_emb: bool = False,
    ) -> np.array:
        # THIS HAS SR FIXED AT 22050
        if feat_type == 'openl3':
            feat_npz = openl3(track)
        elif feat_type == 'yamnet':
            feat_npz = yamnet(track)
        elif feat_type == 'mfcc':
            feat_npz = mfcc(track)
        elif feat_type == 'tempogram':
            feat_npz = tempogram(track)
        elif feat_type == 'chroma':
            feat_npz = chroma(track)
        elif feat_type == 'crema':
            feat_npz = crema(track)
        else:
            raise librosa.ParameterError('bad feature name')

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