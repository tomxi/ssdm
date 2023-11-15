import numpy as np
import librosa
import jams
import os
from glob import glob
import ssdm

class HarmonixTrack(object):
    def __init__(self,
                 tid, 
                 feature_dir='/scratch/qx244/data/Harmonix_set_openl3_and_yamnet_features/features/',
                ):
        self.tid = tid
        self.feature_dir = feature_dir
        self.basename = os.path.basename(glob(os.path.join(self.feature_dir, f'{self.tid}*mfcc.npz'))[0])
        self.title = self.basename.split('_')[1]
        
        self._track_ts = None # populate when .ts() is called.
        
    def representation(self, feature='mfcc', use_track_ts=True):
        feature_path = glob(os.path.join(self.feature_dir, f'{self.tid}*{feature}.npz'))[0]
        npz = np.load(feature_path)
        if not use_track_ts:
            return npz['feature']
        else:
            return npz['feature'][:, :len(self.ts())]
    
    def ts(self) -> np.array:
        if self._track_ts is None:
            num_frames = np.asarray([self.representation(feat, use_track_ts=False).shape[-1] for feat in ssdm.AVAL_FEAT_TYPES])
            self._track_ts = librosa.frames_to_time(list(range(np.min(num_frames))), hop_length=4096, sr=22050)
        return self._track_ts
    
    def ref(self, mode='expand'):
        j = jams.load(f'/home/qx244/harmonixset/dataset/jams/{self.tid}_{self.title}.jams')
        seg_anns = j.search(namespace='segment_open')

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
            return ssdm.openseg2multi(ssdm.expand_hierarchy(anno))
        elif mode == 'normal':
            return ssdm.openseg2multi([anno])
        
    def path_ref(
            self,
            mode: str = 'expand', # {'normal', 'expand'}
            binarize: bool = True,
        ):
        # Get reference annotation
        ref_anno = self.ref(mode)
        # Get annotation meet matrix
        anno_meet = ssdm.anno_to_meet(ref_anno, self.ts())
        # Pull out diagonal
        anno_diag = anno_meet.diagonal(1)
        if binarize:
            anno_diag = anno_diag == np.max(anno_diag)
        return anno_diag.astype(int)
    
    def ssm(self, feat='mfcc', add_noise=True, n_steps=6, delay=2, ssm_width=30, metric='cosine'): 
        # save
        
        # compute
        out = librosa.segment.recurrence_matrix(
            delay_embed(self.representation(feat), add_noise, n_steps, delay),
            width=ssm_width,
            metric=metric,
            mode='affinity',
        )
        return out
    
    def path_sim(self): #SAVE
        pass
    
    def lsd(self): #SAVE
        pass
    
    def tau(self): #SAVE
        pass
    
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

    return librosa.feature.stack_memory(
        feat_mat, 
        mode='edge', 
        n_steps=n_steps, 
        delay=delay
    )

