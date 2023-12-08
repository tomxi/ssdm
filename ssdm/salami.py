import ssdm
from ssdm import base

import os, json
import librosa, jams
import numpy as np
import xarray as xr
import pandas as pd

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

    # def audio(
    #     self, 
    #     sr: float = 22050
    # ) -> tuple:
    #     if self._sr != sr:
    #         self._y, self._sr = librosa.load(self.audio_path, sr=sr)
    #     return self._y, self._sr


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
    
