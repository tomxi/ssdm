import ssdm
from ssdm import base
import os, json, pkg_resources
import librosa, jams
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

from scipy import stats



class Track(base.Track):
    def __init__(
            self, 
            tid: str = '384', 
            dataset_dir: str = '/Users/tomxi/data/salami', 
            output_dir: str = '/Users/tomxi/data/salami',
            feature_dir: str = '/Users/tomxi/data/salami'
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


def get_ids(
    split: str = None,
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    id_path = '/Users/tomxi/ssdm/ssdm/splits.json'
    with open(id_path, 'r') as f:
        id_json = json.load(f)
    ids = id_json[f'slm_{split}']
    ids.sort()
    return ids



def get_samps(split):
    with open('/Users/tomxi/ssdm/ssdm/labels.json', 'r') as f:
        labels = json.load(f)

    for k in labels:
        labels[k] = {
            tuple(k.replace('(', '').replace(')', '').replace("'", '').split(', ')): value for k, value in labels[k].items()
        }

    return labels[f'hmx_{split}_labels']


class NewDS(base.DS):
    def __init__(self, split='train', tids=None, infer=True, 
                 sample_select_fn=get_samps, 
                 **kwargs):
        self.name = 'slm'

        if tids is None:
            self.tids = get_ids(split=split, out_type='list')
            self.split = split
        else:
            self.tids = tids
            self.split = f'custom{len(tids)}'
        
        super().__init__(infer=infer, sample_select_fn=sample_select_fn, **kwargs)
    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)
    