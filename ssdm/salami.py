import ssdm
from ssdm import base
import os, json, pkg_resources
import jams
import ssdm.formatting
import xarray as xr
import pandas as pd
from tqdm import tqdm

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
        self.ds_name = 'slm'
        self.title = tid

    def num_annos(
        self,
    ) -> int:
        return len(self.jam().search(namespace='segment_salami_function'))


    def ref(self, mode='expand', anno_id=0):
        anno_layers = []
        # print('haha')
        if mode == 'function':
            namespaces = ['segment_salami_function']
        else:
            namespaces = ['segment_salami_upper', 'segment_salami_lower']

        for n in namespaces:
            anno = self.jam().search(namespace=n)
            anno_layers.append(base.fill_out_anno(
                anno[anno_id], self.ts(mode='frame')
            ))

        if mode == 'expand':
            expanded_layers = []
            for l in anno_layers:
                # print(l)
                expanded_layers += base.expand_hierarchy(l, dataset=self.ds_name, always_include=False)
            return ssdm.openseg2multi(expanded_layers)
        elif mode == 'normal':
            return ssdm.openseg2multi(anno_layers)

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
    split: str = 'working',
    out_type: str = 'list' # one of {'set', 'list'}
) -> list:
    """
    """
    id_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(id_path, 'r') as f:
            id_json = json.load(f)
    except FileNotFoundError:
        id_json = dict()
        id_json[split] = []
        with open(id_path, 'w') as f:
            json.dump(id_json, f)
    if split == None:
        split = 'working'
    ids = id_json[split]
        
    if out_type == 'set':
        return set(ids)
    elif out_type == 'list':
        ids.sort()
        return ids
    else:
        print('invalid out_type')
        return None


def get_adobe_scores(
    tids=[],
    anno_col_fn=lambda stack: stack.max(dim='anno_id'),
    l_frame_size=0.1
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        score_per_anno = []
        for anno_id in range(track.num_annos()):
            score_per_anno.append(track.adobe_l(anno_id=anno_id, l_frame_size=l_frame_size))

        anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
        track_flat = anno_col_fn(anno_stack)
        score_per_track.append(track_flat)

    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()


# MOVE TO SALAMI
# add new splits to split_ids.json
def update_split_json(split_name='', split_idx=[]):
    # add new splits to split_id.json file at json_path
    # read from json and get dict
    json_path = pkg_resources.resource_filename('ssdm', 'split_ids.json')
    try:
        with open(json_path, 'r') as f:
            split_dict = json.load(f)
    except FileNotFoundError:
        split_dict = dict()
        split_dict[split_name] = split_idx
        with open(json_path, 'w') as f:
            json.dump(split_dict, f)
        return split_dict

    # add new split to dict
    split_dict[split_name] = split_idx

    # save json again
    with open(json_path, 'w') as f:
        json.dump(split_dict, f)

    with open(json_path, 'r') as f:
        return json.load(f)


class PairDS(base.PairDS):
    def __init__(self, **kwargs):
        super().__init__(ds_module=ssdm.slm, name='slm', **kwargs)

class PairDSLmeasure(base.PairDSLmeasure):
    def __init__(self, **kwargs):
        super().__init__(ds_module=ssdm.slm, name='slm', **kwargs)


class InferDS(base.InferDS):
    def __init__(self, **kwargs):
        super().__init__(ds_module=ssdm.slm, name='slm', **kwargs)


class LvlDS(base.LvlDS):
    def __init__(self, **kwargs):
        super().__init__(ds_module=ssdm.slm, name='slm', **kwargs)