from . import base
import ssdm
import xarray as xr

import torch
from torch.utils.data import Dataset

import os, jams, itertools
import pandas as pd
import librosa
# import dataset
from tqdm import tqdm

class Track(base.Track):
    def __init__(
        self,
        tid: str = '1', # 1-456 # melid
        dataset_dir: str = '/home/qx244/jsd/data', 
        output_dir: str = '/vast/qx244/jsd/',
        feature_dir: str = '/vast/qx244/jsd/features/',
        audio_dir: str = '/scratch/qx244/data/weimar_jazz_original_audio/'
    ):
        super().__init__(tid, dataset_dir, output_dir, feature_dir)
        # Track IDs for JSD looks like this
        track_relationships = pd.read_csv('/home/qx244/jsd/data/track_relationships.csv')

        self.info = track_relationships.loc[track_relationships.melid==int(tid)]
        self.title = self.info.filename_solo.item().replace('=', '-').replace('&', '').replace(',', '').replace('.', '')
        self.title = self.title.replace('(', '').replace(')', '').replace('_solo', '_Solo')
        self.track_title = self.info.filename_track.item()
        self.audio_path = os.path.join(audio_dir, f'{self.title}.wav')
        
        self._solo_start = None

        for directory in [output_dir, feature_dir]:
            if not os.path.exists(directory):
                os.system(f'mkdir {directory}')


    def solo_start_time(self):
        raise NotImplementedError
        if self._solo_start == None:
            db = dataset.connect("sqlite:////home/qx244/jsd/data/wjazzd.db")
            transcription_info = db['transcription_info'].find_one(melid=self.tid)
            self._solo_start = transcription_info['solostart_sec']
            db.close()

        return self._solo_start

    def jam(self):
        if self._jam is None:
            # convert from csv to jams. Return the whole and save at self._jam, and then slice when calling
            csv_path = os.path.join(self.dataset_dir, f'annotations_csv/{self.track_title}.csv')
            df = pd.read_csv(csv_path, sep=';', header=0)

            # get song duration
            track_dur = df.iloc[-1].segment_end

            # build jams from df.
            jam = jams.JAMS()
            jam.file_metadata.duration = track_dur
            jam.file_metadata.title = self.track_title

            # Add a new annotation to the JAMS object
            seg_anno = jams.Annotation(namespace='segment_open', duration=track_dur)

            # for the excerpt in hand
            for i in df.index:
                item = df.loc[i]
                start = item.segment_start
                dur = item.segment_end - item.segment_start
                
                seg_anno.append(time=start, duration=dur, value=item.label)

            # Add the annotation to the JAMS object, but trimmed
            jam.annotations.append(seg_anno)
            self._jam = jam
            self._track_dur = track_dur
        

        solo_start = self.solo_start_time()
        file_dur =  librosa.get_duration(path = self.audio_path)
        solo_end = min(solo_start + file_dur, self._track_dur)
        return self._jam.slice(solo_start, solo_end)
       
       
def get_ids(out_type: str = 'list'):
    track_relationships = pd.read_csv('/home/qx244/jsd/data/track_relationships.csv')
    melids = set(list(track_relationships.melid))
    issue_files = set([43, 64, 309, 382]) # issue files from above
    duplicate_files = set([343, 344, 345]) # See JSD, duplicates
    single_segment = set([86, 119, 207])
    flipped_files = set([79 ,82])
    mid_set = melids - issue_files - duplicate_files - flipped_files - single_segment
    mid_set_strings = {str(x) for x in mid_set}
    if out_type == 'list':
        return list(mid_set_strings)
    if out_type == 'set':
        return mid_set_strings
    else:
        raise KeyError('bad out_type')


def get_lsd_scores(
    tids=[], 
    **lsd_score_kwargs
) -> xr.DataArray:
    score_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        score_per_track.append(track.lsd_score(**lsd_score_kwargs))
    
    return xr.concat(score_per_track, pd.Index(tids, name='tid')).rename()
    

def get_taus(
    tids=[], 
    **tau_kwargs,
) -> xr.DataArray:
    tau_per_track = []
    for tid in tqdm(tids):
        track = Track(tid)
        tau_per_track.append(track.tau(**tau_kwargs))
    
    return xr.concat(tau_per_track, pd.Index(tids, name='tid')).rename()


class NewDS(base.DS):
    def __init__(self, tids=None, **kwargs):
        self.name = 'jsd'
        if not tids:
            self.tids=get_ids(out_type='list')
            self.split=''
        else:
            self.tids=tids
            self.split=f'custom{len(tids)}'
        
        super().__init__(infer=True, **kwargs)

    def track_obj(self, **track_kwargs):
        return Track(**track_kwargs)

        
        