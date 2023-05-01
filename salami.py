import os, json
import h5py
import spliter


import numpy as np
import librosa
import jams
import mir_eval
from sklearn import preprocessing
from scipy import spatial, sparse, stats

# import tensorflow_hub as hub
# import openl3


import feature
import scluster as sc

from typing import List, Any


def get_ids(
    split: str = 'dev',
    id_paths: str = '/home/qx244/scanning-ssm/revive/split_ids.json',
) -> set:
    """ split can be ['audio', 'jams', 'excluded', 'val', 'test', 'train']
        Dicts sotred in id_paths json file.
    """
    with open(id_paths, 'r') as f:
        id_json = json.load(f)

    return set(id_json[split])


class Track:
    def __init__(
        self, 
        tid: str = '2', 
        salami_dir: str = '/scratch/qx244/data/salami/',
    ):
        self.tid = tid
        self.salami_dir = salami_dir
        self.audio_path = os.path.join(salami_dir, f'audio/{tid}/audio.mp3')
        self.jam = jams.load(os.path.join(salami_dir, f'jams/{tid}.jams'))
        self.num_annos = len(self.jam.search(namespace='segment_salami_upper'))
        self.y = None
        self.sr = None
        self.min_frame = None
        
    
    def get_audio(self, sr: int = 22050) -> tuple:
        if self.sr != sr:
            self.y, self.sr = librosa.load(self.audio_path, sr=sr)
        return self.y, self.sr
    
    
    def annos(self) -> List[tuple]:
        upper_annos = self.jam.search(namespace='segment_salami_upper')
        lower_annos = self.jam.search(namespace='segment_salami_lower')
        hier_annos = []
        for hier_anno in zip(upper_annos, lower_annos):
            hier_annos.append(hier_anno)
        return hier_annos
    
    def lsd_anno(self, rep_feature='openl3', loc_feature='mfcc'):
        hier_anno = []
        config = {'rec_width': 13, 
                  'rec_smooth': 7, 
                  'evec_smooth': 13,
                  'rep_ftype': rep_feature, 
                  'loc_ftype': loc_feature,
                  'rep_metric': 'cosine',
                  'hier': True,
                  'num_layers': 10}
    
        est_intervals, est_sgmt_labels = self.lsd(config)
        
        hier_anno = []
        for lvl in range(len(est_intervals)):
            itvs = est_intervals[lvl]
            labs = est_sgmt_labels[lvl]
            lvl_anno = jams.Annotation(namespace='segment_open')
            for itv, lab in zip(itvs, labs):
                lvl_anno.append(time=itv[0], duration=itv[1] - itv[0], value=lab)         
            
            hier_anno.append(lvl_anno)   
        return hier_anno
        
    
    
    def annos_mir_eval(self) -> List[tuple]:
        upper_annos = self.jam.search(namespace='segment_salami_upper')
        lower_annos = self.jam.search(namespace='segment_salami_lower')
        upper_inter_vals = [a.to_interval_values() for a in upper_annos]
        upper_intervals  = [pair[0] for pair in upper_inter_vals]
        upper_vals = [pair[1] for pair in upper_inter_vals]
        lower_inter_vals = [a.to_interval_values() for a in lower_annos]
        lower_intervals = [pair[0] for pair in lower_inter_vals]
        lower_vals = [pair[1] for pair in lower_inter_vals]

        hier_intervals = [list(hier_interval) for hier_interval in zip(upper_intervals, lower_intervals)]
        hier_values = [list(hier_value) for hier_value in zip(upper_vals, lower_vals)]
        return [hier_anno for hier_anno in zip(hier_intervals, hier_values)]
    

    def meet_mats(self, ts, no_rep=False):
        out = []
        for hier_anno in self.annos():
            square_meet_mat = _meet_mat(hier_anno, ts, no_rep=no_rep)
            # collect
            out.append(square_meet_mat)        
        return out
        
    
    def sdm(self, feat='chroma', distance='cosine'): 
        # npy path
        sdm_path = os.path.join(self.salami_dir, f'sdms/{self.tid}_{feat}_{distance}.npy')
        
        # see if desired sdm is already computed
        if not os.path.exists(sdm_path):
            # compute sdm
            feat_mat, feat_ts = feature.prep(self, feat)
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


#     def rhos(
#         self, 
#         rho_mode: str = 'local',
#         k: int = 1, 
#         recompute: bool = False,
#     ) -> List[float]:
#         log_path = os.path.join(self.salami_dir, f'rhos/{self.tid}.json')
#         # create blank file if it doesn't exist already
#         if not os.path.exists(log_path):
#             with open(log_path, 'w+') as f:
#                 json.dump({'num_annos': self.num_annos}, f)
        
#         feature_types = ['openl3', 'yamnet', 'mfcc', 'chroma', 'crema', 'tempogram']
#         dist_types = ['cosine'] #, 'euclidean']
        
#         with open(log_path, 'r') as f:
#             score_dict = json.load(f)
        
#         for feat in feature_types:
#             for dist in dist_types:
#                 for anno_idx in range(self.num_annos):
#                     score_key = f'{feat}_{dist}_{rho_mode}_k{k}_a{anno_idx}'
#                     if recompute or score_key not in score_dict.keys():
#                         meet_mat = self.meet_mats(feat_ts)[anno_idx]
#                         print(f'rho with {feat} {dist} {rho_mode} k:{k} anno_id{anno_idx}')
#                         score_dict[score_key] = compute_rho(self.sdm(feat, dist), meet_mat, rho_mode, k)
        
#         with open(log_path, 'w') as f:
#             json.dump(score_dict, f)

#         return score_dict            

    def taus(self, recompute=False, loc_mask=True, k=30):
        log_path = os.path.join(self.salami_dir, f'taus/{self.tid}.json')
        if not os.path.exists(log_path):
            with open(log_path, 'w+') as f:
                json.dump(
                    {'loc': {}, 'rep': {}}, 
                    f
                )
        with open(log_path, 'r') as f:
            score_dict = json.load(f)

        marker = f'lm_{k}' if loc_mask else 'l'
        
        for feat in spliter.AVAL_FEAT_TYPES:
            if recompute or feat not in score_dict['rep'].keys():
                # compute
                meet_mats = self.meet_mats(self.common_ts(), no_rep=False)
                taus = []
                for meet_mat in meet_mats:
                    taus.append(-stats.kendalltau(self.sdm(feat=feat).flatten(), meet_mat.flatten())[0])
                score_dict['rep'][feat] = taus
                
            if recompute or feat not in score_dict['loc'].keys() or f'{marker}' not in score_dict['loc'][feat].keys():
                # compute
                meet_mats = self.meet_mats(self.common_ts(), no_rep=True)
                
                sdm = self.sdm(feat=feat)
                if loc_mask:
                    mask = mask_sdm(sdm, k=k)
                    sdm = np.multiply(mask, sdm) + (-mask + 1) * sdm.flatten().max()
                
                taus = []
                for meet_mat in meet_mats:
                    taus.append(-stats.kendalltau(sdm.flatten(), meet_mat.flatten())[0])
                
                if feat not in score_dict['loc'].keys():
                    score_dict['loc'][feat] = {}
                score_dict['loc'][feat][f'{marker}'] = taus

        with open(log_path, 'w') as f:
            json.dump(score_dict, f)
        return score_dict 
        

    def min_num_frames(self):
        if self.min_frame is None:
            # calculate what's the min frame len for all featuers
            num_frames = np.asarray([
                len(feature.openl3(self)['ts']),
                len(feature.yamnet(self)['ts']),
                len(feature.chroma(self)['ts']),
                len(feature.crema(self)['ts']),
                len(feature.tempogram(self)['ts']),
                len(feature.mfcc(self)['ts'])
            ])
            self.min_frame = np.min(num_frames)
        return self.min_frame

    def common_ts(self):
        return feature.mfcc(self)['ts'][:self.min_num_frames()]

    
    def lsd(self, config=None):
        if config is None:
            config = {'rec_width': 13, 
                      'rec_smooth': 7, 
                      'evec_smooth': 13,
                      'rep_ftype': 'chroma', 
                      'loc_ftype': 'mfcc',
                      'rep_metric': 'cosine',
                      'hier': True,
                      'num_layers': 10}

        rep_f, rep_ts = feature.prep(self, config['rep_ftype'])
        loc_f, loc_ts = feature.prep(self, config['loc_ftype'])

        # Spectral Clustering with Config
        est_bdry_idxs, est_sgmt_labels = sc.do_segmentation(rep_f, loc_f, config)

        # Post processing and ready for mir_eval.hierarchy.lmeasure
        # convert list of frame_indices to times in seconds
        est_bdry_times = [librosa.frames_to_time(frames, sr=22050, hop_length=4096) for frames in est_bdry_idxs]
        est_intervals = [mir_eval.util.boundaries_to_intervals(est_bdry) for est_bdry in est_bdry_times]
        return est_intervals, est_sgmt_labels


    def l_scores(self, config=None, recompute=False):
        # check if l_score already computed with the same config
        scores_path = os.path.join(self.salami_dir, f'l_scores/{self.tid}.json')
        if not os.path.exists(scores_path):
            with open(scores_path, 'w+') as f:
                json.dump([], f)
        with open(scores_path, 'r') as f:
            scores_dict = json.load(f)
        for trial in scores_dict:
            if trial['config'] == config and not recompute:
                # print('reading from disk...')
                return trial
            else:
                continue
        
        # now it didn't find it... calculate and store it!
        est_intervals, est_sgmt_labels = self.lsd(config)
        # for each annotator, store the lmeasure in a list
        l_score_list = []
        for hier_interval, hier_val in self.annos_mir_eval():
            # make last segment for estimation end at the same time as annotation
            end = max(hier_interval[-1][-1, 1], est_intervals[-1][-1, 1])
            for i in range(len(est_intervals)):
                est_intervals[i][-1, 1] = end
            for i in range(len(hier_interval)):
                hier_interval[i][-1, 1] = end
                
            # compute l-measure:
            l_score_list.append(mir_eval.hierarchy.lmeasure(hier_interval, hier_val, est_intervals, est_sgmt_labels))
        trial = {'config': config, 'l_scores': l_score_list}
        scores_dict.append(trial)
        with open(scores_path, 'w') as f:
            json.dump(scores_dict, f)
        return trial
    
    def l_scores_justin(self):
        result_dir = '/scratch/qx244/data/ISMIR21-Segmentations/SALAMI/def_mu_0.1_gamma_0.1/'
        filename = f'{self.tid}.mp3.msdclasscsnmagic.json'
        l_score_filename = f'{self.tid}.l_scores.json'
        
        scores_path = os.path.join(result_dir, l_score_filename)
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                return json.load(f)
        else: # calculate l-score and store in scores_path
            with open(os.path.join(result_dir, filename), 'rb') as f:
                justin_result = json.load(f)

            est_intervals = [np.array(layer[0]) for layer in justin_result]
            est_sgmt_labels = [np.array(layer[1]) for layer in justin_result]
            # for each annotator, store the lmeasure in a list
            l_score_list = []
            for hier_interval, hier_val in self.annos_mir_eval():
                # make last segment for estimation end at the same time as annotation
                end = max(hier_interval[-1][-1][1], est_intervals[-1][-1][1])
                for i in range(len(est_intervals)):
                    est_intervals[i][-1][1] = end
                for i in range(len(hier_interval)):
                    hier_interval[i][-1][1] = end

                # compute l-measure:
                l_score_list.append(
                    mir_eval.hierarchy.lmeasure(hier_interval, hier_val, est_intervals, est_sgmt_labels)
                )
                
            with open(scores_path, 'w+') as f:
                json.dump(l_score_list, f)            
        return l_score_list


################ HELPER FUNCTIONS ##########        
# def compute_rho(
#     tri_sdm: np.ndarray, # not in square form!
#     meet_mat: np.ndarray, # in square form! main diagonal should be 0's.
#     rho_mode: str = 'local', #{'local', 'rep', 'full'}
#     k: int = 1,
# ) -> float:
#     # some protection against square form or not for sdm
#     if len(tri_sdm.shape) == 1:
#         sdm = spatial.distance.squareform(tri_sdm)
#     else:
#         sdm = tri_sdm

#     if rho_mode == 'full':
#         return -stats.spearmanr(sdm, meet_mat)[0]
#     else:
#         # generate the index of where the local and rep 
#         n_frames = sdm.shape[0]
#         assert k+1 < n_frames
#         if rho_mode == 'local':
#             diags = range(1, k+1)
#         elif rho_mode == 'rep':
#             diags = range(k+1, n_frames)
        
#         # using sparse mats to get indices
#         smat = sparse.diags([1]*len(diags), offsets=diags, shape=sdm.shape)
#         idx =  smat.nonzero() 
#         return -stats.spearmanr(sdm[idx], meet_mat[idx])[0]

    
def mask_sdm(sdm, k=13):
    n_frames = sdm.shape[0]
    if k+1 >= n_frames:
        k = n_frames - 2
    diags = np.arange(1, k+1)
    
    # using sparse mats to mask
    masku = sparse.diags([1]*len(diags), offsets=diags, shape=sdm.shape)
    maskl = sparse.diags([1]*len(diags), offsets=-diags, shape=sdm.shape)
    mask = (masku+maskl).todense() + np.eye(n_frames)
    return mask
    
    
# def compute_tau_loc(
#     tri_sdm: np.ndarray, # not in square form!
#     meet_mat: np.ndarray, # in square form! main diagonal should be 0's.
#     k: int = 1,
# ):
#     if len(tri_sdm.shape) == 1:
#         sdm = spatial.distance.squareform(tri_sdm)
#     else:
#         sdm = tri_sdm
        
#     # generate the index of where the local and rep 
#     n_frames = sdm.shape[0]
#     assert k+1 < n_frames
#     diags = np.arange(1, k+1)

#     # using sparse mats to mask
#     masku = sparse.diags([1]*len(diags), offsets=diags, shape=sdm.shape)
#     maskl = sparse.diags([1]*len(diags), offsets=-diags, shape=sdm.shape)
#     mask = (masku+maskl).todense()
    
#     loc_sdm = np.multiply(mask, sdm)
#     out = -stats.kendalltau(loc_sdm.flatten(), meet_mat.flatten())[0]
#     return out


def _meet_mat(hier_anno, ts, no_rep=False):
    n_level = len(hier_anno)
    n_frames = len(ts)
    meet_mat_per_level = np.zeros((n_level, n_frames, n_frames))

    # put meet mat of each level of hierarchy in axis=0
    for level in range(n_level):
        if no_rep:
            layer_anno = no_rep_label(hier_anno[level])
        else:
            layer_anno = hier_anno[level]
            
        label_list = layer_anno.to_samples(
            librosa.times_like(n_frames, sr=1 / (ts[1] - ts[0]), hop_length=1)
        )
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform([l[0] if len(l) > 0 else 'NL' for l in label_list])
        meet_mat_per_level[level] = np.equal.outer(encoded_labels, encoded_labels).astype('float') * (level + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def _meet_mat_diag(hier_anno, ts):
    n_level = len(hier_anno)
    n_frames = len(ts)
    meet_mat_per_level = np.zeros((n_level, n_frames, n_frames))

    # put meet mat of each level of hierarchy in axis=0
    for level in range(n_level):
        label_list = hier_anno[level].to_samples(
            librosa.times_like(n_frames, sr=1 / (ts[1] - ts[0]), hop_length=1)
        )
        le = preprocessing.LabelEncoder()
        encoded_labels = le.fit_transform([l[0] if len(l) > 0 else 'NL' for l in label_list])
        meet_mat_per_level[level] = np.equal.outer(encoded_labels, encoded_labels).astype('float') * (level + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def no_rep_label(anno):
    # Given a salami anno. return the same anno with distinct labels for each section
    new_anno =jams.Annotation(namespace='segment_open')
    for i, obs in enumerate(anno.data):
        new_anno.append(time=obs.time, duration=obs.duration, value=i)
    return new_anno