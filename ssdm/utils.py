import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch import nn

from sklearn import preprocessing, cluster
from sklearn.model_selection import train_test_split
from scipy import sparse
from tqdm import tqdm

import random, os
import jams, mir_eval

import ssdm
import ssdm.scluster as sc
import ssdm.scanner as scn


def anno_to_meet(
    anno: jams.Annotation,  # multi-layer
    ts: list,
    num_layers: int = None,
) -> np.array:
    """
    returns a square mat that's the meet matrix
    """
    #sample the annotation first and input cleaning:
    sampled_anno = anno.to_samples(ts)
    if num_layers is None:
        num_layers = len(sampled_anno[0])
    n_frames = len(ts)
    
    # initialize a 3d array to store all the meet matrices for each layer
    meet_mat_per_level = np.zeros((num_layers, n_frames, n_frames))

    # build hier samples list of list
    hier_labels = [[''] * n_frames for _ in range(num_layers)]

    # run through sampled observations in multi_seg anno
    for t, sample in enumerate(sampled_anno):
        for label_per_level in sample:
            lvl = label_per_level['level']
            label = label_per_level['label']
            if lvl < num_layers:
                hier_labels[lvl][t] = label

    
    # clean labels
    le = preprocessing.LabelEncoder()
    hier_encoded_labels = []
    for lvl_label in hier_labels:
        hier_encoded_labels.append(
            le.fit_transform([l if len(l) > 0 else 'NL' for l in lvl_label])
        )

    # put meet mat of each level of hierarchy in axis=0           
    for l in range(num_layers):
        meet_mat_per_level[l] = np.equal.outer(hier_encoded_labels[l], hier_encoded_labels[l]).astype('float') * (l + 1)

    # get the deepest level matched
    return np.max(meet_mat_per_level, axis=0)


def meet_mat_no_diag(track, rec_mode='expand', diag_mode='refine', anno_id=0):
    diag_block = ssdm.anno_to_meet(track.ref(mode=diag_mode, anno_id=anno_id), ts=track.ts())
    full_rec = ssdm.anno_to_meet(track.ref(mode=rec_mode, anno_id=anno_id), ts=track.ts())
    return (diag_block == 0) * full_rec


def compute_l(
    proposal: jams.Annotation, 
    annotation: jams.Annotation,
    l_frame_size: float = 0.1,
    nlvl = None,
) -> np.array:
    anno_interval, anno_label = ssdm.multi2mireval(annotation)
    proposal_interval, proposal_label = ssdm.multi2mireval(proposal)
    if nlvl:
        assert nlvl > 0
        proposal_interval = proposal_interval[:nlvl]
        proposal_label = proposal_label[:nlvl]

    # make last segment for estimation end at the same time as annotation
    end = max(anno_interval[-1][-1, 1], proposal_interval[-1][-1, 1])
    for i in range(len(proposal_interval)):
        proposal_interval[i][-1, 1] = end
    for i in range(len(anno_interval)):
        anno_interval[i][-1, 1] = end

    return mir_eval.hierarchy.lmeasure(
        anno_interval, anno_label, proposal_interval, proposal_label, 
        frame_size=l_frame_size
    )


def compute_flat(
    proposal: jams.Annotation, #Multi-seg
    annotation: jams.Annotation, #Multi-seg,
    # p_layer = -1,
    a_layer = -1
) -> np.array:
    # TODO change the following 2 lines
    ref_inter, ref_labels = ssdm.multi2mirevalflat(annotation, layer=a_layer) # Which anno layer?

    num_prop_layers = len(ssdm.multi2hier(proposal))

    # get empty dataarray for result
    results_dim = dict(
        m_type=['p', 'r', 'f'],
        # metric=['hr', 'hr3', 'pfc', 'nce'],
        metric=['pfc'],
        layer=[x+1 for x in range(16)]
    )
    results = xr.DataArray(data=None, coords=results_dim, dims=list(results_dim.keys()))
    for p_layer in range(num_prop_layers):
        est_inter, est_labels = ssdm.multi2mirevalflat(proposal, layer=p_layer)
        # make last segment for estimation end at the same time as annotation
        end_time = max(ref_inter[-1, 1], est_inter[-1, 1])
        ref_inter[-1, 1] = end_time
        est_inter[-1, 1] = end_time
        
        layer_result_dict = dict(
            # hr=mir_eval.segment.detection(ref_inter, est_inter, window=.5, trim=True),
            # hr3=mir_eval.segment.detection(ref_inter, est_inter, window=3, trim=True),
            pfc=mir_eval.segment.pairwise(ref_inter, ref_labels, est_inter, est_labels),
            # nce=mir_eval.segment.vmeasure(ref_inter, ref_labels, est_inter, est_labels)
        )

        for metric in layer_result_dict:
            results.loc[dict(layer=p_layer+1, metric=metric)] = list(layer_result_dict[metric])
    return results


def pick_by_taus(
    scores_grid: xr.DataArray, # tids * num_feat * num_feat
    rep_taus: xr.DataArray, # tids * num_feat
    loc_taus: xr.DataArray = None, # tids * num_feat
) -> pd.DataFrame: # tids * ['rep', 'loc', 'score', 'orc_rep', 'orc_loc', 'oracle']
    """pick the best rep and loc features according to taus from a scores_grid"""
    out = pd.DataFrame(index=scores_grid.tid, 
                       columns=['rep_pick', 'loc_pick', 'score', 'orc_rep_pick', 'orc_loc_pick', 'oracle']
                       )
    
    out.orc_rep_pick = scores_grid.max(dim='loc_ftype').idxmax(dim='rep_ftype').squeeze()
    out.orc_loc_pick = scores_grid.max(dim='rep_ftype').idxmax(dim='loc_ftype').squeeze()
    out.oracle = scores_grid.max(dim=['rep_ftype', 'loc_ftype']).squeeze()
    
    rep_pick = rep_taus.idxmax(dim='f_type').sortby('tid')
    
    if loc_taus is None:
        loc_pick = out.orc_loc_pick.to_xarray().sortby('tid')
    else:
        loc_pick = loc_taus.idxmax(dim='f_type').fillna('mfcc').sortby('tid')

    out.rep_pick = rep_pick
    out.loc_pick = loc_pick
    out.score = scores_grid.sortby('tid').sel(rep_ftype=rep_pick, loc_ftype=loc_pick)
    
    return out


def pick_by_net(ds, model_path='', return_raw_result=False, recollect=False, multi_loss=False):
    assert ds.mode == 'both'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #extract info from model_path
    model_basename = os.path.basename(model_path)
    # Check for cached inference results:
    ml_str = 'multi_loss' if multi_loss else ''
    infer_result_fp = os.path.join(ds.output_dir, f'tauhats/{model_basename}_{ds}{ml_str}.nc')
    try:
        if recollect:
            raise NotImplementedError
        inference_result = xr.load_dataarray(infer_result_fp)
    except:
        print('could not load cache, doing inference')
        
        # check all the available models in scanner.py to see if anyone fits
        net = None
        for model_id in scn.AVAL_MODELS:
            if model_basename.find(model_id + '_') >= 0:
                print(model_basename)
                print(model_id)
                # returns the first model found in the file base name
                net = scn.AVAL_MODELS[model_id]()
                continue
        if net == None:
            raise IndexError('could not figure out which model architecutre to initialize.')      
        # load model and do inference
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict=state_dict)
        if not ds.infer:
            if multi_loss:
                return scn.net_eval_multi_loss(ds, net, nn.BCELoss(), nn.MSELoss(), device=device)
            else:
                return scn.net_eval(ds, net, nn.BCELoss(), device=device)
        else:
            if multi_loss:
                inference_result = scn.net_infer_multi_loss(ds, net=net, device=device)
            else:
                inference_result = scn.net_infer(ds, net=net, device=device)
    try:
        inference_result.to_netcdf(infer_result_fp)
    except:
        print('failed to cache result...')

    # return inference_result
    # Return either the raw inference result or the best choosen scores
    if return_raw_result:
        return inference_result
    else:
        # collection
        if multi_loss:
            util_score = inference_result.loc[:, :, :, 'util']
            nlvl_score = inference_result.loc[:, :, :, 'nlvl']
            best_chocie_idx = util_score.argmax(dim=['rep_ftype', 'loc_ftype'])
            best_chocie_idx
            return nlvl_score.isel(best_chocie_idx).clip(max=16, min=2) # from index to layer
        else:
            best_chocie_idx = inference_result.argmax(dim=['rep_ftype', 'loc_ftype'])
            return inference_result.isel(best_chocie_idx) # from index to layer


def quantize(data, quantize_method='percentile', quant_bins=8):
    # method can me 'percentile' 'kmeans'. Everything else will be no quantize
    data_shape = data.shape
    if quantize_method == 'percentile':
        bins = [np.percentile(data[data > 0], bin * (100.0/(quant_bins - 1))) for bin in range(quant_bins)]
        # print(bins)
        quant_data_flat = np.digitize(data.flatten(), bins=bins, right=False)
    elif quantize_method == 'kmeans':
        kmeans_clusterer = cluster.KMeans(n_clusters=quant_bins)
        quantized_non_zeros = kmeans_clusterer.fit_predict(data[data>0][:, None])
        # make sure the kmeans group are sorted with asending centroid and relabel
        new_ccenter_order = np.argsort(kmeans_clusterer.cluster_centers_.flatten())
        nco = new_ccenter_order[new_ccenter_order]
        quantized_non_zeros = np.array([nco[g] for g in quantized_non_zeros], dtype=int)

        quant_data = np.zeros(data.shape)
        quant_data[data>0] = quantized_non_zeros + 1
        quant_data_flat = quant_data.flatten()
    elif quantize_method is None:
        quant_data_flat = data.flatten()
    else:
        assert('bad quantize method')

    return quant_data_flat.reshape(data_shape)


def laplacian(rec_mat, normalization='random_walk'):
    degree_matrix = np.diag(np.sum(rec_mat, axis=1))
    unnormalized_laplacian = degree_matrix - rec_mat
    # Compute the Random Walk normalized Laplacian matrix
    if normalization == 'random_walk':
        degree_inv = np.linalg.inv(degree_matrix)
        return degree_inv @ unnormalized_laplacian       
    elif normalization == 'symmetrical':
        sqrt_degree_inv = np.linalg.inv(np.sqrt(degree_matrix))
        return sqrt_degree_inv @ unnormalized_laplacian @ sqrt_degree_inv
    elif normalization == None:
        return unnormalized_laplacian
    else:
        raise NotImplementedError(f'bad laplacian normalization mode: {normalization}')


# Create test train val splits on the fly, but with random seed.
def create_splits(arr, val_ratio=0.15, test_ratio=0.15, random_state=20230327):
    dev_set, test_set = train_test_split(arr, test_size = test_ratio, random_state = random_state)
    train_set, val_set = train_test_split(dev_set, test_size = val_ratio / (1 - test_ratio), random_state = random_state)
    train_set.sort()
    val_set.sort()
    test_set.sort()
    return dict(train=train_set, val=val_set, test=test_set)


def adjusted_best_layer(cube, tolerance=0):
    max_score_over_layer = cube.max('layer')
    # We can tolerate a bit of perforamcne, so let's take that off from the max
    thresh = max_score_over_layer - tolerance
    # The first layer over threshold
    over_thresh = cube >= thresh
    return over_thresh.idxmax('layer')


# HELPER functin to run laplacean spectral decomposition TODO this is where the rescaling happens
def run_lsd(
    track,
    config: dict,
    beat_sync: bool = False,
    recompute_ssm: bool = False,
    loc_sigma: float = 95
) -> jams.Annotation:
    def mask_diag(sq_mat, width=2):
        # carve out width from the full ssm
        sq_mat_lil = sparse.lil_matrix(sq_mat)
        for diag in range(-width + 1, width):
            sq_mat_lil.setdiag(0, diag)
        return sq_mat_lil.toarray()

    # Compute/get SSM mats
    rep_ssm = track.ssm(feature=config['rep_ftype'],
                        distance=config['rep_metric'],
                        width=config['rec_width'],
                        full=bool(config['rec_full']),
                        recompute=recompute_ssm,
                        beat_sync=beat_sync,
                        add_noise=config['add_noise'],
                        n_steps = config['n_steps'],
                        delay = config['delay']
                        )
    
    if config['rec_full']:
        rep_ssm = mask_diag(rep_ssm, width=config['rec_width'])

    # track.path_sim alwasy recomputes.
    path_sim = track.path_sim(feature=config['loc_ftype'],
                              distance=config['loc_metric'],
                              sigma_percentile=loc_sigma,
                              recompute=True,
                              beat_sync=beat_sync,
                              add_noise=config['add_noise'],
                              n_steps = config['n_steps'],
                              delay = config['delay'])

    # Spectral Clustering with Config
    est_bdry_idxs, est_sgmt_labels = sc.do_segmentation_ssm(rep_ssm, path_sim, config)
    # print(est_bdry_idxs, est_sgmt_labels)
    if beat_sync:
        ts = track.ts(mode='beat', pad=True)
    else:
        ts = track.ts(mode='frame')
    est_bdry_itvls = [sc.times_to_intervals(ts[lvl]) for lvl in est_bdry_idxs]
    return ssdm.mireval2multi(est_bdry_itvls, est_sgmt_labels)


def get_lsd_scores(
    ds,
    shuffle=False,
    anno_col_fn=lambda stack: stack.max(dim='anno_id'), # activated when there are more than 1 annotation for a track
    heir=True,
    beat_sync=True,
    recollect=False,
    **lsd_score_kwargs,
) -> xr.DataArray:
    ds_str = str(ds).replace('rep', '').replace('loc', '')
    sync_str = "_bsync" if beat_sync else ""
    if heir:
        save_path = f'/vast/qx244/{ds_str}_{len(ds.tids)}_heir{sync_str}_lsd_scores.nc'
    else:
        save_path = f'/vast/qx244/{ds_str}_{len(ds.tids)}_flat{sync_str}_lsd_scores.nc'

    if 'custom' in str(ds):
        recollect = True
    if recollect or (not os.path.exists(save_path)):
        score_per_track = []
        tids = ds.tids
        if shuffle:
            random.shuffle(tids)

        for tid in tqdm(tids):
            track = ds.track_obj(tid=tid)
            if track.num_annos() == 1:
                if heir:
                    score_per_track.append(track.new_lsd_score(beat_sync=beat_sync, **lsd_score_kwargs))
                else:
                    score_per_track.append(track.lsd_score_flat(beat_sync=beat_sync, **lsd_score_kwargs))
            else:
                score_per_anno = []
                for anno_id in range(track.num_annos()):
                    if heir:
                        score_per_anno.append(track.new_lsd_score(anno_id=anno_id, beat_sync=beat_sync, **lsd_score_kwargs))
                    else:
                        score_per_anno.append(track.lsd_score_flat(anno_id=anno_id, beat_sync=beat_sync, **lsd_score_kwargs))
                # print(score_per_anno)
                anno_stack = xr.concat(score_per_anno, pd.Index(range(len(score_per_anno)), name='anno_id'))
                score_per_track.append(anno_col_fn(anno_stack))

        out = xr.concat(score_per_track, pd.Index(tids, name='tid'), coords='minimal').rename()
        try:
            out.to_netcdf(save_path)
        except:
            os.system(f'rm {save_path}')
            out.to_netcdf(save_path)

    return xr.load_dataarray(save_path).sortby('tid')


def get_taus(
    ds,
    shuffle=False,
    anno_col_fn=lambda stack: stack.max(dim='anno_id'), # activated when there are more than 1 annotation for a track
    recollect=False,
    beat_sync=True,
    **tau_kwargs,
):
    if ds.infer:
        infer_flag = 'infer'
    else:
        infer_flag = 'train'

    sync_str = "_bsync" if beat_sync else ""
    if ds.mode == 'both':
        save_path = os.path.join(ds.track_obj().output_dir, f'{ds}_{infer_flag}_{ds.lap_norm}{sync_str}_blocky_taus.nc')
    else:
        save_path = os.path.join(ds.track_obj().output_dir, f'{ds}_{infer_flag}{sync_str}_blocky_taus.nc')
    
    try:
        out = xr.load_dataarray(save_path)
    except:
        recollect = True

    if recollect:
        tau_per_track = []
        tids = ds.tids
        if shuffle:
            random.shuffle(tids)
        for tid in tqdm(tids):
            track = ds.track_obj(tid=tid)
            tau_per_anno = []
            for anno_id in range(track.num_annos()):
                if ds.mode == 'both':
                    tau_per_anno.append(track.tau_both(anno_id=anno_id, lap_norm=ds.lap_norm, beat_sync=beat_sync, **tau_kwargs))
                else:
                    tau_per_anno.append(track.tau(anno_id=anno_id, beat_sync=beat_sync, **tau_kwargs))
            anno_stack = xr.concat(tau_per_anno, pd.Index(range(len(tau_per_anno)), name='anno_id'))
            tau_per_track.append(anno_col_fn(anno_stack))

        out = xr.concat(tau_per_track, pd.Index(tids, name='tid'), coords='minimal')
        out.rename().sortby('tid')
        try: 
            out.to_netcdf(save_path)
        except:
            os.system(f'rm {save_path}')
            out.to_netcdf(save_path)
    
    return out.sortby('tid')


def net_pick_performance(ds, net, heir=True):
    score_da = get_lsd_scores(ds, heir=heir)
    if not heir:
        score_da = score_da.sel(metric='pfc')
    net_output = scn.net_infer_multi_loss(ds, net, ds.device)
    util_score = net_output.loc[:, :, :, 'util']
    nlvl_pick = net_output.loc[:, :, :, 'nlvl']
    best_util_idx = util_score.argmax(dim=['rep_ftype', 'loc_ftype'])
    net_pick = nlvl_pick.isel(best_util_idx).astype(int)
    return score_da.sel(rep_ftype=net_pick.rep_ftype, loc_ftype=net_pick.loc_ftype).isel(layer=net_pick)

# def dataset_performance(score_da, tau_hat_rep, tau_hat_loc, heir=False):
#     if heir:
#         working_da = score_da.sortby('tid')
#     else:
#         best_layer = score_da.sel(m_type='f').idxmax(dim='layer', fill_value=4)
#         working_da = score_da.sel(layer=best_layer.drop_vars('m_type')).sortby('tid')

#     rep_pick = tau_hat_rep.idxmax(dim='f_type').sortby('tid')
#     loc_pick = tau_hat_loc.idxmax(dim='f_type').sortby('tid')

#     tau_hat_rep_score = working_da.sel(rep_ftype=rep_pick).drop_vars('rep_ftype').expand_dims(rep_ftype=['tau_hat'])
#     tau_hat_loc_score = working_da.sel(loc_ftype=loc_pick).drop_vars('loc_ftype').expand_dims(loc_ftype=['tau_hat'])
#     tau_hat_both_score = working_da.sel(rep_ftype=rep_pick, loc_ftype=loc_pick).drop_vars(['loc_ftype', 'rep_ftype']).expand_dims(loc_ftype=['tau_hat'], rep_ftype=['tau_hat'])

#     score_with_tau_rep = xr.concat([working_da, tau_hat_rep_score], dim='rep_ftype')
#     full_tau_loc_score = xr.concat([tau_hat_loc_score, tau_hat_both_score], dim='rep_ftype')
#     full_score = xr.concat([score_with_tau_rep, full_tau_loc_score], dim='loc_ftype')
#     return full_score


# def full_performance(ds, rep_model='RepNet20240303_epoch28', loc_model='LocNet20240303_epoch17', multi_model='', both=False):
#     # Performance with flat and heir scores
#     ds_str = str(ds).replace('loc', 'rep')
#     tau_hat_rep = xr.open_dataarray(f'/vast/qx244/salami/tau_hat_{ds_str}-{rep_model}.nc')
#     ds_str = str(ds).replace('rep', 'loc')
#     tau_hat_loc = xr.open_dataarray(f'/vast/qx244/salami/tau_hat_{ds_str}-{loc_model}.nc')
    
    
#     sda_flat = get_lsd_scores(ds, shuffle=False, heir=False)
#     sda_heir = get_lsd_scores(ds, shuffle=False, heir=True)
#     full_score_flat = dataset_performance(sda_flat, tau_hat_rep, tau_hat_loc, heir=False)
#     full_score_heir = dataset_performance(sda_heir, tau_hat_rep, tau_hat_loc, heir=True)
#     full_score = xr.concat(
#         [full_score_flat.drop_vars('layer'),
#          full_score_heir.rename(l_type='m_type').assign_coords(m_type=['p', 'r', 'f']).expand_dims(metric=['l'])],
#         dim='metric'
#     )
    
#     ds_str = str(ds).replace('loc', '').replace('rep', '')
#     return full_score.assign_coords(tid=ds_str + full_score.tid)


# def dev_deploy_perf(
#     dev_ds, all_ds_list, 
#     m_type='r', metric='l', 
#     rep_model='RepNet20240303_epoch28', 
#     loc_model='LocNet20240303_epoch17',
#     drop_feats=[]
# ):
#     # look at the dev_ds and pick the best pair of features on average
#     dev_mean_score = full_performance(dev_ds, rep_model=rep_model, loc_model=loc_model).sel(m_type=m_type, metric=metric).mean('tid')
#     dev_mean_score = dev_mean_score.drop_sel(rep_ftype=drop_feats + ['tau_hat'], loc_ftype=drop_feats + ['tau_hat'])
#     dev_rep_pick = dev_mean_score.max(dim='loc_ftype').idxmax(dim='rep_ftype').item()
#     dev_loc_pick = dev_mean_score.max(dim='rep_ftype').idxmax(dim='loc_ftype').item()
    
#     # Don't include the dev_ds in the deploy set, and combine all other ds.
#     deploy_sda_list = []
#     for ds in all_ds_list:
#         if type(ds) is not type(dev_ds):
#             deploy_sda_part = full_performance(ds, rep_model=rep_model, loc_model=loc_model).sel(m_type=m_type, metric=metric)
#             deploy_sda_part = deploy_sda_part.drop_sel(rep_ftype=drop_feats, loc_ftype=drop_feats)
#             deploy_sda_list.append(deploy_sda_part)
#     deploy_sda = xr.concat(deploy_sda_list, dim='tid', coords='minimal')

#     deploy_naive_perf = deploy_sda.sel(rep_ftype=dev_rep_pick, loc_ftype=dev_loc_pick)

#     ## Get all the tau_hats!
#     tau_hat_reps = []
#     tau_hat_locs = []
#     for ds in all_ds_list:
#         if type(ds) is type(dev_ds):
# #             print('skipping the dev set')
#             continue
#         else:
#             ds_str = str(ds).replace('loc', 'rep')
#             tau_hat_rep = xr.open_dataarray(f'/vast/qx244/salami/tau_hat_{ds_str}-{rep_model}.nc')
#             tau_hat_rep = tau_hat_rep.drop_sel(f_type=drop_feats)
#             tau_hat_reps.append(tau_hat_rep.assign_coords(tid=ds_str.replace('rep', '') + tau_hat_rep.tid))
#             ds_str = str(ds).replace('rep', 'loc')
#             tau_hat_loc = xr.open_dataarray(f'/vast/qx244/salami/tau_hat_{ds_str}-{loc_model}.nc')
#             tau_hat_loc = tau_hat_loc.drop_sel(f_type=drop_feats)
#             tau_hat_locs.append(tau_hat_loc.assign_coords(tid=ds_str.replace('loc', '') + tau_hat_loc.tid))

#     deploy_th_rep = xr.concat(tau_hat_reps, dim='tid')
#     deploy_th_loc = xr.concat(tau_hat_locs, dim='tid')
#     deploy_tau_perf = pick_by_taus(deploy_sda, deploy_th_rep, deploy_th_loc)
    
#     return deploy_naive_perf, deploy_tau_perf, (dev_rep_pick, dev_loc_pick)


# DEPRECATED: old Sample selection functions
def select_samples_using_tau_percentile(ds, low=30, high=85, m_type='f'):
    ds_scores_full = get_lsd_scores(ds, heir=True, beat_sync=ds.beat_sync).sel(m_type=m_type)
    # best_layer_scores = ds_scores_full.max('layer')
    best_layers = adjusted_best_layer(ds_scores_full, tolerance=0)

    taus_train = ssdm.get_taus(type(ds)(split='train', infer=True, mode=ds.mode, lap_norm=ds.lap_norm,
        beat_sync=ds.beat_sync), shuffle=True)
    
    if ds.split and ds.split.find('custom') == -1:
        taus_full = ssdm.get_taus(type(ds)(split=ds.split, infer=True, mode=ds.mode, lap_norm=ds.lap_norm), 
                                  shuffle=True, beat_sync=ds.beat_sync)
    else:
        taus_full = ssdm.get_taus(type(ds)(tids=ds.tids, infer=True, mode=ds.mode, lap_norm=ds.lap_norm,
            sample_select_fn=ds.sample_select_fn), shuffle=True, beat_sync=ds.beat_sync, recollect=True)
    if ds.mode == 'both':
        tau_flat = taus_full.stack(sid=['tid', 'rep_ftype', 'loc_ftype'])
    else:
        tau_flat = taus_full.sel(tau_type=ds.mode).stack(sid=['tid', 'f_type'])
    neg_sids = tau_flat.where(tau_flat < np.percentile(taus_train, low), drop=True).indexes['sid']
    pos_sids = tau_flat.where(tau_flat > np.percentile(taus_train, high), drop=True).indexes['sid']
    neg_samples = {sid: (0, best_layers.loc[sid].item())  for sid in neg_sids if sid[0] in ds.tids}
    pos_samples = {sid: (1, best_layers.loc[sid].item()) for sid in pos_sids if sid[0] in ds.tids}

    print(f'{ds} has \n \t {len(pos_samples)} pos samples; {len(neg_samples)} neg samples.')
    return {**neg_samples, **pos_samples}


def sel_samp_l(ds, m_type='f', recollect=False, tr_high=85, tr_low=30, corp_high=80, corp_low=20):
    # It should be in the top 1/3 of all the scores in a track
    # AND
    # It should be in the top 1/3 of all l scores
    ds_scores_full = get_lsd_scores(ds, heir=True, beat_sync=ds.beat_sync, recollect=recollect).sel(m_type=m_type)
    best_layer_scores = ds_scores_full.max('layer')
    best_layers = adjusted_best_layer(ds_scores_full)
    
    train_ds = type(ds)(split='train', mode=ds.mode, infer=True, beat_sync=ds.beat_sync)
    train_scores_full = get_lsd_scores(train_ds, heir=True, beat_sync=ds.beat_sync, recollect=recollect).sel(m_type=m_type)
    best_layer_scores_train = train_scores_full.max('layer')
    train_ds_low_cut = np.percentile(best_layer_scores_train, corp_low)
    train_ds_high_cut = np.percentile(best_layer_scores_train, corp_high)
    # print(train_ds_low_cut, train_ds_high_cut)

    per_track_hc=dict()
    per_track_lc=dict()
    for tr in best_layer_scores:
        per_track_lc[tr.tid.item()] = np.percentile(tr, tr_low)
        per_track_hc[tr.tid.item()] = np.percentile(tr, tr_high)
    per_track_lc = xr.DataArray(
        list(per_track_lc.values()), 
        coords={'tid': list(per_track_lc.keys())}, 
        dims='tid'
    )
    per_track_hc = xr.DataArray(
        list(per_track_hc.values()), 
        coords={'tid': list(per_track_hc.keys())}, 
        dims='tid'
    )
    lc_loc = best_layer_scores <= per_track_lc
    hc_loc = best_layer_scores >= per_track_hc
    scores_flat = best_layer_scores.stack(sid=['tid', 'rep_ftype', 'loc_ftype'])
    lc_loc_flat = lc_loc.stack(sid=['tid', 'rep_ftype', 'loc_ftype'])
    hc_loc_flat = hc_loc.stack(sid=['tid', 'rep_ftype', 'loc_ftype'])

    neg_sids = scores_flat.where(
        (scores_flat <= train_ds_low_cut) * lc_loc_flat,
        drop=True,
    ).indexes['sid']
    pos_sids = scores_flat.where(
        (scores_flat >= train_ds_high_cut) * hc_loc_flat,
        drop=True,
    ).indexes['sid']
    # return scores_flat, train_ds_high_cut
    pos_samples = {s: (1, best_layers.loc[s].item()) for s in pos_sids}
    neg_samples = {s: (0, best_layers.loc[s].item()) for s in neg_sids}

    print(f'{ds} has \n \t {len(pos_samples)} pos samples; {len(neg_samples)} neg samples.')
    return {**neg_samples, **pos_samples}


def select_samples_using_outstanding_l_score(ds, neg_eps_pct=50, m_type='f'):
    scores_full = ssdm.get_lsd_scores(type(ds)(split='train', infer=True), heir=True).sel(m_type=m_type).max(dim='layer')
    best_on_avg_rep_feat = scores_full.mean(dim='tid').max(dim='loc_ftype').idxmax(dim='rep_ftype').item()
    best_on_avg_loc_feat = scores_full.mean(dim='tid').max(dim='rep_ftype').idxmax(dim='loc_ftype').item()

    if ds.split and ds.split.find('custom') == -1:
        ds_score = ssdm.get_lsd_scores(type(ds)(split=ds.split, infer=True), heir=True).sel(m_type=m_type)
    else:
        ds_score = ssdm.get_lsd_scores(type(ds)(tids=ds.tids, infer=True), heir=True).sel(m_type=m_type)

    diff_from_boa = ds_score - ds_score.sel(rep_ftype=best_on_avg_rep_feat, loc_ftype=best_on_avg_loc_feat)

    # Different ds.modes requires different treatment
    if ds.mode == 'both':
        diff_flat = diff_from_boa.stack(sid=['tid', 'rep_ftype', 'loc_ftype'])
    elif ds.mode == 'rep':
        diff_flat = diff_from_boa.mean(dim='loc_ftype').stack(sid=['tid', 'rep_ftype'])
    elif ds.mode == 'loc':
        diff_flat = diff_from_boa.mean(dim='rep_ftype').stack(sid=['tid', 'loc_ftype'])

    all_neg_perf_gaps = diff_flat.where(diff_flat < 0, drop=True)
    neg_eps = np.percentile(all_neg_perf_gaps, neg_eps_pct)
    neg_sids = diff_flat.where(diff_flat < neg_eps, drop=True).indexes['sid']
    pos_sids = diff_flat.where(diff_flat >= 0, drop=True).indexes['sid']

    pos_samples = {s: 1 for s in pos_sids if ds_score.loc[s].mean() >= np.median(ds_score)}
    neg_samples = {s: 0 for s in neg_sids if ds_score.loc[s].mean() <= np.median(ds_score)}

    print(f'{ds} has \n \t {len(pos_samples)} pos samples; {len(neg_samples)} neg samples.')
    return {**neg_samples, **pos_samples}
