import xarray as xr
import numpy as np
import ssdm

def sel_samp_l(ds, m_type='f', recollect=False):
    # It should be in the top 1/3 of all the scores in a track
    # AND
    # It should be in the top 1/3 of all l scores
    ds_scores_full = get_lsd_scores(ds, heir=True, beat_sync=ds.beat_sync, recollect=recollect).sel(m_type=m_type)
    best_layer_scores = ds_scores_full.max('layer')
    best_layers = adjusted_best_layer(ds_scores_full, tolerance=0)
    
    train_ds = type(ds)(split='train', mode=ds.mode, infer=True, beat_sync=ds.beat_sync)
    train_scores_full = get_lsd_scores(train_ds)
    best_layer_scores_train = train_scores_full.max('layer')
    train_ds_low_cut = np.percentile(best_layer_scores_train, 33)
    train_ds_high_cut = np.percentile(best_layer_scores_train, 66)

    per_track_hc=dict()
    per_track_lc=dict()
    for tr in best_layer_scores:
        per_track_lc[tr.tid.item()] = np.percentile(tr, 33)
        per_track_hc[tr.tid.item()] = np.percentile(tr, 66)
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
