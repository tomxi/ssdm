import numpy as np
import pandas as pd

import panel as pn
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from ssdm import viz

import torch

import xarray as xr

import ssdm

def scatter(perf_series1, perf_series2, xlabel='s1', ylabel='s2', title='Scores', score_gap=0, side='both'):
    # side can be 'both', s1' 's2'
    # put 2 series together into a dataframe
    df = pd.concat([perf_series1, perf_series2, perf_series1.index.to_series()], axis=1)
    df.columns = ['s1', 's2', 'tid']

    # get the gap smaller than score_gap out of the picture
    if side == 's2':
        df_gap = df[df.s2 - df.s1 >= score_gap]
    elif side == 's1':
        df_gap = df[df.s1 - df.s2 >= score_gap]
    elif side == 'both':
        df_gap = df[np.abs(df.s1 - df.s2) >= score_gap]
    else:
        raise AssertionError(f"bad side: {side}, can only be 'both', s1' 's2'.")
    
    # build some texts
    count_tx = hv.Text(0.2, 0.05, f'tracks in plot: {len(df_gap)}\n with gap={score_gap}')
    s1_tx = hv.Text(0.8, 0.05, f'{xlabel} is better: {len(df_gap[df_gap.s1 - df_gap.s2 > 0])}')
    s2_tx = hv.Text(0.2, 0.95, f'{ylabel} is better: {len(df_gap[df_gap.s1 - df_gap.s2 < 0])}')
    texts = s1_tx * s2_tx * count_tx
    # build the scatter plot
    plot = hv.Scatter(
        df_gap, kdims=['s1'], vdims=['s2', 'tid']).opts(
        tools=['hover'], frame_width=500, frame_height=500, size=3.5, 
        xlabel=xlabel, ylabel=ylabel, title=title,
    )
    # marker line that devides improve/worsen
    diag_line = hv.Curve(([0, 1], [0, 1])).opts(
        color='red', line_dash='dashed', line_width=2
    )
    # Marker lines that shows the gap (diagonal strip)
    gap_line_lower = hv.Curve(([score_gap, 1], [0, 1 - score_gap])).opts(
        color='blue', line_dash='dashed', line_width=1
    )
    gap_line_upper = hv.Curve(([0, 1 - score_gap], [score_gap, 1])).opts(
        color='blue', line_dash='dashed', line_width=1
    )
    diag_band = diag_line * gap_line_upper * gap_line_lower

    # overlay them all    s
    return (plot * diag_band * texts).opts(xlim=(0,1), ylim=(0,1))


def follow_along(track):
    # all the panel elements for selection etc.
    audio = pn.pane.Audio(track.audio_path, sample_rate=22050, name=track.tid, throttle=250, width=300)
    selectr = pn.widgets.Select(
        name='lsd rep feature', options=ssdm.AVAL_FEAT_TYPES, width=90, value='openl3'
    )
    selectl = pn.widgets.Select(
        name='lsd loc feature', options=ssdm.AVAL_FEAT_TYPES, width=90, value='mfcc'
    )
    sel_layers = pn.widgets.EditableIntSlider(
        name='LSD layers to show', 
        start=1, end=10, step=1, value=7, 
        fixed_start=1, fixed_end=10,
        width=160
    )
    selecta = pn.widgets.Select(
        name='anno id', options=[x for x in range(track.num_annos())], width=65
    )
    sel_hier = pn.widgets.Select(
        name='Hierarchy Expansion', options=['expand', 'normal', 'refine', 'coarse'], width=120
    )
    slider_tau_width = pn.widgets.DiscreteSlider(
        name='tau-rep width', options=[16, 22, 27, 30, 32, 54], value=30, width=170
    )
    lfs_dropdown = pn.widgets.Select(
        name='l frame size', options=[0.1, 0.5, 1], width=100)
    qm_sel = pn.widgets.Select(
        name='quantize method', options=[None, 'percentile'], value='percentile', width=100
    )
    qb_slider = pn.widgets.EditableIntSlider(
        name='Quantize Bins', 
        start=1, end=12, step=1, value=7, 
        fixed_start=1, fixed_end=10,
        width=170
    )
    # pn_text = pn.widgets.StaticText(name='Processing', value='Done!')
    
    # hv options
    options = [
        opts.Image(
            cmap='inferno',
            colorbar=True,
            aspect='equal',
            frame_width=300,
            frame_height=300,
        )
    ]
    hv.Dimension.type_formatters[np.float64]='%.3f'

    # lsd l score grid for all feature combos
    @pn.depends(anno_id=selecta, l_frame_size=lfs_dropdown)
    def lsd_score_heatmap(anno_id, l_frame_size):
        lsd_score = track.new_lsd_score(l_frame_size=l_frame_size, anno_id=anno_id).sel(l_type='lr')
        lsd_score_grid = hv.HeatMap(lsd_score).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable')
        score_label = hv.Labels(lsd_score_grid)
        return lsd_score_grid * score_label
    
    @pn.depends(anno_id=selecta, anno_mode=sel_hier, tau_width=slider_tau_width, quantize=qm_sel, quant_bins=qb_slider)
    def tau_heatmap(anno_id, anno_mode, tau_width, quantize, quant_bins):
        taus = track.tau(anno_id=anno_id, rec_width=tau_width, anno_mode=anno_mode, quantize=quantize, quant_bins=quant_bins, recompute=False)
        tau_grid = hv.HeatMap(taus).opts(frame_width=300, frame_height=100, cmap='coolwarm', toolbar='disable')
        score_label = hv.Labels(tau_grid)
        return tau_grid * score_label

    # def tau_hat_rep():
    #     tau_rep_hats = track.tau_hat_rep()
    #     tau_grid = hv.HeatMap(tau_rep_hats).opts(frame_width=300, frame_height=100, cmap='coolwarm', toolbar='disable')
    #     score_label = hv.Labels(tau_grid)

    @pn.depends(rep_feat=selectr, loc_feat=selectl, layers2show=sel_layers)
    def update_lsd_meet(rep_feat, loc_feat,layers2show):
        lsd_meet_mat = ssdm.anno_to_meet(
            track.lsd({'rep_ftype': rep_feat, 'loc_ftype': loc_feat}), 
            track.ts(), 
            num_layers=layers2show
        )
        return hv.Image(
            (track.ts(), track.ts(), lsd_meet_mat),
        ).opts(*options)
    
    @pn.depends(anno_id=selecta, anno_mode=sel_hier)
    def update_anno_meet(anno_id, anno_mode):
        anno_meet = ssdm.anno_to_meet(track.ref(anno_id=anno_id, mode=anno_mode), track.ts())
        return hv.Image(
            (track.ts(), track.ts(), anno_meet),
        ).opts(*options)
    
    @pn.depends(time=audio.param.time)
    def update_playhead(time):
        return hv.VLine(time).opts(color='white') *  hv.HLine(time).opts(color='white')

    @pn.depends(feature=selectr, 
                tau_width=slider_tau_width, 
                quant_method=qm_sel,
                quant_bins=qb_slider)
    def update_ssm(feature, tau_width, quant_method, quant_bins):
        ssm = track.ssm(feature=feature, width=tau_width, add_noise=True, n_steps=3, delay=1)
        quant_ssm = ssdm.quantize(ssm, quantize_method=quant_method, quant_bins=quant_bins)
        return hv.Image((track.ts(), track.ts(), quant_ssm)).opts(*options)
    

    # @pn.depends(anno_id=selecta, anno_mode=sel_hier)
    # def anno_meet_diag(anno_id, anno_mode):
    #     meet_mat = ssdm.anno_to_meet(track.ref(anno_id=anno_id, mode=anno_mode), track.ts())
    #     meet_diag = np.diag(meet_mat, k=1)
    #     return hv.Scatter((track.ts()[1:], meet_diag)).opts(width=300, height=100, shared_axes=False).redim.range(y=(0, max(meet_diag)))

   

    
    playhead = hv.DynamicMap(update_playhead)
    lsd_meet = hv.DynamicMap(update_lsd_meet)
    anno_meet = hv.DynamicMap(update_anno_meet)
    ssm_img = hv.DynamicMap(update_ssm)
    lsd_hm = hv.DynamicMap(lsd_score_heatmap)
    tau_hm = hv.DynamicMap(tau_heatmap)
    # meetmat_diag = hv.DynamicMap(anno_meet_diag)
    
    layout = pn.Column(
        pn.Row(audio),
        pn.Row(selectr, selectl, sel_layers, selecta, sel_hier),
        pn.Row(lsd_meet * playhead, anno_meet * playhead),
        pn.Row(qm_sel, qb_slider, slider_tau_width, lfs_dropdown),
        pn.Row(ssm_img * playhead, lsd_hm),
        pn.Row(tau_hm)
    )
    return layout


def template(track, state_dict_paths, default_rep='openl3', default_loc='mfcc', device='cuda:0'):
    # all the panel elements for selection etc.
    print(track.tid)

    audio = pn.pane.Audio(track.audio_path, sample_rate=22050, name=track.tid, throttle=250, width=300)
    selectr = pn.widgets.Select(
        name='lsd rep feature', options=ssdm.AVAL_FEAT_TYPES, width=90, value=default_rep
    )
    selectl = pn.widgets.Select(
        name='lsd loc feature', options=ssdm.AVAL_FEAT_TYPES, width=90, value=default_loc
    )
    sel_net = pn.widgets.Select(
        name='Model', options=state_dict_paths, width=500, value=state_dict_paths[0]
    )
    sel_layers = pn.widgets.EditableIntSlider(
        name='LSD Layer', 
        start=1, end=16, step=1, value=7, 
        fixed_start=1, fixed_end=16,
        width=160
    )
    selecta = pn.widgets.Select(
        name='anno id', options=[x for x in range(track.num_annos())], width=65
    )
    # hv options
    options = [
        opts.QuadMesh(
            cmap='inferno',
            colorbar=True,
            aspect='equal',
            frame_width=300,
            frame_height=300,
            line_alpha = 0,
            line_width=0,
        )
    ]
    hv.Dimension.type_formatters[np.float64]='%.3f'

    @pn.depends(rep_feat=selectr, loc_feat=selectl, layers2show=sel_layers)
    def update_lsd_meet(rep_feat, loc_feat, layers2show):
        lsd_meet_mat = ssdm.anno_to_meet(
            track.lsd({'rep_ftype': rep_feat, 'loc_ftype': loc_feat}, beat_sync=True), 
            track.ts(mode='beat'), 
            num_layers=layers2show
        )
        return hv.QuadMesh(
            (track.ts(mode='beat'), track.ts(mode='beat'), lsd_meet_mat),
        ).opts(*options).opts(title='lsd meet')
    

    @pn.depends(anno_id=selecta)
    def update_anno_meet(anno_id):
        anno_meet = ssdm.anno_to_meet(track.ref(mode='normal', anno_id=anno_id), track.ts(mode='beat'))
        return hv.QuadMesh(
            (track.ts(mode='beat'), track.ts(mode='beat'), anno_meet),
        ).opts(*options).opts(title='anno meet')


    # lsd l score grid for all feature combos
    @pn.depends(anno_id=selecta)
    def perf_score_heatmap(anno_id):
        lsd_score = track.new_lsd_score(anno_id=anno_id).sel(m_type='f').max('layer')
        lsd_score_grid = hv.HeatMap(lsd_score).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable', title='perf score')
        score_label = hv.Labels(lsd_score_grid).opts(text_color='black')

        # nlvl_pick = track.new_lsd_score(anno_id=anno_id).sel(m_type='f').argmax('layer').astype(int)
        # nlvl_grid = hv.HeatMap(nlvl_pick).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable', title='net nlvl pick')
        # nlvl_label = hv.Labels(nlvl_grid).opts(text_color='black')
        return lsd_score_grid * score_label # + nlvl_grid * nlvl_label

    
    @pn.depends(model_path=sel_net)
    def net_score_heatmap(model_path):
        net = ssdm.load_net(model_path)
        util_score, nlvl_score = track.scan_by(net, device)
        util_grid = hv.HeatMap(util_score).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable', title='net util score')
        score_label = hv.Labels(util_grid).opts(text_color='black')
        # nlvl_pick = nlvl_score.argmax('layer').astype(int)
        # nlvl_grid = hv.HeatMap(nlvl_pick).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable', title='net nlvl pick')
        # nlvl_label = hv.Labels(nlvl_grid).opts(text_color='black')
        return util_grid * score_label # + nlvl_grid * nlvl_label

    lsd_meet = hv.DynamicMap(update_lsd_meet)
    anno_meet = hv.DynamicMap(update_anno_meet)
    perf_hm = hv.DynamicMap(perf_score_heatmap)
    net_hm = hv.DynamicMap(net_score_heatmap)   

    
    layout = pn.Column(
        pn.Row(audio, selecta, selectr, selectl, sel_layers),
        pn.Row(sel_net),
        pn.Row(lsd_meet, anno_meet),
        pn.Row(perf_hm, net_hm),
    )
    return layout


def ds_xplore(ds, net=None, device='cuda:0', eval_result=None):
    tids = ds.tids
    hv.Dimension.type_formatters[np.float64]='%.3f'
    
    sel_tid = pn.widgets.Select(
        name='Track ID', options=tids, width=120
    )
    sel_layers = pn.widgets.EditableIntSlider(
        name='LSD Layer', 
        start=1, end=16, step=1, value=7, 
        fixed_start=1, fixed_end=16,
        width=160
    )

    if eval_result is not None:
        if net is not None:
            util_loss = torch.nn.BCELoss()
            nlvl_loss = torch.nn.MSELoss()
            eval_result = ssdm.scanner.net_eval_multi_loss(ds, net, util_loss, nlvl_loss, device=device, verbose=True)

    @pn.depends(tid=sel_tid, layer=sel_layers)
    def update_lsd_scores(tid, layer):
        track = ssdm.hmx.Track(tid=tid)
        track_score = track.new_lsd_score().sel(m_type='f')
        best_layer = track_score.idxmax('layer')

        lsd_score_grid = hv.HeatMap(track_score.sel(layer=layer)).opts(
            frame_width=300, frame_height=300, 
            cmap='coolwarm', toolbar='disable',
            title='L measure'
        )
        score_label = hv.Labels(lsd_score_grid).opts(text_color='black')

        lsd_best_layer_grid = hv.HeatMap(best_layer).opts(
            frame_width=300, frame_height=300, 
            cmap='coolwarm', toolbar='disable',
            title='Best layer'
        )
        best_layer_label = hv.Labels(lsd_best_layer_grid).opts(text_color='black')
        return lsd_score_grid * score_label + lsd_best_layer_grid * best_layer_label

    def get_samples(tid):
        track_lab = {k: str(ds.labels[k]) for k in ds.labels if k.split('_')[0] == tid}
        return track_lab

    def show_samp_evecs(tid):
        track_lab = get_samples(tid)
        imgs = []
        for k in track_lab:
            samp_idx = ds.samples.index(k)
            if ds[samp_idx]['label'] == 1:
                evecs = ds[samp_idx]['data'].cpu().numpy().squeeze()
                imgs.append(hv.Image(evecs).opts(
                    xaxis=None, yaxis=None, frame_width=200, frame_height=300,
                    cmap='RdBu', colorbar=False, toolbar='disable',
                    title = k + '\n' + str(ds.labels[k])
                ))

        if len(imgs) != 0:
            return hv.Layout(imgs).cols(7)

    lsd_score = hv.DynamicMap(update_lsd_scores)

    def get_lvl_est(tid):
        samps = get_samples(tid)
        if eval_result is not None:
            return {s: eval_result.nlvl[s] for s in samps if eval_result.label[s] == 1}


    layout = pn.Column(
        pn.Row(sel_tid, sel_layers),
        pn.Row(lsd_score),
        pn.Row(pn.bind(get_samples, sel_tid), pn.bind(get_lvl_est, sel_tid)),
        pn.Row(pn.bind(show_samp_evecs, sel_tid))
    )
    return layout

