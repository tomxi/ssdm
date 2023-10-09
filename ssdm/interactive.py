import numpy as np

import panel as pn
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

import ssdm

def follow_along(track):
    # all the panel elements for selection etc.
    audio = pn.pane.Audio(track.audio_path, sample_rate=22050, name=track.tid, throttle=250, width=300)
    selectr = pn.widgets.Select(
        name='lsd rep feature', options=ssdm.AVAL_FEAT_TYPES, width=90
    )
    selectl = pn.widgets.Select(
        name='lsd loc feature', options=ssdm.AVAL_FEAT_TYPES, width=90
    )
    sel_layers = pn.widgets.EditableIntSlider(
        name='LSD layers to show', 
        start=1, end=10, step=1, value=10, 
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
        name='tau-rep width', options=[16, 22, 27, 32, 54], value=27, width=170
    )
    lfs_dropdown = pn.widgets.Select(
        name='l frame size', options=[0.1, 1], width=100)
    qm_sel = pn.widgets.Select(
        name='quantize method', options=[None, 'percentile'], value='percentile', width=100
    )
    qb_slider = pn.widgets.EditableIntSlider(
        name='Quantize Bins', 
        start=1, end=12, step=1, value=7, 
        fixed_start=1, fixed_end=10,
        width=170
    )
    
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
        lsd_score = track.lsd_score(l_frame_size=l_frame_size, anno_id=anno_id).sel(l_type='lr')
        lsd_score_grid = hv.HeatMap(lsd_score).opts(frame_width=300, frame_height=300, cmap='coolwarm', toolbar='disable')
        score_label = hv.Labels(lsd_score_grid)
        return lsd_score_grid * score_label
    
    @pn.depends(anno_id=selecta, anno_mode=sel_hier, tau_width=slider_tau_width, quantize=qm_sel, quant_bins=qb_slider)
    def tau_heatmap(anno_id, anno_mode, tau_width, quantize, quant_bins):
        taus = track.tau(anno_id=anno_id, rec_width=tau_width, anno_mode=anno_mode, quantize=quantize, quant_bins=quant_bins)
        tau_grid = hv.HeatMap(taus).opts(frame_width=300, frame_height=100, cmap='coolwarm', toolbar='disable')
        score_label = hv.Labels(tau_grid)
        return tau_grid * score_label

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
        ssm = track.ssm(feature=feature, width=tau_width, **ssdm.REP_FEAT_CONFIG[feature])
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
        audio, 
        pn.Row(selectr, selectl, sel_layers, selecta, sel_hier),
        pn.Row(lsd_meet * playhead, anno_meet * playhead),
        pn.Row(qm_sel, qb_slider, slider_tau_width, lfs_dropdown),
        pn.Row(ssm_img * playhead, lsd_hm),
        pn.Row(tau_hm)
    )
    return layout

