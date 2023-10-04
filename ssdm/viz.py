import numpy as np
import pandas as pd
import librosa

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv 
from holoviews import opts
import panel as pn

from mir_eval import display

import ssdm
# import musicsections as ms


hv.extension("bokeh", logo=False)

def anno_meet_mats(track, mode='expand'):
    """
    mode can be one of {'normal', 'expand', 'refine', 'coarse'}
    """
    _, axs = plt.subplots(1, track.num_annos(), figsize=(5*track.num_annos() + 1, 4))
    if isinstance(axs, matplotlib.axes.Axes):
        axs = [axs]
    
    for anno_id in range(track.num_annos()):
        anno_jam = track.ref(mode=mode, anno_id=anno_id)
        quadmesh = librosa.display.specshow(ssdm.anno_to_meet(anno_jam, track.ts()), x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[anno_id])
        plt.colorbar(quadmesh)        
    return axs


def lsd_meet_mat(track, config=dict(), layer_to_show=None):
    lsd_config = ssdm.DEFAULT_LSD_CONFIG.copy()
    lsd_config.update(config)
    lsd_seg = track.lsd(lsd_config)
    lsd_meet_mat = ssdm.anno_to_meet(lsd_seg, track.ts(), num_layers=layer_to_show)
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(lsd_meet_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax

def rec_mat(track, **ssm_config):
    rec_mat = track.ssm(**ssm_config)
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(rec_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax

def path_sim(track, quant_bins=8, **path_sim_config):
    path_sim = track.path_sim(**path_sim_config)

    if quant_bins is not None:
        bins = [np.percentile(path_sim, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
            # print('tau_loc_bins:', bins)
        path_sim = np.digitize(path_sim, bins=bins, right=False)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(track.ts()[1:], path_sim)
    return fig, ax

def multi_seg(multi_seg):
    ## From ADOBE musicsection
    def plot_levels(inters, labels, figsize):
        """Plots the given hierarchy."""
        N = len(inters)
        fig, axs = plt.subplots(N, figsize=figsize) # add 1 for time axis
        for level in range(N):
            display.segments(np.asarray(inters[level]), labels[level], ax=axs[level])
            axs[level].set_yticks([0.5])
            axs[level].set_yticklabels([N - level])
            axs[level].set_xticks([])
        axs[0].xaxis.tick_top()
        fig.subplots_adjust(top=0.8)  # Otherwise savefig cuts the top
        
        return fig, axs

    def plot_segmentation(seg, figsize=(13, 3)):
        inters = []
        labels = []
        for level in seg[::-1]:
            inters.append(level[0])
            labels.append(level[1])

        fig, axs = plot_levels(inters, labels, figsize)
        fig.text(0.08, 0.47, 'Segmentation Levels', va='center', rotation='vertical')
        return fig, axs

    hier = ssdm.multiseg_to_hier(multi_seg)
    return plot_segmentation(hier)


def heatmap(da, ax=None, title=None, xlabel=None, ylabel=None, colorbar=True):   
    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))

    da = da.squeeze()
    
    im = ax.imshow(da, cmap='coolwarm')
    
    ycoord, xcoord = da.dims
    xticks = da.indexes[xcoord]
    yticks = da.indexes[ycoord]
    if xlabel is None:
        xlabel = xcoord
    if ylabel is None:
        ylabel = ycoord
    
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            ax.text(j, i, f"{da.values[i, j]:.3f}",
                    ha="center", va="center", color="k")

    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, shrink=0.8)
    return ax


def scatter_scores(
    x_data: pd.Series, 
    y_data: pd.Series,
    title: str = 'Scores per track',
    xlabel: str = 'x label',
    ylabel: str = 'y label',
    ax: any = None,
) -> matplotlib.axes._axes.Axes:
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        x=x_data, 
        y=y_data, 
        alpha=0.5, 
        s=3, 
    )
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.plot([0,1], [0,1], 'r:')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


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
        width=200
    )
    selecta = pn.widgets.Select(
        name='anno id', options=[x for x in range(track.num_annos())], width=65
    )
    sel_hier = pn.widgets.Select(
        name='Hierarchy Expansion', options=['expand', 'normal', 'refine', 'coarse'], width=120
    )
    slider_tau_width = pn.widgets.DiscreteSlider(
        name='tau-rep width', options=[16, 22, 27, 32, 54], value=27
    )
    tau_rep_output = pn.widgets.StaticText(name='tau_rep with expanded annotation', value='--')
    
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
    @pn.depends(anno_id=selecta)
    def lsd_score_heatmap(anno_id):
        lsd_score = track.lsd_score(l_frame_size=1, anno_id=anno_id).sel(l_type='lr')
        lsd_score_grid = hv.HeatMap(lsd_score).opts(frame_width=300, frame_height=300, cmap='coolwarm')
        score_label = hv.Labels(lsd_score_grid)
        return lsd_score_grid * score_label
    
    @pn.depends(anno_id=selecta, tau_width=slider_tau_width)
    def tau_heatmap(anno_id, tau_width):
        taus = track.tau(anno_id=anno_id, rec_width=tau_width)
        tau_grid = hv.HeatMap(taus).opts(frame_width=300, frame_height=100, cmap='coolwarm')
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

    @pn.depends(feature=selectr, tau_width=slider_tau_width, anno_id=selecta)
    def update_ssm(feature, tau_width, anno_id):
        ssm = track.ssm(feature=feature, width=tau_width, **ssdm.REP_FEAT_CONFIG[feature])
        tau_rep = track.tau(
            tau_sel_dict=dict(tau_type='rep', f_type=feature),
            anno_id=anno_id,
            rec_width=tau_width
        ).item()
        tau_rep_output.value = f'{tau_rep:.4f}'
        return hv.Image((track.ts(), track.ts(), ssm)).opts(*options)
    
    playhead = hv.DynamicMap(update_playhead)
    lsd_meet = hv.DynamicMap(update_lsd_meet)
    anno_meet = hv.DynamicMap(update_anno_meet)
    ssm_img = hv.DynamicMap(update_ssm)
    lsd_hm = hv.DynamicMap(lsd_score_heatmap)
    tau_hm = hv.DynamicMap(tau_heatmap)


    layout = pn.Column(
        audio, 
        pn.Row(selectr, selectl, sel_layers, selecta, sel_hier, ),
        pn.Row(lsd_meet * playhead, anno_meet * playhead),
        pn.Row(ssm_img * playhead, lsd_hm),
        pn.Row(slider_tau_width, tau_rep_output),
        tau_hm,
          
    )

    return layout