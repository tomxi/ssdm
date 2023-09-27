import numpy as np
import pandas as pd
import librosa

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv 
from holoviews import opts
import panel as pn

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


def lsd_meet_mat(track, config=ssdm.DEFAULT_LSD_CONFIG, layer_to_show=7):
    lsd_seg = track.lsd(config)
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


def multi_seg(multi_seg):
    hier = ssdm.multiseg_to_hier(multi_seg)
    raise NotImplementedError
    # return ms.plot_segmentation(hier)


def heatmap(da, ax=None, title=None, xlabel=None, ylabel=None):   
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




