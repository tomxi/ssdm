import numpy as np
import pandas as pd
import librosa
import matplotlib
import matplotlib.pyplot as plt
from mir_eval import display
import ssdm


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

# Visualize a multi level segmentation jams.Annotation
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
