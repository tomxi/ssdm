import numpy as np
import pandas as pd
import librosa
import matplotlib
import matplotlib.pyplot as plt
from mir_eval import display
import ssdm
import ssdm.scluster as sc
import holoviews as hv
import json

import xarray as xr


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

def rec_mat(track, blocky=False, rec_diag=None, **ssm_config):
    """
    blocky: bool = False, # if true, use low rank appoximation,
    rec_diag: np.ndarray = None, # used when blocky is True, to combine with SSM to form the Laplacian
    feature: str = 'mfcc',
    distance: str = 'cosine',
    width = 5, # width param for librosa.segment.rec_mat <= 1
    bw: str = 'med_k_scalar', # one of {'med_k_scalar', 'mean_k', 'gmean_k', 'mean_k_avg', 'gmean_k_avg', 'mean_k_avg_and_pair'}
    full: bool = False, # whether the rec mat is sparse or full
    add_noise: bool = False, # padding representation with a little noise to avoid divide by 0 somewhere...
    n_steps: int = 1, # Param for time delay embedding of representation
    delay: int = 1, # Param for time delay embedding of representation
    recompute: bool = False,
    """
    rec_mat = track.ssm(**ssm_config)
    if blocky:
        # REFACTOR THE FOLLOWING CODE TODO
        combined_graph = sc.combine_ssms(rec_mat, rec_diag)
        _, evecs = sc.embed_ssms(combined_graph, evec_smooth=13)
        first_evecs = evecs[:, :10]
        rec_mat = ssdm.quantize(np.matmul(first_evecs, first_evecs.T), quant_bins=7)
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(rec_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax

def path_sim(track, quant_bins=None, **path_sim_config):
    path_sim = track.path_sim(**path_sim_config)

    if quant_bins is not None:
        bins = [np.percentile(path_sim, bin * (100.0/quant_bins)) for bin in range(quant_bins + 1)]
            # print('tau_loc_bins:', bins)
        path_sim = np.digitize(path_sim, bins=bins, right=False)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(track.ts()[1:], path_sim)
    return fig, ax

def path_ref(track, **path_ref_args):
    """
    mode: str = 'expand', # {'normal', 'expand', 'refine', 'coarse'},
    anno_id: int = 0,
    binarize: bool = True,
    """
    path_ref = track.path_ref(**path_ref_args)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(track.ts()[1:], path_ref)
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

    hier = ssdm.multi2hier(multi_seg)
    return plot_segmentation(hier)


def heatmap(da, ax=None, title=None, xlabel=None, ylabel=None, colorbar=True):   
    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))

    da = da.squeeze()
    if len(da.shape) == 1:
        da = da.expand_dims(dim='_', axis=0)
        da = da.assign_coords(_=[''])
    
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


## Inspecting model weight tensors
def view_channels_(cube, channel=0, **hv_img_args):
    if type(cube) is not np.ndarray:
        cube = cube.cpu().numpy()

    img_kwargs = dict(
        frame_width=300, aspect='equal', cmap='inferno', active_tools=['box_zoom'],
        xaxis=None, yaxis=None,
    )
    img_kwargs.update(hv_img_args)

    cube = cube.squeeze()
    ticks = list(range(cube.shape[-1]))
    if len(cube.shape) == 2:
        print('only 1 layer, channel is ignored')
        square = cube
    else:
        square = cube[channel].squeeze()
    return hv.Image((ticks, ticks, square)).opts(**img_kwargs) 


def cube(cube, **kwargs):
    cube = cube.squeeze()
    channel_imgs = []
    if len(cube.shape) == 2:
        channels = 1
    else:
        channels = cube.shape[0]
    for channel in range(channels):
        channel_imgs.append(view_channels_(cube, channel=channel, title=f'channel: {channel}', **kwargs))

    return hv.Layout(channel_imgs)


def train_curve(json_path):
    with open(json_path) as f:
        train_curves = json.load(f)

    train_loss = hv.Curve(train_curves['train_loss'], label='Train CE loss')
    val_loss = hv.Curve(train_curves['val_loss'], label='Val CE loss')

    best_epoch = np.asarray(train_curves['val_loss']).argmin()
    best_val_loss = np.asarray(train_curves['val_loss']).min()
    best_epoch_line = hv.VLine(x=best_epoch)
    best_val_text = hv.Text(
        1, 0, f"Best epoch {best_epoch}: val loss: {best_val_loss:.3f}"
    ).opts(text_align='left', text_baseline='bottom')


    return train_loss * val_loss * best_epoch_line * best_val_text


# Show dataset performance heatmap... a bit too general
def dataset_performance(dataset, 
                        tau_hat_rep_path, 
                        tau_hat_loc_path):
    # L SCORES # WITH TAU-HAT PICKING
    score_per_track = []
    for tid in dataset.tids:
        track = dataset.track_obj(tid=tid)
        score_per_track.append(track.lsd_score())
    
    score_da = xr.concat(score_per_track, pd.Index(dataset.tids, name='tid')).rename().sortby('tid')
    

    # add tau-hat pick performance:
    tau_hat_rep = xr.open_dataarray(tau_hat_rep_path)
    tau_hat_loc = xr.open_dataarray(tau_hat_loc_path)

    rep_pick = tau_hat_rep.idxmax(dim='f_type').sortby('tid')
    loc_pick = tau_hat_loc.idxmax(dim='f_type').sortby('tid')

    tau_hat_rep_score = score_da.sel(rep_ftype=rep_pick).drop_vars('rep_ftype').expand_dims(rep_ftype=['tau_hat'])
    tau_hat_loc_score = score_da.sel(loc_ftype=loc_pick).drop_vars('loc_ftype').expand_dims(loc_ftype=['tau_hat'])
    tau_hat_both_score = score_da.sel(rep_ftype=rep_pick, loc_ftype=loc_pick).drop_vars(['loc_ftype', 'rep_ftype']).expand_dims(loc_ftype=['tau_hat'], rep_ftype=['tau_hat'])

    score_with_tau_rep = xr.concat([score_da, tau_hat_rep_score], dim='rep_ftype')
    full_tau_loc_score = xr.concat([tau_hat_loc_score, tau_hat_both_score], dim='rep_ftype')
    full_score = xr.concat([score_with_tau_rep, full_tau_loc_score], dim='loc_ftype')
    
    if 'anno_id' not in full_score.indexes:
        average_performance = full_score.mean(dim=['tid'])
    else:
        average_performance = full_score.mean(dim=['anno_id', 'tid'])

    o = []
    for l_type in ['lr']:
        o.append(heatmap(average_performance.sel(l_type=l_type), title=f'{dataset} {l_type} average score'))
    
    # F SCORES

    return o

