import numpy as np
import pandas as pd
import librosa
import matplotlib
import matplotlib.pyplot as plt
from mir_eval import display
import ssdm
import ssdm.formatting
import ssdm.scluster as sc
import holoviews as hv

import json
import torch
from tqdm import tqdm

import xarray as xr


def scatter_scores(
    x_data: pd.Series, 
    y_data: pd.Series,
    title: str = 'L scores per track',
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


def anno_meet_mats(track, mode='expand'):
    """
    mode can be one of {'normal', 'expand'}
    """
    fig, axs = plt.subplots(1, track.num_annos(), figsize=(5*track.num_annos() + 1, 4))
    if isinstance(axs, matplotlib.axes.Axes):
        axs = [axs]
    
    for anno_id in range(track.num_annos()):
        anno_jam = track.ref(mode=mode, anno_id=anno_id)
        quadmesh = librosa.display.specshow(ssdm.anno_to_meet(anno_jam, track.ts()), x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[anno_id])
        plt.colorbar(quadmesh)        
    return fig, axs


def lsd_meet_mat(track, config=dict(), beat_sync=False, layer_to_show=None):
    lsd_config = ssdm.DEFAULT_LSD_CONFIG.copy()
    if beat_sync:
        lsd_config.update(ssdm.BEAT_SYNC_CONFIG_PATCH)
    lsd_config.update(config)
    lsd_seg = track.lsd(lsd_config, beat_sync=beat_sync)
    lsd_meet_mat = ssdm.anno_to_meet(lsd_seg, track.ts(), num_layers=layer_to_show)
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(lsd_meet_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax

def rec_mat(track, blocky=False, rec_diag=None, ax=None, **ssm_config):
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
    config=ssdm.DEFAULT_LSD_CONFIG
    if blocky:
        # REFACTOR THE FOLLOWING CODE TODO
        combined_graph = sc.combine_ssms(rec_mat, rec_diag, rec_smooth=config['rec_smooth'])
        _, evecs = sc.embed_ssms(combined_graph, evec_smooth=config['evec_smooth'])
        first_evecs = evecs[:, :10]
        rec_mat = ssdm.quantize(np.matmul(first_evecs, first_evecs.T), quant_bins=7)

    if ax == None:
        fig, ax = plt.subplots(figsize=(5, 4))
        quadmesh = librosa.display.specshow(rec_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
        fig.colorbar(quadmesh, ax=ax)
        return fig, ax
    else:
        return librosa.display.specshow(rec_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)

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
def multi_seg(multi_seg, hier_depth=None, figsize=(10, 2.4), **eval_kwargs):
    ## From ADOBE musicsection
    def plot_levels(inters, labels, figsize):
        """Plots the given hierarchy."""
        N = len(inters)
        fig, axs = plt.subplots(N, figsize=figsize) 
        if N == 1:
            axs = [axs]
        for level in range(N):
            display.segments(np.asarray(inters[level]), labels[level], ax=axs[level], edgecolor='white', **eval_kwargs)
            axs[level].set_yticks([0.5])
            axs[level].set_yticklabels([level + 1])
            axs[level].set_xticks([])
            axs[level].yaxis.tick_right()
            axs[level].yaxis.set_label_position("right")

        axs[0].xaxis.tick_top()
        fig.subplots_adjust(top=0.8)  # Otherwise savefig cuts the top
        
        return fig, axs

    def plot_segmentation(seg, figsize=figsize):
        inters = []
        labels = []
        for level in seg:
            inters.append(level[0])
            labels.append(level[1])

        fig, axs = plot_levels(inters, labels, figsize)
        # fig.text(0.08, 0.47, 'Segmentation Levels', va='center', rotation='vertical')
        return fig, axs

    hier = ssdm.multi2hier(multi_seg)[:hier_depth]
    return plot_segmentation(hier)


def heatmap(da, ax=None, title=None, xlabel=None, ylabel=None, colorbar=True, figsize=(5,5), no_deci=False):   
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    da = da.squeeze()
    if len(da.shape) == 1:
        da = da.expand_dims(dim='_', axis=0)
        da = da.assign_coords(_=[''])
    
    im = ax.imshow(da.values.astype(float), cmap='coolwarm')
    
    try:
        # try to get axis label and ticks from dataset coords
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
    except:
        pass
    
    
    for i in range(da.shape[0]):
        for j in range(da.shape[1]):
            if no_deci:
                ax.text(j, i, f"{da.values[i, j]}",
                        ha="center", va="center", color="k")
            else:
                ax.text(j, i, f"{da.values[i, j]:.3f}",
                        ha="center", va="center", color="k")

    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, shrink=0.8)
    return fig, ax


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


def train_curve(json_path, weighted_loss=False):
    with open(json_path) as f:
        train_curves = json.load(f)

    if weighted_loss:
        train_loss_list = [pair[0] + pair[1] * 0.1 for pair in train_curves['train_loss']]
        val_loss_list = [pair[0] + pair[1] * 0.1 for pair in train_curves['val_loss']]
    else:
        train_loss_list = train_curves['train_loss']
        val_loss_list = train_curves['val_loss']

    train_loss = hv.Curve(train_loss_list, label='Train loss')
    val_loss = hv.Curve(val_loss_list, label='Val loss')


    best_epoch = np.asarray(val_loss_list).argmin()
    best_val_loss = np.asarray(val_loss_list).min()
    best_epoch_line = hv.VLine(x=best_epoch)
    best_val_text = hv.Text(
        1, 1e-4, f"Best epoch {best_epoch}: val loss: {best_val_loss:.3f}"
    ).opts(text_align='left', text_baseline='bottom')


    return train_loss * val_loss * best_epoch_line * best_val_text


def train_curve_multi_loss(json_path):
    with open(json_path) as f:
        train_curves = json.load(f)

    util_train_loss = [pair[0] for pair in train_curves['train_loss']]
    nlvl_train_loss = [pair[1] for pair in train_curves['train_loss']]
    util_val_loss = [triplet[0] for triplet in train_curves['val_loss']]
    nlvl_val_loss = [triplet[-1] for triplet in train_curves['val_loss']]

    train_u_loss = hv.Curve(util_train_loss, label='train util')
    val_u_loss = hv.Curve(util_val_loss, label='val util')

    train_lvl_loss = hv.Curve(nlvl_train_loss, label='train n_lvl')
    val_lvl_loss = hv.Curve(nlvl_val_loss, label='val n_lvl')

    best_u_epoch = np.asarray(util_val_loss).argmin()
    best_val_u_loss = np.asarray(util_val_loss).min()
    best_u_epoch_line = hv.VLine(x=best_u_epoch)
    

    best_nlvl_epoch = np.asarray(nlvl_val_loss).argmin()
    best_val_nlvl_loss = np.asarray(nlvl_val_loss).min()
    best_nlvl_epoch_line = hv.VLine(x=best_nlvl_epoch)
    boa_nlvl_baseline_val_l = hv.HLine(y=0.012391).opts(color='orange', line_width=1)
    boa_nlvl_baseline_train_l = hv.HLine(y=0.010598).opts(color='navy', line_width=1)
    boa_nlvl_baseline_val_pfc = hv.HLine(y=0.032466).opts(color='orange')
    boa_nlvl_baseline_train_pfc = hv.HLine(y=0.031441).opts(color='navy')
    baselines = boa_nlvl_baseline_val_l * boa_nlvl_baseline_train_l * boa_nlvl_baseline_val_pfc * boa_nlvl_baseline_train_pfc

    best_val_text = hv.Text(
        1, 0.3, 
        f" Best util epoch {best_u_epoch}, loss: {best_val_u_loss:.3f}"
    ).opts(text_align='left', text_baseline='bottom')
    best_nlvl_text = hv.Text(
        1, 2e-2, 
        f" Best nlvl epoch {best_nlvl_epoch}, loss: {best_val_nlvl_loss:.3f}"
    ).opts(text_align='left', text_baseline='bottom')

    util_overlay = train_u_loss * val_u_loss * best_u_epoch_line * best_val_text
    nlvl_overlay = train_lvl_loss * val_lvl_loss * best_nlvl_epoch_line * best_nlvl_text * baselines

    return util_overlay, nlvl_overlay


def train_curve_contrastive(json_path):
    with open(json_path) as f:
        train_curves = json.load(f)

    util_train_loss = [triplet[0] for triplet in train_curves['train_loss']]
    nlvl_train_loss = [triplet[1] for triplet in train_curves['train_loss']]
    entropy_train_penalty = [-triplet[2] for triplet in train_curves['train_loss']]

    train_u_loss = hv.Curve(util_train_loss, label='train util')
    train_lvl_loss = hv.Curve(nlvl_train_loss, label='train n_lvl')
    train_nlvl_entropy = hv.Curve(entropy_train_penalty, label='val util')

    return train_u_loss + train_lvl_loss + train_nlvl_entropy


def nlvl_est(ds, net, device='cuda:0', pos_only=True, plot=True, **img_kwargs):
    first_s = ds[0]
    nlvl_output_size = first_s['layer_score'].shape[-1]
    print('nlvl_output_size: ', nlvl_output_size)
    xr_coords = dict(sid=ds.samples, pred=['pred', 'target'], layer=list(range(nlvl_output_size)))
    nlvl_outputs = xr.DataArray(None, coords=xr_coords, dims=xr_coords.keys())
    nlvl_loss_fn = ssdm.scn.NLvlLoss()
    loss = dict()
    target_best_layer = dict()
    pred_best_layer = dict()
    
    net.eval()
    net.to(device)
    counter = 0
    with torch.no_grad():
        for s in tqdm(ds):
            if s['label'] == 0 and pos_only:
                nlvl_outputs = nlvl_outputs.drop_sel(sid=s['info'])
            else:
                util, nlvl = net(s['data'].to(device))
                nlvl_outputs.loc[s['info'], 'pred'] = nlvl.detach().cpu().numpy().squeeze()
                nlvl_outputs.loc[s['info'], 'target'] = s['layer_score'].detach().cpu().numpy().squeeze()
                loss[s['info']] = nlvl_loss_fn(nlvl, s['layer_score'].to(device)).detach().cpu().numpy().squeeze()
                pred_best_layer[s['info']] = nlvl_outputs.loc[s['info'], 'pred'].argmax().item()
                target_best_layer[s['info']] = nlvl_outputs.loc[s['info'], 'target'].argmax().item()
                counter += 1
            
            # if counter >= 30:
            #     break
    
    sid_by_best_layer = sorted(target_best_layer, key=target_best_layer.get)
    sorted_nlvl_outputs = nlvl_outputs.loc[sid_by_best_layer]


    losses = np.array(list(loss.values()))
    print('mean nlvl loss:', losses.mean())
    if plot:
        x_axis = list(range(len(loss)))
        y_axis = list(range(nlvl_output_size))
        best_layer_line = hv.Curve(sorted_nlvl_outputs.sel(pred='target').argmax('layer').values).opts(interpolation='steps-mid', color='white')
        pred_img = hv.Image((x_axis, y_axis, sorted_nlvl_outputs.sel(pred='pred').values.T)).opts(colorbar=True, cmap='coolwarm', width=500, height=150, **img_kwargs)
        target_img = hv.Image((x_axis, y_axis, sorted_nlvl_outputs.sel(pred='target').values.T)).opts(colorbar=True, cmap='coolwarm', width=500, height=150, **img_kwargs)
        layout = [pred_img*best_layer_line, target_img*best_layer_line]
        hv.output(hv.Layout(layout).cols(2))
    
    return sorted_nlvl_outputs
