import numpy as np
import pandas as pd
import librosa
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mir_eval import display
import ssdm
import ssdm.formatting
import ssdm.scluster as sc
import holoviews as hv

import itertools
from cycler import cycler
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


def assign_label_styles(labels, **kwargs):
    labels = labels.copy()
    unique_labels = []
    while labels:
        l = labels.pop()
        if isinstance(l, list):
            labels.extend(l)  
        elif l not in unique_labels:
            unique_labels.append(l)
    unique_labels.sort()

    if len(unique_labels) <= 80:
        hatch_cycler = cycler(hatch=['',  '..', 'xx', 'O.',  '*',  '\\O', 'oo', 'xxO'])
        fc_cycler = cycler(color=plt.get_cmap('tab10').colors)
        p_cycler = hatch_cycler * fc_cycler
    else:
        hatch_cycler = cycler(hatch=['', 'oo', 'xx', 'O.', '*', '..', '\\', '\\O',
                                     '--', 'oo--', 'xx--', 'O.--', '*--', '\\--', '\\O--'
                                    ])
        fc_cycler = cycler(color=plt.get_cmap('tab20').colors)
        p_cycler = itertools.cycle(hatch_cycler * fc_cycler)
    
    seg_map=dict()
    for lab, properties in zip(unique_labels, p_cycler):
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }
        # Swap color -> facecolor here so we preserve edgecolor on rects
        if 'color' in style:
            style.setdefault("facecolor", style["color"])
            style.pop("color", None)
        seg_map[lab] = dict(linewidth=1, edgecolor='white')
        seg_map[lab].update(style)
        seg_map[lab].update(kwargs)
        seg_map[lab]["label"] = lab
    return seg_map


def segments(
    intervals,
    labels,
    ax=None,
    text=False,
    style_map=None,
):
    if ax is None:
        fig = plt.gcf(); ax = fig.gca()
        ax.set_yticks([])
    ax.set_xlim(intervals[0][0], intervals[-1][-1])

    if style_map is None:
        style_map = assign_label_styles(labels, edgecolor='white')
    transform = ax.get_xaxis_transform()
    
    for ival, lab in zip(intervals, labels):
        rect = ax.axvspan(ival[0], ival[1], ymin=0, ymax=1, **style_map[lab])
        if text:
            ann = ax.annotate(
                lab,
                xy=(ival[0], 1), xycoords=transform, xytext=(8, -10),
                textcoords="offset points", va='top', clip_on=True, 
                bbox=dict(boxstyle="round", facecolor="white")
            )
            ann.set_clip_path(rect)
    return ax


def multi_seg(ms_anno, figsize=(8, 4), reindex=True, legend_ncol=6, text=False, y_label=True, x_label=True):
    """Plots the given multi_seg annotation.
    """
    hier = ssdm.formatting.multi2hier(ms_anno)
    if reindex:
        hier = sc.reindex(hier)
    N = len(hier)
    fig, axs = plt.subplots(N, figsize=figsize) 
    if N == 1:
        axs = [axs]

    _, lbls = ssdm.formatting.hier2mireval(hier)
    style_map = assign_label_styles(lbls)
    legend_handles = [mpatches.Patch(**style) for style in style_map.values()]

    for level, (itvl, lbl) in enumerate(hier):
        ax = segments(
            itvl, lbl, ax=axs[level], 
            style_map=style_map, text=text
        )
        if y_label:
            ax.set_yticks([0.5])
            ax.set_yticklabels([level + 1])
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    
    if x_label:
        # Show time axis on the last layer
        axs[-1].xaxis.set_major_locator(ticker.AutoLocator())
        axs[-1].xaxis.set_major_formatter(librosa.display.TimeFormatter())
        axs[-1].set_xlabel('Time')

    if legend_ncol:
        fig.legend(handles=legend_handles, loc='lower center', ncol=legend_ncol, bbox_to_anchor=(0.5, -0.06 * (len(legend_handles)//legend_ncol + 2.2)))
    if y_label:
        fig.text(0.94, 0.55, 'Segmentation Levels', va='center', rotation='vertical')
    # fig.tight_layout(rect=[0,0,0.95,1])
    return fig, axs


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
