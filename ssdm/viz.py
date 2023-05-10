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


hv.extension("bokeh", logo=False)

# get it to consume ax object for easier ploting.
def sdm(track, feat='mfcc', ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    # Get track with tid and get sdm
    sdm = track.sdm(feature=feat, distance='cosine')
    quadmesh = librosa.display.specshow(sdm, y_axis='time', x_axis='time', ax=ax, hop_length=4096, sr=22050)
    ax.set_title(f"track: {track.tid} | feature: {feat}")
    plt.colorbar(quadmesh)
    return quadmesh
    

def all_sdms(track):
    fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(9,10))
    for i, feat in enumerate(ssdm.AVAL_FEAT_TYPES):
        quadmesh = sdm(track, feat=feat, ax=axs[i // 2][i % 2]);
    fig.tight_layout()
    return fig, axs


def anno_meet_mats(track, mode='normal'):
    """
    mode can be one of {'normal', 'expand', 'refine', 'coarse'}
    """
    _, axs = plt.subplots(1, track.num_annos(), figsize=(5*track.num_annos() + 1, 4))
    if isinstance(axs, matplotlib.axes.Axes):
        axs = [axs]
    
    for anno_id in range(track.num_annos()):
        anno_jam = track.segmentation_annotation(mode=mode, anno_id=anno_id)
        quadmesh = librosa.display.specshow(ssdm.segmentation_to_meet(anno_jam, track.ts()), x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[anno_id])
        plt.colorbar(quadmesh)        
    return axs


def lsd_meet_mat(track, rep_feature='openl3', loc_feature='mfcc', layer=10):
    lsd_config = {'rec_width': 13, 
                  'rec_smooth': 7, 
                  'evec_smooth': 13,
                  'rep_ftype': rep_feature, 
                  'loc_ftype': loc_feature,
                  'rep_metric': 'cosine',
                  'hier': True,
                  'num_layers': 10}
    
    lsd_seg = track.segmentation_lsd(lsd_config)
    lsd_meet_mat = ssdm.segmentation_to_meet(lsd_seg, track.ts(), num_layers=layer)
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(lsd_meet_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax


def heatmap(df, ax=None, title=None, xlabel='Local Feature', ylabel='Repetition Feature'):   
    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))


    im = ax.imshow(df, cmap='coolwarm')
    ax.set_xticks(np.arange(6), labels=df.columns)
    ax.set_yticks(np.arange(6), labels=df.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{df.to_numpy()[i, j]:.3f}",
                    ha="center", va="center", color="k")

    ax.set_title(title)
    plt.colorbar(im, shrink=0.8)
    return ax


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


def update_cross_dmap(time):
    vline = hv.VLine(time).opts(color='cyan', line_width=1.5)
    hline = hv.HLine(time).opts(color='cyan', line_width=1.5)
    return vline * hline


def explore_track(tid, rep_feature, loc_feature, lsd_layer=10):
    # init track object
    track = ssdm.Track(tid)
    if lsd_layer > 10:
        lsd_layer = 10
    elif lsd_layer <2:
        lsd_layer = 2

    # sdms
    rep_sdm = track.sdm(feat=rep_feature)
    loc_sdm = track.sdm(feat=loc_feature)

    # annotations meet mats
    meet = track.meet_mats(track.common_ts())[0]
    # meet_hv = track.meet_mats(track.common_ts())[1]

    # LSD results:
    hier_anno = track.lsd_anno(rep_feature=rep_feature, loc_feature=loc_feature)
    lsd = ssdm._meet_mat(hier_anno[:lsd_layer], ts=track.common_ts())
    
    # audio
    audio = pn.pane.Audio(track.audio_path, name=tid, throttle=250)

    # Imgs
    loc_sdm_hv = hv.Image((track.common_ts(), track.common_ts(), loc_sdm)).opts(data_aspect=1, cmap='inferno', colorbar=True, title=f'Loc Feature SDM: {loc_feature}')
    rep_sdm_hv = hv.Image((track.common_ts(), track.common_ts(), rep_sdm)).opts(data_aspect=1, cmap='inferno', colorbar=True, title=f'Rep Feature SDM: {rep_feature}')

    meet_hvs = [
        hv.Image(
        (track.common_ts(), track.common_ts(), meet)).opts(
            data_aspect=1, cmap='inferno', colorbar=True, title=f'Anno({i}) Meet Mat'
        ) for i, meet in enumerate(track.meet_mats(track.common_ts()))
    ]

    # meet_hv = hv.Image((track.common_ts(), track.common_ts(), meet)).opts(data_aspect=1, cmap='inferno', colorbar=True, title='Anno Meet Mat')
    lsd_hv = hv.Image((track.common_ts(), track.common_ts(), lsd)).opts(data_aspect=1, cmap='inferno', colorbar=True, title='Spectral Clustering Meet Mat')

    

    playhead_dmap = hv.DynamicMap(update_cross_dmap, streams=[audio.param.time])

    return pn.Column(audio,
                     pn.Row(rep_sdm_hv * playhead_dmap, loc_sdm_hv * playhead_dmap),
                     pn.Row(*[meet_hv * playhead_dmap for meet_hv in meet_hvs]),
                     (lsd_hv * playhead_dmap),
                    )


def explore_sdm_tau(tid='384'):
    # init track and get audio
    track = ssdm.Track(tid)
    audio = pn.pane.Audio(track.audio_path, name=tid, throttle=250)
    tau_df = pd.read_pickle('./tau_df.pkl')

    hv_collection = {}

    dmap = hv.DynamicMap(update_cross_dmap, streams=[audio.param.time])

    for feat in ssdm.AVAL_FEAT_TYPES:
        # tau_r and tau_l for each sdm
        tau_r = tau_df.loc[tid][feat, 'rep']
        tau_l = tau_df.loc[tid][feat, 'loc']
        tau_text = f'tau_r: {tau_r:.3f}    tau_l: {tau_l:.3f}'

        # image:
        hv_collection[feat] = hv.Image(
            (track.common_ts(), track.common_ts(), track.sdm(feat=feat)), 
            kdims=['x','y']
        ).opts(
            title=f'{feat} SDM \n{tau_text}',
            data_aspect=1,
            frame_width=250,
            cmap='inferno',
            colorbar=True,
        ) * dmap 

    # return layout
    return pn.Column(
        audio,
        pn.Row(hv_collection['tempogram'], hv_collection['chroma']),
        pn.Row(hv_collection['openl3'], hv_collection['yamnet']),
        pn.Row(hv_collection['mfcc'], hv_collection['crema']),
    )
