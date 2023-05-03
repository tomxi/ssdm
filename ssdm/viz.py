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
def show_sdm(tid='384', feat='mfcc', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Get track with tid and get sdm
    track = ssdm.Track(tid=tid)
    sdm = track.sdm(feature=feat, distance='cosine')
    quadmesh = librosa.display.specshow(sdm, y_axis='time', x_axis='time', ax=ax, hop_length=4096, sr=22050)
    
    ax.set_title(f"track: {tid} | feature: {feat}")
    return quadmesh
    

def show_all_sdms(tid='384'):
    fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(9,10))
    for i, feat in enumerate(ssdm.AVAL_FEAT_TYPES):
        quadmesh = show_sdm(tid=tid, feat=feat, ax=axs[i // 2][i % 2]);
        fig.colorbar(quadmesh, ax=axs[i // 2][i % 2]);
    fig.tight_layout()
    return fig, axs


def show_anno_meet_mats(tid='384', mode='normal'):
    """
    mode can be one of {'normal', 'expand', 'refine', 'coarse'}
    """
    track = ssdm.Track(tid=tid)
    fig, axs = plt.subplots(1, track.num_annos(), figsize=(5*track.num_annos() + 1, 4))
    if isinstance(axs, matplotlib.axes.Axes):
        axs = [axs]
    
    for anno_id in range(track.num_annos()):
        anno_jam = track.segmentation_annotation(mode=mode, anno_id=anno_id)
        quadmesh = librosa.display.specshow(ssdm.segmentation_to_meet(anno_jam, track.ts()), x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[anno_id])
        fig.colorbar(quadmesh, ax=axs[anno_id])        
    return fig, axs


def show_lsd_meet_mat(tid='384', rep_feature='openl3', loc_feature='mfcc', layer=10):
    lsd_config = {'rec_width': 13, 
                  'rec_smooth': 7, 
                  'evec_smooth': 13,
                  'rep_ftype': rep_feature, 
                  'loc_ftype': loc_feature,
                  'rep_metric': 'cosine',
                  'hier': True,
                  'num_layers': 10}
    
    track = ssdm.Track(tid=tid)
    lsd_seg = track.segmentation_lsd(lsd_config)
    lsd_meet_mat = ssdm.segmentation_to_meet(lsd_seg, track.ts())
    fig, ax = plt.subplots(figsize=(5, 4))
    quadmesh = librosa.display.specshow(lsd_meet_mat, x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=ax)
    fig.colorbar(quadmesh, ax=ax)      
    return fig, ax


# def scatter_scores(compare_scores_df, x_heuristic, y_heuristic, ax=None):
#     if ax is None:
#         _, ax = plt.subplots()
   
#     compare_scores_df.plot.scatter(
#         x=x_heuristic, 
#         y=y_heuristic, 
#         alpha=0.5, 
#         s=3, 
#         xlim=(0,1), 
#         ylim=(0,1), 
#         ax=ax
#     )
#     ax.plot([0,1], [0,1], 'r:')
#     ax.set_aspect('equal', 'box')
#     ax.set_title('L-Recall scores')
#     return ax


# def score_delta_kde(compare_scores_df, x_heuristic='Best Avg Pair', y_heuristic='Rep Pick', ax=None):
#     score_delta = compare_scores_df[x_heuristic] - compare_scores_df[y_heuristic]

#     if ax is None:
#         _, ax = plt.subplots()
        
#     ax = sns.kdeplot(score_delta, ax=ax, common_grid=True)
#     ymax = ax.viewLim.ymax
#     ax.vlines([score_delta.mean()], ymin=0, ymax=ax.viewLim.ymax, color='red', linestyle=':', label=f'Mean gap: {score_delta.mean()*100:.2f}%')
#     ax.set_ylim((0, ymax))
#     ax.set_title(f'Distribution of performance gap between \n \
#     {x_heuristic} and {y_heuristic}')
#     ax.legend()
#     ax.grid()
#     return ax


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


def scatter_all_scores():
    compare_scores_df = ssdm.score_comparison_df()
    
    axd = plt.figure(layout="tight", figsize=(15, 15)).subplot_mosaic(
        """
        aci
        bdj
        egk
        fhl
        """,
        height_ratios=[1, 4, 1, 4],
    )

    score_delta_kde(compare_scores_df, 'Rep Pick', 'Best Avg Pair', ax=axd['a'])
    scatter_scores(compare_scores_df,  'Rep Pick', 'Best Avg Pair',ax=axd['b'])

    score_delta_kde(compare_scores_df, 'Loc Pick', 'Best Avg Pair', ax=axd['c'])
    scatter_scores(compare_scores_df, 'Loc Pick', 'Best Avg Pair',  ax=axd['d'])

    score_delta_kde(compare_scores_df, 'Both Pick', 'Best Avg Pair', ax=axd['i'])
    scatter_scores(compare_scores_df, 'Both Pick', 'Best Avg Pair', ax=axd['j'])

    score_delta_kde(compare_scores_df, 'Rep Pick', 'Oracle', ax=axd['e'])
    scatter_scores(compare_scores_df, 'Rep Pick', 'Oracle', ax=axd['f'])

    score_delta_kde(compare_scores_df, 'Loc Pick', 'Oracle', ax=axd['g'])
    scatter_scores(compare_scores_df, 'Loc Pick', 'Oracle', ax=axd['h'])

    score_delta_kde(compare_scores_df, 'Both Pick', 'Oracle', ax=axd['k'])
    scatter_scores(compare_scores_df, 'Both Pick', 'Oracle', ax=axd['l'])
    return axd


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



def fig1(track, anno_idx=0):
    fig, axs = plt.subplots(2, 2, figsize=(7,6), sharex=True, sharey=True)
    # top row: CREMA SDM (rep) MFCC SDM (loc)
    quadmesh = librosa.display.specshow(track.sdm(feat='chroma'), y_axis='time', x_axis='time', ax=axs[0][0], hop_length=4096, sr=22050)
    fig.colorbar(quadmesh, ax=axs[0][0])
    
    quadmesh = librosa.display.specshow(track.sdm(feat='mfcc'), y_axis='time', x_axis='time', ax=axs[0][1], hop_length=4096, sr=22050)
    fig.colorbar(quadmesh, ax=axs[0][1])
    

    # bot row: Full Meet Mat   Diag Meet Mat
    diag_meet_mats = track.meet_mats(track.common_ts(), no_rep=True)
    full_meet_mats = track.meet_mats(track.common_ts(), no_rep=False)
    
    quadmesh = librosa.display.specshow(full_meet_mats[anno_idx], x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[1][0])
    fig.colorbar(quadmesh, ax=axs[1][0])
    
    quadmesh = librosa.display.specshow(diag_meet_mats[anno_idx], x_axis='time', y_axis='time', hop_length=4096, sr=22050, ax=axs[1][1])
    fig.colorbar(quadmesh, ax=axs[1][1])
    
    axs[0][0].set_title('Chroma SDM')
    axs[0][1].set_title('MFCC SDM')
    axs[1][0].set_title('Annotation Meet Matrix')
    axs[1][1].set_title('Diagonal Meet Matrix')
    fig.tight_layout()
    return fig, axs


def l_heatmap(tid=None, ax=None, l_type='lr'):
    # figure 2 in paper
    # When tid is None, do the average of the entire corpus.
    # get l_score_df:
    l_df = ssdm.get_l_df(l_type=l_type)
    l_score = pd.read_pickle('./l_score_df.pkl')
    l_df = l_score.loc[(slice(None), l_type), slice(None)]
    l_df.index = l_df.index.droplevel(1)
    
    if tid is None:
        lr_scores_to_show = l_df.mean()
        title_suffix = 'all pairs (Average)'
    else:
        lr_scores_to_show = l_df.loc[tid]
        title_suffix = f'track {tid}'

    
    img_mat = np.zeros((6,6))
    for x, rep_feat in enumerate(ssdm.AVAL_FEAT_TYPES):
        for y, loc_feat in enumerate(ssdm.AVAL_FEAT_TYPES):
            img_mat[x, y] = lr_scores_to_show[rep_feat, loc_feat]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(img_mat, cmap='coolwarm')
    ax.set_xticks(np.arange(6), labels=ssdm.AVAL_FEAT_TYPES)
    ax.set_yticks(np.arange(6), labels=ssdm.AVAL_FEAT_TYPES)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Local Feature')
    ax.set_ylabel('Repetition Feature')
    
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, '{0:.3f}'.format(img_mat[i, j]),
                           ha="center", va="center", color="k")

    score_text = {'lr': 'L-Recall', 'lp': 'L-Precision', 'l': 'L-Measure'}
    title = f"{score_text[l_type]} score for {title_suffix}"
    ax.set_title(title)
    plt.colorbar(im, shrink=0.8)
    return ax
