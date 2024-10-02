import xarray as xr
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import re, itertools
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import scipy.stats as ss

import lightning as L
import wandb
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
import argparse

import ssdm

ds_module_dict = dict(jsd = ssdm.jsd, slm = ssdm.slm, hmx = ssdm.hmx, rwcpop = ssdm.rwcpop)

## Evec Layer
class ExpandEvecs(nn.Module):
    def __init__(self, max_lvl=16):
        super().__init__()
        self.max_lvl=max_lvl

    def forward(self, evecs):
        with torch.set_grad_enabled(False):
            lras = []
            if self.max_lvl == None:
                self.max_lvl = evecs.shape[-1]
            for lvl in range(self.max_lvl):
                first_evecs = evecs[:, :, :, :lvl + 1]
                lras.append(torch.matmul(first_evecs, first_evecs.transpose(-1, -2)))

            cube = torch.cat(lras, dim=1)
        return cube


## MaxPool that doesn't pool to 1
class ConditionalMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(ConditionalMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        # Get the height and width of the input
        height, width = x.shape[2], x.shape[3]

        # Check if both dimensions are greater than 4
        if height > 4 and width > 4:
            # Apply max pooling
            return nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
        else:
            # Return the input unchanged
            return x


# My LVL Loss
class NLvlLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targets):
        diviation = targets.max() - targets
        # losses = torch.linalg.vecdot(pred, diviation)
        losses = torch.nn.functional.cosine_similarity(pred, diviation, dim=1, eps=1e-7)
        log_p = torch.log(pred + 1e-7)
        entropy = -torch.sum(pred * log_p, dim=-1)
        return torch.mean(losses), torch.mean(entropy)


# Custom Conv2d layer that scales gradients based on the number of filter applications
class InputSizeAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        # Wrapping standard Conv2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.H_out = None
        self.W_out = None
        self._register_gradient_hook()

    def forward(self, input):
        # Perform the forward pass, saving the output dimensions (H_out, W_out)
        output = self.conv(input)
        self.H_out, self.W_out = output.size(2), output.size(3)  # Save height and width of the output
        return output

    def _register_gradient_hook(self):    
        # Register hook to scale the gradient based on the number of filter applications
        def hook_fn(grad):
            if self.H_out is not None and self.W_out is not None:
                # Number of times the filter was applied (H_out * W_out)
                scaling_factor = self.H_out * self.W_out
                return grad / scaling_factor
            return grad

        # Apply the hook to the weight gradient
        self.conv.weight.register_hook(hook_fn)
        if self.conv.bias is not None:
            self.conv.bias.register_hook(hook_fn)


class LitMultiModel(L.LightningModule):
    def __init__(self, training_loss_mode='duo', branch_off='early', norm_cnn_grad=True, entropy_scale=0.01, x_conv_filters=1):
        super().__init__()
        self.training_loss_mode = training_loss_mode # 'util', 'nlvl', 'duo', 'triple'
        self.branch_off = branch_off # 'early' or 'late'
        self.norm_cnn_grad = norm_cnn_grad
        self.entropy_scale = entropy_scale
        self.expand_evecs = ExpandEvecs()
        if self.norm_cnn_grad:
            Conv2d = InputSizeAwareConv2d
        else:
            Conv2d = nn.Conv2d

        n = x_conv_filters
        self.convlayers1 = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers2 = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers3 = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers4 = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers5 = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((24, 24))
        self.adapool_med = nn.AdaptiveMaxPool2d((12, 12))
        self.adapool_sm = nn.AdaptiveMaxPool2d((6, 6))

        self.post_big_pool = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.maxpool,
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
            self.maxpool,
        )
        self.post_med_pool = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
            self.maxpool,
        )
        self.post_sm_pool = nn.Sequential(
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
        )

        if self.branch_off == 'late':
            self.shared_linear = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(108, 64, bias=False),
                nn.SiLU(),
            )
            self.util_head = nn.Sequential(
                nn.Linear(64, 16, bias=False),
                nn.SiLU(),
                nn.Linear(16, 16, bias=False),
                nn.SiLU(),
                nn.Linear(16, 1, bias=False),
            )
            self.nlvl_head = nn.Sequential(
                nn.Linear(64, 16, bias=False),
                nn.SiLU(),
                nn.Linear(16, 16, bias=True),
                nn.Softmax(dim=-1)
            )

        elif self.branch_off == 'early':
            self.shared_linear = nn.Identity()
            self.util_head = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(108, 32, bias=False),
                nn.SiLU(),
                nn.Linear(32, 1, bias=False)
            )
            self.nlvl_head = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(108, 32, bias=False),
                nn.SiLU(),
                nn.Linear(32, 16, bias=True),
                nn.Softmax(dim=-1)
            )

        self.lvl_loss_fn = NLvlLoss()
        self.save_hyperparameters()


    def forward(self, x):
        x = self.expand_evecs(x)
        x1 = self.convlayers1(x)
        x2 = self.convlayers2(self.maxpool(x1))
        x3 = self.convlayers3(self.maxpool(x2))
        x4 = self.convlayers4(self.maxpool(x3))
        x5 = self.convlayers5(self.maxpool(x4))
        
        x_sm = self.adapool_sm(x1) + self.adapool_sm(x2) + self.adapool_sm(x3) + self.adapool_sm(x4) + self.adapool_sm(x5)
        x_med = self.adapool_med(x1) + self.adapool_med(x2) + self.adapool_med(x3) + self.adapool_med(x4) + self.adapool_med(x5)
        x_big = self.adapool_big(x1) + self.adapool_big(x2) + self.adapool_big(x3) + self.adapool_big(x4) + self.adapool_big(x5)

        x_sm = self.post_sm_pool(x_sm)
        x_med = self.post_med_pool(x_med)
        x_big = self.post_big_pool(x_big)

        x_sm_ch_softmax = (x_sm.softmax(1) * x_sm).sum(1)
        x_med_ch_softmax = (x_med.softmax(1) * x_med).sum(1)
        x_big_ch_softmax = (x_big.softmax(1) * x_big).sum(1)

        multires_embeddings = torch.cat([torch.flatten(x_sm_ch_softmax, 1), 
                                         torch.flatten(x_med_ch_softmax, 1), 
                                         torch.flatten(x_big_ch_softmax, 1)], 1)
        shared_embedding = self.shared_linear(multires_embeddings)
        return self.util_head(shared_embedding), self.nlvl_head(shared_embedding)


    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.cat([batch['x1'], batch['x2']], 0)
        contrast_label = batch['perf_gap'].sign()
        
        util_est, nlvl_est = self(x)
        ranking_loss = nn.functional.margin_ranking_loss(
            util_est[0, None], util_est[1, None], 
            contrast_label, margin=1
        )
        self.log('ranking loss', ranking_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        # Compute the lvl loss only on samples with good feature segmentations
        lvl_loss_accumulator = torch.tensor([0.0], device=ranking_loss.device)
        entropy_accumulator = torch.tensor([0.0], device=ranking_loss.device)
        if batch['x1_rank'] <= 8:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].view(1, 16), batch['x1_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss
            entropy_accumulator += entropy
        if batch['x2_rank'] <= 8:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[1].view(1, 16), batch['x2_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss
            entropy_accumulator += entropy
        if batch['x2_rank'] > 8 and batch['x1_rank'] > 8: # This is a hack loop to makesure DDP doesn't complain about unsued parameters
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].view(1, 16), batch['x1_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss * 1e-8
            entropy_accumulator += entropy * 1e-8

        lvl_loss = lvl_loss_accumulator
        entropy = entropy_accumulator

        self.log('nlvl loss', lvl_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('entropy', entropy, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        if self.training_loss_mode == 'util':
            return ranking_loss + lvl_loss * 1e-8
        elif self.training_loss_mode == 'nlvl':
            return ranking_loss * 1e-8 + lvl_loss - entropy * self.entropy_scale
        elif self.training_loss_mode == 'duo':
            return ranking_loss + lvl_loss - entropy * self.entropy_scale
        else:
            return None


    def on_validation_epoch_start(self):
        self.trainer.datamodule.setup('validate')
        self.val_ds = self.trainer.datamodule.val_dataloader().dataset
        # full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        self.util_result_coords = dict(
            tid=[], 
            rep_ftype=self.val_ds.AVAL_FEAT_TYPES, loc_ftype=self.val_ds.AVAL_FEAT_TYPES,
        )
        self.nlvl_result_coords = dict(
            tid=[], 
            rep_ftype=self.val_ds.AVAL_FEAT_TYPES, loc_ftype=self.val_ds.AVAL_FEAT_TYPES,
            layer=list(range(1,17))
        )
        self.val_util_predictions = xr.DataArray(np.nan, 
                                                    coords=self.util_result_coords, 
                                                    dims=self.util_result_coords.keys()
                                                    ).sortby('tid')
        self.val_nlvl_predictions = xr.DataArray(np.nan, 
                                                    coords=self.nlvl_result_coords, 
                                                    dims=self.nlvl_result_coords.keys()
                                                    ).sortby('tid')


    def validation_step(self, batch, batch_idx):
        full_tid = batch['info']
        
        track_result_coords = self.util_result_coords.copy()
        track_result_coords.update(tid=[full_tid])
        track_util_result = xr.DataArray(np.nan, coords=track_result_coords, dims=track_result_coords.keys())
        track_result_coords.update(layer=list(range(1,17)))
        track_nlvl_result = xr.DataArray(np.nan, coords=track_result_coords, dims=track_result_coords.keys())
        
        for i, feat_pair in enumerate(batch['feat_order']):
            rep_feat, loc_feat = feat_pair.split('_')
            util, nlvl = self(batch['data'][i][None, :])
            track_util_result.loc[full_tid, rep_feat, loc_feat] = util.cpu().numpy().squeeze()
            track_nlvl_result.loc[full_tid, rep_feat, loc_feat] = nlvl.cpu().numpy().squeeze()

        track_util_output = track_util_result.loc[full_tid].squeeze().to_numpy().flatten()
        try:
            roc_auc = self.get_util_roc_auc(full_tid, track_util_output)
            self.log('val roc_auc', roc_auc, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        except:
            print('roc failed')
        self.val_util_predictions = xr.concat([self.val_util_predictions, track_util_result], dim="tid")
        self.val_nlvl_predictions = xr.concat([self.val_nlvl_predictions, track_nlvl_result], dim="tid")
        return None

    
    def get_util_roc_auc(self, full_tid, track_util_output):
        track_lmeasure = self.val_ds.scores.sel(tid=full_tid).max('layer').squeeze().to_numpy().flatten()
        track_good_combos = [track_lmeasure.max() - l <= 0.02 for l in track_lmeasure]
        try:
            # Some tracks are degenerate, and has only 1 class in track_good_combos. roc_auc not defined in such cases.
            return roc_auc_score(track_good_combos, track_util_output)
        except:
            print(full_tid, 'failed roc_auc due to single class')
            return 0.5


    def on_validation_epoch_end(self):
        self.val_util_predictions = self.val_util_predictions.dropna(dim="tid").sortby('tid')
        self.val_nlvl_predictions = self.val_nlvl_predictions.dropna(dim="tid").sortby('tid')

        if self.trainer.world_size > 1:
            # Gather xarray.DataArray objects from all ranks
            gathered_val_util_predictions = [None] * self.trainer.world_size
            gathered_val_nlvl_predictions = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(gathered_val_util_predictions, self.val_util_predictions)
            torch.distributed.all_gather_object(gathered_val_nlvl_predictions, self.val_nlvl_predictions)
            self.val_util_predictions = xr.concat(gathered_val_util_predictions, dim='tid').sortby('tid')
            self.val_nlvl_predictions = xr.concat(gathered_val_nlvl_predictions, dim='tid').sortby('tid')

        
        tid_subset = self.val_util_predictions.tid
        ds_score = self.val_ds.scores.sel(tid=tid_subset).max('layer').sortby('tid').squeeze()
        
        net_feat_picks = self.val_util_predictions.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])
        
        # Logging average utility scores
        net_pick = ds_score.isel(net_feat_picks)
        boa_pick = ds_score.isel(boa_feat_picks)
        orc_pick = ds_score.isel(orc_feat_picks)
        self.log('net pick', net_pick.mean().item(), sync_dist=True)
        self.log('net_feat orc_lvl', net_pick.mean().item(), sync_dist=True)
        self.log('boa_feat orc_lvl', boa_pick.mean().item(), sync_dist=True)
        self.log('orc_feat orc_lvl', orc_pick.mean().item(), sync_dist=True)
        self.log('net_feat orc_lvl std', net_pick.std().item(), sync_dist=True)
        self.log('boa_feat orc_lvl std', boa_pick.std().item(), sync_dist=True)
        self.log('orc_feat orc_lvl std', orc_pick.std().item(), sync_dist=True)
        # print('net, boa, orc:', net_pick, boa_pick, orc_pick)
        nlvl_outputs = self.val_nlvl_predictions.isel(orc_feat_picks).squeeze().to_numpy()

        # Log table to wandb
        try:
            wandb.log({'nlvl ouputs': wandb.Image(nlvl_outputs)})
            self.log_wandb_table(tid_subset, net_feat_picks, boa_feat_picks, orc_feat_picks)
        except:
            print('could not log to wandb')


    def log_wandb_table(self, tid_subset, net_feat_picks, boa_feat_picks, orc_feat_picks):
        # Logging to wandb table
        val_wandb_table = wandb.Table(columns = [
            'tid', 'title',
            # 'track/l_measure', 'track/net_util',
            # 'track/annotation',
            'net/score', 'net/feat', 
            'boa/score', 'boa/feat', 
            'orc/score', 'orc/feat', 
            'best_lvl', 'net_lvl',
            'util AUCROC'
        ])

        ds_score = self.val_ds.scores.sel(tid=tid_subset).max('layer').sortby('tid').squeeze()

        for tid in self.val_util_predictions.tid:
            # track_lmeasure_plot = wandb.Image(ssdm.viz.heatmap(ds_score.sel(tid=tid))[0])
            # track_net_util_plot = wandb.Image(ssdm.viz.heatmap(self.val_util_predictions.sel(tid=tid))[0])
            ds_module = ds_module_dict[re.sub('\d', '', tid.item())]
            track = ds_module.Track(re.sub('\D', '', tid.item()))
            # track_annotation_plot = wandb.Image(ssdm.viz.anno_meet_mats(track)[0])
            ssdm.viz.plt.close('all')
            track_util_output = self.val_util_predictions.sel(tid=tid).to_numpy().flatten()

            val_wandb_table.add_data(
                tid.item(), track.title,
                # track_lmeasure_plot, track_net_util_plot,
                # track_annotation_plot,
                ds_score.isel(net_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(net_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(net_feat_picks).sel(tid=tid).loc_ftype.item(),
                ds_score.isel(boa_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(boa_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(boa_feat_picks).sel(tid=tid).loc_ftype.item(),
                ds_score.isel(orc_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(orc_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(orc_feat_picks).sel(tid=tid).loc_ftype.item(),
                self.val_ds.scores.sel(tid=tid_subset).squeeze().isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item(), 
                self.val_nlvl_predictions.isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item(),
                self.get_util_roc_auc(tid, track_util_output)
            )
        wandb.log({'validation results': val_wandb_table})
        return None

    
    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n],
                "weight_decay": 0.005,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(grouped_parameters)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=2e-6, max_lr=1e-2, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=5000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 2}]


class LitMultiModel914n(L.LightningModule):
    def __init__(self, training_loss_mode='duo', branch_off='early', norm_cnn_grad=True, entropy_scale=0.01):
        super().__init__()
        self.training_loss_mode = training_loss_mode # 'util', 'nlvl', 'duo', 'triple'
        self.branch_off = branch_off # 'early' or 'late'
        self.norm_cnn_grad = norm_cnn_grad
        self.entropy_scale = entropy_scale
        if self.norm_cnn_grad:
            Conv2d = InputSizeAwareConv2d
        else:
            Conv2d = nn.Conv2d
        self.expand_evecs = ExpandEvecs()
        self.convlayers1 = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True, eps=1e-3), nn.SiLU()
        )
        self.convlayers2 = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True, eps=1e-3), nn.SiLU()
        )
        self.convlayers3 = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True, eps=1e-3), nn.SiLU()
        )
        self.convlayers4 = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True, eps=1e-3), nn.SiLU()
        )
        self.post_big_pool = nn.Sequential(
            Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True, eps=1e-3), nn.SiLU(),
            Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True, eps=1e-3), nn.SiLU()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((32, 32))
        self.adapool_med = nn.AdaptiveMaxPool2d((16, 16))
        self.adapool_sm = nn.AdaptiveMaxPool2d((4, 4)) 

        if self.branch_off == 'late':
            self.shared_linear = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(528, 64, bias=False),
                nn.SiLU(),
            )
            self.util_head = nn.Linear(64, 1, bias=False)
            self.nlvl_head = nn.Sequential(
                nn.Linear(64, 16, bias=True),
                nn.Softmax(dim=-1)
            )

        elif self.branch_off == 'early':
            self.shared_linear = nn.Identity()
            self.util_head = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(528, 32, bias=False),
                nn.SiLU(),
                nn.Linear(32, 1, bias=False)
            )
            self.nlvl_head = nn.Sequential(
                nn.Dropout(0.2, inplace=False),
                nn.Linear(528, 32, bias=False),
                nn.SiLU(),
                nn.Linear(32, 16, bias=True),
                nn.Softmax(dim=-1)
            )

        self.lvl_loss_fn = NLvlLoss()
        self.save_hyperparameters()


    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x1 = self.convlayers2(self.maxpool(x))
        x2 = self.convlayers3(self.maxpool(x1))
        x3 = self.convlayers4(self.maxpool(x2))
        
        x_sm = self.adapool_sm(x1) + self.adapool_sm(x2) + self.adapool_sm(x3)
        x_med = self.adapool_med(x1) + self.adapool_med(x2) + self.adapool_med(x3)
        x_big = self.adapool_big(x1) + self.adapool_big(x2) + self.adapool_big(x3)
        x_big = self.maxpool(self.post_big_pool(x_big))

        x_sm_ch_softmax = (x_sm.softmax(1) * x_sm).sum(1)
        x_med_ch_softmax = (x_med.softmax(1) * x_med).sum(1)
        x_big_ch_softmax = (x_big.softmax(1) * x_big).sum(1)

        multires_embeddings = torch.cat([torch.flatten(x_sm_ch_softmax, 1), 
                                         torch.flatten(x_med_ch_softmax, 1), 
                                         torch.flatten(x_big_ch_softmax, 1)], 1)
        shared_embedding = self.shared_linear(multires_embeddings)
        return self.util_head(shared_embedding), self.nlvl_head(shared_embedding)


    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.cat([batch['x1'], batch['x2']], 0)
        contrast_label = batch['perf_gap'].sign()
        
        util_est, nlvl_est = self(x)
        ranking_loss = nn.functional.margin_ranking_loss(
            util_est[0, None], util_est[1, None], 
            contrast_label, margin=1
        )
        self.log('ranking loss', ranking_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        # Compute the lvl loss only on samples with good feature segmentations
        lvl_loss_accumulator = torch.tensor([0.0], device=ranking_loss.device)
        entropy_accumulator = torch.tensor([0.0], device=ranking_loss.device)
        if batch['x1_rank'] <= 8:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].view(1, 16), batch['x1_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss
            entropy_accumulator += entropy
        if batch['x2_rank'] <= 8:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[1].view(1, 16), batch['x2_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss
            entropy_accumulator += entropy
        if batch['x2_rank'] > 8 and batch['x1_rank'] > 8: # This is a hack loop to makesure DDP doesn't complain about unsued parameters
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].view(1, 16), batch['x1_layer_score'].view(1, 16))
            lvl_loss_accumulator += lvl_loss * 1e-8
            entropy_accumulator += entropy * 1e-8

        lvl_loss = lvl_loss_accumulator
        entropy = entropy_accumulator

        self.log('nlvl loss', lvl_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('entropy', entropy, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        if self.training_loss_mode == 'util':
            return ranking_loss + lvl_loss * 1e-8
        elif self.training_loss_mode == 'nlvl':
            return ranking_loss * 1e-8 + lvl_loss - entropy * self.entropy_scale
        elif self.training_loss_mode == 'duo':
            return ranking_loss + lvl_loss - entropy * self.entropy_scale
        else:
            return None


    def on_validation_epoch_start(self):
        self.trainer.datamodule.setup('validate')
        self.val_ds = self.trainer.datamodule.val_dataloader().dataset
        # full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        self.util_result_coords = dict(
            tid=[], 
            rep_ftype=self.val_ds.AVAL_FEAT_TYPES, loc_ftype=self.val_ds.AVAL_FEAT_TYPES,
        )
        self.nlvl_result_coords = dict(
            tid=[], 
            rep_ftype=self.val_ds.AVAL_FEAT_TYPES, loc_ftype=self.val_ds.AVAL_FEAT_TYPES,
            layer=list(range(1,17))
        )
        self.val_util_predictions = xr.DataArray(np.nan, 
                                                    coords=self.util_result_coords, 
                                                    dims=self.util_result_coords.keys()
                                                    ).sortby('tid')
        self.val_nlvl_predictions = xr.DataArray(np.nan, 
                                                    coords=self.nlvl_result_coords, 
                                                    dims=self.nlvl_result_coords.keys()
                                                    ).sortby('tid')


    def validation_step(self, batch, batch_idx):
        full_tid = batch['info']
        
        track_result_coords = self.util_result_coords.copy()
        track_result_coords.update(tid=[full_tid])
        track_util_result = xr.DataArray(np.nan, coords=track_result_coords, dims=track_result_coords.keys())
        track_result_coords.update(layer=list(range(1,17)))
        track_nlvl_result = xr.DataArray(np.nan, coords=track_result_coords, dims=track_result_coords.keys())
        
        for i, feat_pair in enumerate(batch['feat_order']):
            rep_feat, loc_feat = feat_pair.split('_')
            util, nlvl = self(batch['data'][i][None, :])
            track_util_result.loc[full_tid, rep_feat, loc_feat] = util.cpu().numpy().squeeze()
            track_nlvl_result.loc[full_tid, rep_feat, loc_feat] = nlvl.cpu().numpy().squeeze()

        try:
            track_util_output = track_util_result.loc[full_tid].squeeze().to_numpy().flatten()
            roc_auc = self.get_util_roc_auc(full_tid, track_util_output)
            self.log('val roc_auc', roc_auc, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        except:
            pass
        self.val_util_predictions = xr.concat([self.val_util_predictions, track_util_result], dim="tid")
        self.val_nlvl_predictions = xr.concat([self.val_nlvl_predictions, track_nlvl_result], dim="tid")
        return None

    
    def get_util_roc_auc(self, full_tid, track_util_output):
        track_lmeasure = self.val_ds.scores.sel(tid=full_tid).max('layer').squeeze().to_numpy().flatten()
        track_good_combos = [track_lmeasure.max() - l <= 0.02 for l in track_lmeasure]
        # track_good_combos = [r <= 6 for r in ss.rankdata(track_lmeasure)]
        try:
            # Some tracks are degenerate, and has only 1 class in track_good_combos. roc_auc not defined in such cases.
            return roc_auc_score(track_good_combos, track_util_output)
        except:
            print(full_tid, 'failed roc_auc due to single class')
            return 0.5


    def on_validation_epoch_end(self):
        self.val_util_predictions = self.val_util_predictions.dropna(dim="tid").sortby('tid')
        self.val_nlvl_predictions = self.val_nlvl_predictions.dropna(dim="tid").sortby('tid')

        if self.trainer.world_size > 1:
            # Gather xarray.DataArray objects from all ranks
            gathered_val_util_predictions = [None] * self.trainer.world_size
            gathered_val_nlvl_predictions = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(gathered_val_util_predictions, self.val_util_predictions)
            torch.distributed.all_gather_object(gathered_val_nlvl_predictions, self.val_nlvl_predictions)
            self.val_util_predictions = xr.concat(gathered_val_util_predictions, dim='tid').sortby('tid')
            self.val_nlvl_predictions = xr.concat(gathered_val_nlvl_predictions, dim='tid').sortby('tid')
        
        tid_subset = self.val_util_predictions.tid
        if self.val_ds.name in ['spam']:
            return self.val_util_predictions
        ds_score = self.val_ds.scores.sel(tid=tid_subset).max('layer').sortby('tid').squeeze()
        
        net_feat_picks = self.val_util_predictions.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])
        
        # Logging average utility scores
        net_pick = ds_score.isel(net_feat_picks)
        boa_pick = ds_score.isel(boa_feat_picks)
        orc_pick = ds_score.isel(orc_feat_picks)
        self.log('net pick', net_pick.mean().item(), sync_dist=True)
        self.log('net_feat orc_lvl', net_pick.mean().item(), sync_dist=True)
        self.log('boa_feat orc_lvl', boa_pick.mean().item(), sync_dist=True)
        self.log('orc_feat orc_lvl', orc_pick.mean().item(), sync_dist=True)
        self.log('net_feat orc_lvl std', net_pick.std().item(), sync_dist=True)
        self.log('boa_feat orc_lvl std', boa_pick.std().item(), sync_dist=True)
        self.log('orc_feat orc_lvl std', orc_pick.std().item(), sync_dist=True)
        # print('net, boa, orc:', net_pick, boa_pick, orc_pick)
        nlvl_outputs = self.val_nlvl_predictions.isel(orc_feat_picks).squeeze().to_numpy()
        # Log table to wandb
        try:
            self.trainer.logger.experiment.log({'nlvl ouputs': wandb.Image(nlvl_outputs)})
            self.log_wandb_table(tid_subset, net_feat_picks, boa_feat_picks, orc_feat_picks)
        except:
            print('could not log to wandb')
        
        return self.val_util_predictions


    def log_wandb_table(self, tid_subset, net_feat_picks, boa_feat_picks, orc_feat_picks):
        # Logging to wandb table
        val_wandb_table = wandb.Table(columns = [
            'tid', 'title',
            # 'track/l_measure', 'track/net_util',
            # 'track/annotation',
            'net/score', 'net/feat', 
            'boa/score', 'boa/feat', 
            'orc/score', 'orc/feat', 
            'best_lvl', 'net_lvl',
            'util AUCROC'
        ])

        ds_score = self.val_ds.scores.sel(tid=tid_subset).max('layer').sortby('tid').squeeze()

        for tid in self.val_util_predictions.tid:
            # track_lmeasure_plot = wandb.Image(ssdm.viz.heatmap(ds_score.sel(tid=tid))[0])
            # track_net_util_plot = wandb.Image(ssdm.viz.heatmap(self.val_util_predictions.sel(tid=tid))[0])
            ds_module = ds_module_dict[re.sub('\d', '', tid.item())]
            track = ds_module.Track(re.sub('\D', '', tid.item()))
            # track_annotation_plot = wandb.Image(ssdm.viz.anno_meet_mats(track)[0])
            ssdm.viz.plt.close('all')
            track_util_output = self.val_util_predictions.sel(tid=tid).to_numpy().flatten()

            val_wandb_table.add_data(
                tid.item(), track.title,
                # track_lmeasure_plot, track_net_util_plot,
                # track_annotation_plot,
                ds_score.isel(net_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(net_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(net_feat_picks).sel(tid=tid).loc_ftype.item(),
                ds_score.isel(boa_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(boa_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(boa_feat_picks).sel(tid=tid).loc_ftype.item(),
                ds_score.isel(orc_feat_picks).sel(tid=tid).item(), 
                ds_score.isel(orc_feat_picks).sel(tid=tid).rep_ftype.item() + '_' + ds_score.isel(orc_feat_picks).sel(tid=tid).loc_ftype.item(),
                self.val_ds.scores.sel(tid=tid_subset).squeeze().isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item(), 
                self.val_nlvl_predictions.isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item(),
                self.get_util_roc_auc(tid, track_util_output)
            )
        self.trainer.logger.experiment.log({'validation results': val_wandb_table})
        return None
    
    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n and 'conv' not in n],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n and 'conv' in n],
                "weight_decay": 0.005,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(grouped_parameters)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=1e-5, max_lr=1e-2, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=3000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 3}]


# New validation dataloaders...
class PairDataModule(L.LightningDataModule):
    def __init__(self, ds_module=ssdm.rwcpop, perf_margin=0.05, val_split='val', predict_split=None, num_loader_workers=4):
        super().__init__()
        self.perf_margin = perf_margin
        self.ds_module = ds_module
        self.val_split = val_split
        self.predict_split = predict_split
        self.num_loader_workers = num_loader_workers
    
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = self.ds_module.PairDSLmeasure(split='train', perf_margin=self.perf_margin) 
            # self.val_ds = self.ds_module.PairDSLmeasure(split='val', perf_margin=self.perf_margin) 
            self.val_infer_ds = self.ds_module.InferDS(split=self.val_split)

        if stage == 'validate':
            self.val_infer_ds = self.ds_module.InferDS(split=self.val_split)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_infer_ds = self.ds_module.InferDS(split='test')

        if stage == "predict":
            self.predict_ds = self.ds_module.InferDS(split=self.predict_split) 


    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, shuffle=True,
            num_workers=self.num_loader_workers, pin_memory=True,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_infer_ds, batch_size=None, shuffle=False,
            num_workers=self.num_loader_workers, pin_memory=True,
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_infer_ds, batch_size=None, shuffle=False,
            num_workers=self.num_loader_workers, pin_memory=True,
        )


    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, batch_size=None, shuffle=False,
            num_workers=self.num_loader_workers, pin_memory=True,
        )
    

class DevPairDataModule(PairDataModule):
    def __init__(self, perf_margin=0.01, split='single'):
        super().__init__(ds_module=ssdm.rwcpop, perf_margin=perf_margin)
        self.split = split


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = ssdm.rwcpop.PairDSLmeasure(split=self.split, perf_margin=self.perf_margin) 
            self.val_infer_ds = ssdm.rwcpop.InferDS(split=self.split)

        if stage == 'validate':
            self.val_infer_ds = ssdm.rwcpop.InferDS(split=self.split)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_infer_ds = ssdm.rwcpop.InferDS(split=self.split)

        if stage == "predict":
            self.predict_ds = ssdm.rwcpop.InferDS(split=self.split) 


class HybridPairDM(L.LightningDataModule):
    def __init__(self, ds_modules=[ssdm.rwcpop], perf_margin=0.05, val_split='val', num_loader_workers=4):
        super().__init__()
        self.perf_margin = perf_margin
        self.ds_modules = ds_modules
        self.val_split = val_split
        self.num_loader_workers = num_loader_workers

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_ds_list, val_ds_list, score_list, vmeasure_list = [], [], [], []
            for ds_module in self.ds_modules:
                train_ds_list.append(ds_module.PairDSLmeasure(split='train', perf_margin=self.perf_margin))
                val_ds_list.append(ds_module.InferDS(split=self.val_split))
                score_list.append(val_ds_list[-1].scores)
                vmeasure_list.append(val_ds_list[-1].vmeasures)
            self.train_ds = ConcatDataset(train_ds_list)
            self.val_infer_ds = ConcatDataset(val_ds_list)
            self.val_infer_ds.scores = xr.concat(score_list, dim='tid')
            self.val_infer_ds.vmeasures = xr.concat(vmeasure_list, dim='tid')
            self.val_infer_ds.AVAL_FEAT_TYPES = val_ds_list[0].AVAL_FEAT_TYPES

        if stage == 'validate':
            val_ds_list, score_list, vmeasure_list = [], [], []
            for ds_module in self.ds_modules:
                val_ds_list.append(ds_module.InferDS(split=self.val_split))
                score_list.append(val_ds_list[-1].scores)
                vmeasure_list.append(val_ds_list[-1].vmeasures)
            self.val_infer_ds = ConcatDataset(val_ds_list)
            self.val_infer_ds.scores = xr.concat(score_list, dim='tid')
            self.val_infer_ds.vmeasures = xr.concat(vmeasure_list, dim='tid')
            self.val_infer_ds.AVAL_FEAT_TYPES = val_ds_list[0].AVAL_FEAT_TYPES
        
        

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, shuffle=True,
            num_workers=self.num_loader_workers, pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_infer_ds, batch_size=None, shuffle=False,
            num_workers=self.num_loader_workers, pin_memory=True,
        )


def run_inference(model_path, dataset_id='rwcpop', split='test', project='ssdm_test'):
    # Run inference on given dataset split with model and log to wandb.
    # ds_module_dict = dict(jsd = ssdm.jsd, slm = ssdm.slm, hmx = ssdm.hmx, rwcpop = ssdm.rwcpop)
    if dataset_id == 'all':
        dm = HybridPairDM(ds_modules=[ds_module_dict[ds] for ds in ds_module_dict], perf_margin=0)
    else:
        dm = PairDataModule(ds_module=ds_module_dict[dataset_id], perf_margin=0, val_split=split)
    lit_net = LitMultiModel.load_from_checkpoint(model_path)
    wandb_logger = L.pytorch.loggers.WandbLogger(project=project, name=dataset_id+'_'+split)
    validator = L.pytorch.Trainer(
        num_sanity_val_steps = 0,
        max_epochs = 1,
        accelerator = "gpu",
        logger = wandb_logger
    )
    wandb_logger.experiment.config.update(dict(model_path=model_path, applied_to=dataset_id+'_'+split))
    validator.validate(lit_net, datamodule=dm)
    wandb.finish()
    # Check validator.util_predictions and validator.nlvl_predictions for xr dataarrays storing all results
    return validator 


def debug_trinity(name='debug'):
    wandb_logger = L.pytorch.loggers.WandbLogger(project='debug', name=name)
    dm = HybridPairDM(ds_modules=[ds_module_dict[ds] for ds in ds_module_dict], perf_margin=0.05)
    model = LitMultiModel(training_loss_mode='duo', branch_off='late', entropy_scale=0.05)
    trainer = L.pytorch.Trainer(max_epochs=1, logger=wandb_logger)
    return trainer, model, dm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Util Model')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('num_gpu', help='number of gpus to setup trainer')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')
    kwargs = parser.parse_args()

    margin, ets, filter_multiple, ds = list(itertools.product(
        [0.05],
        [0.05],
        [1],
        ['all', 'hmx', 'jsd', 'rwcpop', 'slm'],
    ))[int(kwargs.config_idx)]

    loss_mode = 'duo'
    # ds = 'rwcpop'
    branch_off = 'late'

    # initialise the wandb logger and name your wandb project
    wandb_logger = L.pytorch.loggers.WandbLogger(project='ssdm3', name=kwargs.date[-4:]+ds+str(margin)+loss_mode+'_m'+str(filter_multiple)+'_ets'+str(ets))
    # Log things
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(dict(margin=margin, ds=ds))

    checkpoint_callback = ModelCheckpoint(
        monitor='val roc_auc',  # The metric to monitor
        dirpath=f'checkpoints/{ds}{margin}{loss_mode}{branch_off}/',  # Directory to save the checkpoints
        filename='{epoch:02d}-{val roc_auc:.4f}',  # Naming convention for checkpoints
        save_top_k=15,  # Save only the best model based on the monitored metric
        mode='max',  # Mode to determine if a lower or higher metric is better
        save_last=True  # Optionally save the last epoch model as well
    )

    trainer = L.pytorch.Trainer(
        max_epochs = 50,
        max_steps = 5000 * 1000,
        devices = int(kwargs.num_gpu),
        accelerator = "gpu",
        accumulate_grad_batches = 8,
        logger = wandb_logger,
        val_check_interval = 0.2,
        callbacks = [TQDMProgressBar(refresh_rate=500),
                     LearningRateMonitor(logging_interval='step'),
                     checkpoint_callback]
    )

    if ds == 'all':
        dm = HybridPairDM(ds_modules=[ds_module_dict[ds] for ds in ds_module_dict], perf_margin=margin)
    else:
        dm = PairDataModule(ds_module=ds_module_dict[ds], perf_margin=margin)
    lit_net = LitMultiModel(
        training_loss_mode=loss_mode, 
        branch_off=branch_off,
        norm_cnn_grad=True,
        entropy_scale=ets,
        x_conv_filters=filter_multiple,
    )
    wandb_logger.watch(lit_net, log='all', log_graph=False)
    trainer.fit(lit_net, datamodule=dm)
    wandb.finish()
    print('done without failure!')

