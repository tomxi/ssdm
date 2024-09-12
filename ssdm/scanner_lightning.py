import xarray as xr
import numpy as np
import torch
import re, itertools
from torch import optim, nn
from torch.utils.data import DataLoader

import lightning as L
import wandb
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
import argparse

from ssdm.scanner import ExpandEvecs, ConditionalMaxPool2d
from ssdm import AVAL_FEAT_TYPES
import ssdm

# My LVL Loss
class NLvlLoss(torch.nn.Module):
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
        self._register_gradient_hook(256)

    def forward(self, input):
        # Perform the forward pass, saving the output dimensions (H_out, W_out)
        output = self.conv(input)
        self.H_out, self.W_out = output.size(2), output.size(3)  # Save height and width of the output
        return output

    def _register_gradient_hook(self, c):    
        # Register hook to scale the gradient based on the number of filter applications
        def hook_fn(grad):
            if self.H_out is not None and self.W_out is not None:
                # Number of times the filter was applied (H_out * W_out)
                scaling_factor = self.H_out * self.W_out
                return grad / scaling_factor * c
            return grad

        # Apply the hook to the weight gradient
        self.conv.weight.register_hook(hook_fn)
        if self.conv.bias is not None:
            self.conv.bias.register_hook(hook_fn)
              

class LitMultiModel(L.LightningModule):
    def __init__(self, training_loss_mode='util', branch_off='early', norm_cnn_grad=True, entropy_scale=0.1):
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

        # Compute the lvl loss only on the better sample
        if 1 in contrast_label:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].view(1, 16), batch['x1_layer_score'].view(1, 16))
        else:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[1].view(1, 16), batch['x2_layer_score'].view(1, 16))
        self.log('nlvl loss', lvl_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('entropy', entropy, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        if self.training_loss_mode == 'util':
            return ranking_loss
        elif self.training_loss_mode == 'nlvl':
            return lvl_loss - entropy * self.entropy_scale
        elif self.training_loss_mode == 'duo':
            return ranking_loss + lvl_loss - entropy * self.entropy_scale
        else:
            return None


    def on_validation_epoch_start(self):
        self.trainer.datamodule.setup('validate')
        self.val_ds = self.trainer.datamodule.val_dataloader().dataset
        full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        util_result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
        )
        nlvl_result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
            layer=list(range(1,17))
        )
        self.trainer.util_predictions = xr.DataArray(np.nan, 
                                                    coords=util_result_coords, 
                                                    dims=util_result_coords.keys()
                                                    ).sortby('tid')
        self.trainer.nlvl_predictions = xr.DataArray(np.nan, 
                                                    coords=nlvl_result_coords, 
                                                    dims=nlvl_result_coords.keys()
                                                    ).sortby('tid')


    def validation_step(self, batch, batch_idx):
        tid, rep_feat, loc_feat = batch['info'].split('_')
        util, nlvl = self(batch['data'])
        self.trainer.util_predictions.loc[self.val_ds.name+tid, rep_feat, loc_feat] = util.item()
        self.trainer.nlvl_predictions.loc[self.val_ds.name+tid, rep_feat, loc_feat] = nlvl.detach().cpu().numpy().squeeze()


    def on_validation_epoch_end(self):
        ds_score = self.val_ds.scores.max('layer').sortby('tid')
        net_feat_picks = self.trainer.util_predictions.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])
        
        # Logging average scores
        net_pick = ds_score.isel(net_feat_picks).mean().item()
        boa_pick = ds_score.isel(boa_feat_picks).mean().item()
        orc_pick = ds_score.isel(orc_feat_picks).mean().item()
        self.log('net pick', net_pick, sync_dist=True)
        self.log('boa pick', boa_pick, sync_dist=True)
        self.log('orc pick', orc_pick, sync_dist=True)
        # print('net, boa, orc:', net_pick, boa_pick, orc_pick)

        # Log table to wandb
        self.log_wandb_table(ds_score, net_feat_picks, boa_feat_picks, orc_feat_picks)


    def log_wandb_table(self, ds_score, net_feat_picks, boa_feat_picks, orc_feat_picks):
        # Logging to wandb table
        val_wandb_at = wandb.Artifact("validation" + str(wandb.run.id), type="validation predictions")
        val_wandb_table = wandb.Table(columns = [
            'tid', 'title',
            # 'track/l_measure', 'track/net_util',
            # 'track/annotation',
            'net/score', 'net/feat', 
            'boa/score', 'boa/feat', 
            'orc/score', 'orc/feat', 
            'orc/feat/best_lvl', 'orc/feat/net_lvl'
        ])

        for tid in self.trainer.util_predictions.tid:
            # track_lmeasure_plot = wandb.Image(ssdm.viz.heatmap(ds_score.sel(tid=tid))[0])
            # track_net_util_plot = wandb.Image(ssdm.viz.heatmap(self.trainer.util_predictions.sel(tid=tid))[0])
            track = self.val_ds.ds_module.Track(re.sub('\D', '', tid.item()))
            # track_annotation_plot = wandb.Image(ssdm.viz.anno_meet_mats(track)[0])
            # ssdm.viz.plt.close('all')

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
                self.val_ds.scores.isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item(), 
                self.trainer.nlvl_predictions.isel(orc_feat_picks).sel(tid=tid).idxmax('layer').item()
            )
        wandb.log({'validation results': val_wandb_table})
        val_wandb_at.add(val_wandb_table, "validation predictions")
        wandb.run.log_artifact(val_wandb_at)
        return None


    def predict_step(self, batch, batch_idx):
        tid, rep_feat, loc_feat = batch['info'].split('_')
        x, nlvl = self(batch['data'])
        self.trainer.prediction.loc[self.predict_ds.name+tid, rep_feat, loc_feat] = x.item()
        return nlvl.squeeze()


    def on_predict_start(self):
        self.trainer.datamodule.setup('predict')
        self.predict_ds = self.trainer.datamodule.predict_dataloader().dataset
        full_tids = [self.predict_ds.name + tid for tid in self.predict_ds.tids]
        result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
        )
        self.trainer.prediction = xr.DataArray(np.nan, 
                                             coords=result_coords, 
                                             dims=result_coords.keys()
                                            ).sortby('tid')

    
    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n and 'conv' not in n],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n and 'conv' in n],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(grouped_parameters)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=1e-6, max_lr=1e-2, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=4000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 5}]


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


def run_inference(model_path, dataset_id='rwcpop', split='test'):
    # Run inference on given dataset split with model and log to wandb.
    ds_module_dict = dict(jsd = ssdm.jsd, slm = ssdm.slm, hmx = ssdm.hmx, rwcpop = ssdm.rwcpop)
    dm = PairDataModule(ds_module=ds_module_dict[dataset_id], perf_margin=0, val_split=split)
    lit_net = LitMultiModel.load_from_checkpoint(model_path)
    wandb_logger = L.pytorch.loggers.WandbLogger(project='ssdm_test', name=dataset_id+'_'+split)
    validator = L.pytorch.Trainer(
        num_sanity_val_steps = 0,
        max_epochs = 1,
        devices = 1,
        accelerator = "gpu",
        logger = wandb_logger
    )
    wandb_logger.experiment.config.update(dict(model_path=model_path, applied_to=dataset_id+'_'+split))
    validator.validate(lit_net, datamodule=dm)
    wandb.finish()
    # Check validator.util_predictions and validator.nlvl_predictions for xr dataarrays storing all results
    return validator 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')
    kwargs = parser.parse_args()

    loss_mode, ds, branch_off = list(itertools.product(
        ['duo', 'nlvl'],
        ['hmx', 'slm', 'jsd', 'rwcpop'],
        ['late'],
    ))[int(kwargs.config_idx)]

    if ds in ['hmx', 'slm']:
        margin = 0.06
    elif ds in ['jsd', 'rwcpop']:
        margin = 0.03
    else:
        # Default specified by the script
        margin = 0.1

    # initialise the wandb logger and name your wandb project
    wandb_logger = L.pytorch.loggers.WandbLogger(project='ssdm', name=kwargs.date[-4:]+ds+str(margin)+loss_mode)
    # Log things
    wandb_logger.experiment.config.update(dict(margin=margin, ds=ds))

    checkpoint_callback = ModelCheckpoint(
        monitor='net pick',  # The metric to monitor
        dirpath=f'checkpoints/{ds}{margin}{loss_mode}{branch_off}/',  # Directory to save the checkpoints
        filename='{epoch:02d}-{net pick:.4f}',  # Naming convention for checkpoints
        save_top_k=5,  # Save only the best model based on the monitored metric
        mode='max',  # Mode to determine if a lower or higher metric is better
        save_last=True  # Optionally save the last epoch model as well
    )

    trainer = L.pytorch.Trainer(
        num_sanity_val_steps = 0,
        max_epochs = 70,
        max_steps = 3000 * 1000,
        devices = 1,
        accelerator = "gpu",
        accumulate_grad_batches = 32,
        logger = wandb_logger,
        log_every_n_steps = 1,
        val_check_interval = 0.25,
        callbacks = [TQDMProgressBar(refresh_rate=500),
                     checkpoint_callback,
                     LearningRateMonitor(logging_interval='step', log_weight_decay=True),
                    ]
    )

    ds_module_dict = dict(jsd = ssdm.jsd, slm = ssdm.slm, hmx = ssdm.hmx, rwcpop = ssdm.rwcpop)
    dm = PairDataModule(ds_module=ds_module_dict[ds], perf_margin=margin)
    lit_net = LitMultiModel(
        training_loss_mode=loss_mode, 
        branch_off=branch_off,
        norm_cnn_grad=True,
        entropy_scale=0.05,
    )
    wandb_logger.watch(lit_net, log='all')
    trainer.fit(lit_net, datamodule=dm)
    wandb.finish()
    print('done without failure!')

