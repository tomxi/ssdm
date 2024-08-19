import os
import xarray as xr
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, ConcatDataset, Sampler

import lightning as L

from ssdm.scanner import ExpandEvecs, ConditionalMaxPool2d
from ssdm import AVAL_FEAT_TYPES
import ssdm

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitUtilModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.expand_evecs = ExpandEvecs()
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )
        self.convlayers2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )
        self.convlayers3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )
        self.convlayers4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((36, 36))
        self.adapool_med = nn.AdaptiveMaxPool2d((9, 9))
        self.adapool_sm = nn.AdaptiveMaxPool2d((3, 3))

        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(1386, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 1, bias=False)
        )


    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x1 = self.convlayers2(self.maxpool(x))
        x2 = self.convlayers3(self.maxpool(x1))
        x3 = self.convlayers3(self.maxpool(x2))
        
        x_sm = self.adapool_sm(x1) + self.adapool_sm(x2) + self.adapool_sm(x3)
        x_med = self.adapool_med(x1) + self.adapool_med(x2) + self.adapool_med(x3)
        x_big = self.adapool_big(x1) + self.adapool_big(x2) + self.adapool_big(x3)

        x_sm_ch_softmax = (x_sm.softmax(1) * x_sm).sum(1)
        x_med_ch_softmax = (x_med.softmax(1) * x_med).sum(1)
        x_big_ch_softmax = (x_big.softmax(1) * x_big).sum(1)

        mutlires_embeddings = [torch.flatten(x_sm_ch_softmax, 1), 
                               torch.flatten(x_med_ch_softmax, 1), 
                               torch.flatten(x_big_ch_softmax, 1)]

        pre_util_head = torch.cat(mutlires_embeddings, 1)
        return self.util_head(pre_util_head)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.cat([batch['x1'], batch['x2']], 0)
        contrast_label = batch['perf_gap'].sign()
        
        util_est = self(x)
        loss = nn.functional.margin_ranking_loss(
            util_est[0, None], util_est[1, None], 
            contrast_label, margin=1
        )

        # Logging to TensorBoard (if installed) by default
        self.log('ranking loss', loss, on_step=False, on_epoch=True, batch_size=1)
        # self.log('ranking loss offset', loss + 0.1, on_step=False, on_epoch=True, batch_size=1)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning rate', lr, on_step=False, on_epoch=True, batch_size=1)
        
        return loss


    # def configure_optimizers_adamw(self):
    #     grouped_parameters = [
    #         {
    #             "params": [p for n, p in self.named_parameters() if 'bias' not in n],
    #             "weight_decay": 1,
    #         },
    #         {
    #             "params": [p for n, p in self.named_parameters() if 'bias' in n],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = optim.AdamW(grouped_parameters)
    #     lr_scheduler = optim.lr_scheduler.CyclicLR(
    #         optimizer, 
    #         base_lr=1e-5, max_lr=1e-2, 
    #         cycle_momentum=False, mode='triangular2', 
    #         step_size_up=2048 * 3
    #     )
    #     return [optimizer], [lr_scheduler]


    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n],
                "weight_decay": 0.001,	
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.SGD(grouped_parameters, lr=2e-2)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # goes down by ~1k every 135 steps
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}]


    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')
        tb_writer = self.logger.experiment
        layout = {
            "metrics": {
                "loss": ["Multiline", ['ranking loss']],
                "learning rate": ["Multiline", ['learning rate']],
                "val performance": ["Multiline", ['net pick', 'boa pick', 'orc pick']],
            }
        }
        tb_writer.add_custom_scalars(layout)


    def on_validation_epoch_start(self):
        self.trainer.fit_loop.setup_data()
        # self.val_ds = self.trainer.val_dataloader().dataset
        self.val_ds = self.trainer.datamodule.val_infer_ds
        full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
        )
        self.val_inference_result = xr.DataArray(np.nan, coords=result_coords, dims=result_coords.keys()).sortby('tid')


    def validation_step(self, batch, batch_idx):
        tid, rep_feat, loc_feat = batch['info'].split('_')
        x = batch['data']
        self.val_inference_result.loc[self.val_ds.name+tid, rep_feat, loc_feat] = self(x).item()

    
    def on_validation_epoch_end(self):
        ds_score = self.val_ds.scores.max('layer').sortby('tid')
        net_feat_picks = self.val_inference_result.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])

        self.log('net pick', ds_score.isel(net_feat_picks).mean().item())
        self.log('boa pick', ds_score.isel(boa_feat_picks).mean().item())
        self.log('orc pick', ds_score.isel(orc_feat_picks).mean().item())




# New validation dataloaders...
class PairDataModule(L.LightningDataModule):
    def __init__(self, ds_module=ssdm.rwcpop, perf_margin=0.05):
        super().__init__()
        self.perf_margin = perf_margin
        self.ds_module = ds_module
    
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds = self.ds_module.PairDSLmeasure(split='train', perf_margin=self.perf_margin) 
            # self.val_ds = self.ds_module.PairDSLmeasure(split='val', perf_margin=self.perf_margin) 
            self.val_infer_ds = self.ds_module.InferDS(split='val')

        if stage == 'validate':
            self.val_infer_ds = self.ds_module.InferDS(split='val')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_infer_ds = self.ds_module.InferDS(split='test')

        if stage == "predict":
            self.full_ds = self.ds_module.InferDS(split=None) 

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, shuffle=True,
            num_workers=4, pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_infer_ds, batch_size=None, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_infer_ds, batch_size=None, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.full_ds, batch_size=None, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    

class DevPairDataModule(PairDataModule):
    def __init__(self, perf_margin=0.001, split='single'):
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

