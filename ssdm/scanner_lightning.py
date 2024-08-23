import xarray as xr
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
import argparse, itertools
from tqdm import tqdm

from ssdm.scanner import ExpandEvecs, ConditionalMaxPool2d
from ssdm import AVAL_FEAT_TYPES
import ssdm

# My LVL Loss
class NLvlLoss(torch.nn.Module):
    def __init__(self, scale_by_target=False):
        super().__init__()
        self.scale_by_target = scale_by_target

    def forward(self, pred, targets):
        diviation = targets.max() - targets
        losses = torch.linalg.vecdot(pred, diviation)
        if self.scale_by_target:
            losses = losses * targets.max()
        
        log_p = torch.log(pred + 1e-9)
        entropy = -torch.sum(pred * log_p, dim=-1)
        return torch.mean(losses), torch.mean(entropy)
    

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
        self.post_big_pool = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((32, 32))
        self.adapool_med = nn.AdaptiveMaxPool2d((16, 16))
        self.adapool_sm = nn.AdaptiveMaxPool2d((4, 4))

        

        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(528, 32, bias=False),
            nn.SiLU(),
            nn.Linear(32, 1, bias=False)
        )


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
        self.log('ranking loss', loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, on_epoch=False, batch_size=1, sync_dist=True)
        
        return loss


    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n],
                "weight_decay": 1,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(grouped_parameters)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=3e-6, max_lr=3e-3, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=4000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}]


    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')
        tb_writer = self.logger.experiment
        layout = {
            "metrics": {
                "learning rate": ["Multiline", ['learning rate']],
                "loss": ["Multiline", ['ranking loss']],
                # "train performance": [],
                "val performance": ["Multiline", ['net pick', 'boa pick', 'orc pick']],
            }
        }
        tb_writer.add_custom_scalars(layout)


    def on_validation_epoch_start(self):
        self.trainer.fit_loop.setup_data()
        self.val_ds = self.trainer.datamodule.val_dataloader().dataset
        full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
        )
        self.val_inference_result = xr.DataArray(np.nan, 
                                                 coords=result_coords, 
                                                 dims=result_coords.keys()
                                                ).sortby('tid')


    def validation_step(self, batch, batch_idx):
        tid, rep_feat, loc_feat = batch['info'].split('_')
        x = batch['data']
        self.val_inference_result.loc[self.val_ds.name+tid, rep_feat, loc_feat] = self(x).item()

    
    def on_validation_epoch_end(self):
        ds_score = self.val_ds.scores.max('layer').sortby('tid')
        net_feat_picks = self.val_inference_result.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])
        net_pick = ds_score.isel(net_feat_picks).mean().item()
        boa_pick = ds_score.isel(boa_feat_picks).mean().item()
        orc_pick = ds_score.isel(orc_feat_picks).mean().item()
        self.log('net pick', net_pick, sync_dist=True)
        self.log('boa pick', boa_pick, sync_dist=True)
        self.log('orc pick', orc_pick, sync_dist=True)
        # print('net, boa, orc:', net_pick, boa_pick, orc_pick)
        

class LitMultiModel(L.LightningModule):
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
        self.post_big_pool = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((32, 32))
        self.adapool_med = nn.AdaptiveMaxPool2d((16, 16))
        self.adapool_sm = nn.AdaptiveMaxPool2d((4, 4)) 

        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(528, 32, bias=False),
            nn.SiLU(),
            nn.Linear(32, 1, bias=False)
        )
        self.nlvl_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(528 * 16, 16, bias=False),
            nn.SiLU(),
            nn.Linear(16, 16, bias=True),
            nn.Softmax(dim=-1)
        )
        self.lvl_loss_fn = NLvlLoss()


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

        mutlires_embeddings = [torch.flatten(x_sm_ch_softmax, 1), 
                               torch.flatten(x_med_ch_softmax, 1), 
                               torch.flatten(x_big_ch_softmax, 1)]
        
        mutlires_embeddings_per_layer = [torch.flatten(x_sm, 1), 
                                         torch.flatten(x_med, 1), 
                                         torch.flatten(x_big, 1)]

        pre_util_head = torch.cat(mutlires_embeddings, 1)
        pre_nlvl_head = torch.cat(mutlires_embeddings_per_layer, 1)
        return self.util_head(pre_util_head), self.nlvl_head(pre_nlvl_head)

    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')
        tb_writer = self.logger.experiment
        layout = {
            "metrics": {
                "learning rate": ["Multiline", ['learning rate']],
                "loss": ["Multiline", ['ranking loss', 'nlvl loss', 'entropy']],
                "val performance": ["Multiline", ['net pick', 'boa pick', 'orc pick']],
            }
        }
        tb_writer.add_custom_scalars(layout)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.cat([batch['x1'], batch['x2']], 0)
        contrast_label = batch['perf_gap'].squeeze().sign()
        
        util_est, nlvl_est = self(x)
        ranking_loss = nn.functional.margin_ranking_loss(
            util_est[0].squeeze(), util_est[1].squeeze(), 
            contrast_label, margin=1
        )

        if contrast_label.item() == 1:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[0].squeeze(), batch['x1_layer_score'].squeeze())
        else:
            lvl_loss, entropy = self.lvl_loss_fn(nlvl_est[1].squeeze(), batch['x2_layer_score'].squeeze()) 
        scaled_entropy = 0.01 * entropy

        # Logging to TensorBoard (if installed) by default
        self.log('ranking loss', ranking_loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('nlvl loss', lvl_loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('entropy', scaled_entropy, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning rate', lr, on_step=True, on_epoch=False, batch_size=1, sync_dist=True)
        
        return ranking_loss + lvl_loss - scaled_entropy

    def on_validation_epoch_start(self):
        self.trainer.fit_loop.setup_data()
        self.val_ds = self.trainer.datamodule.val_dataloader().dataset
        full_tids = [self.val_ds.name + tid for tid in self.val_ds.tids]
        result_coords = dict(
            tid=full_tids, 
            rep_ftype=AVAL_FEAT_TYPES, loc_ftype=AVAL_FEAT_TYPES,
        )
        self.val_inference_result = xr.DataArray(np.nan, 
                                                 coords=result_coords, 
                                                 dims=result_coords.keys()
                                                ).sortby('tid')
    
    def validation_step(self, batch, batch_idx):
        tid, rep_feat, loc_feat = batch['info'].split('_')
        x, _ = self(batch['data'])
        self.val_inference_result.loc[self.val_ds.name+tid, rep_feat, loc_feat] = x.item()

    def on_validation_epoch_end(self):
        ds_score = self.val_ds.scores.max('layer').sortby('tid')
        net_feat_picks = self.val_inference_result.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        boa_feat_picks = ds_score.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])
        net_pick = ds_score.isel(net_feat_picks).mean().item()
        boa_pick = ds_score.isel(boa_feat_picks).mean().item()
        orc_pick = ds_score.isel(orc_feat_picks).mean().item()
        self.log('net pick', net_pick, sync_dist=True)
        self.log('boa pick', boa_pick, sync_dist=True)
        self.log('orc pick', orc_pick, sync_dist=True)
        # print('net, boa, orc:', net_pick, boa_pick, orc_pick)


    def configure_optimizers(self):
        grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if 'bias' not in n],
                "weight_decay": 10,
            },
            {
                "params": [p for n, p in self.named_parameters() if 'bias' in n],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(grouped_parameters)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=1e-6, max_lr=3e-3, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=3000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}]



# New validation dataloaders...
class PairDataModule(L.LightningDataModule):
    def __init__(self, ds_module=ssdm.rwcpop, perf_margin=0.05, val_split='val'):
        super().__init__()
        self.perf_margin = perf_margin
        self.ds_module = ds_module
        self.val_split = val_split
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('margin', help='sampling score margin')
    parser.add_argument('num_device')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    margin = float(kwargs.margin) # 0.05

    model_id, ds = list(itertools.product(
        ['MultiResSoftmaxUtil'],
        # ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-jsd', 'all-but-rwcpop', 'all-but-slm', 'all-but-hmx'],
        ['jsd', 'rwcpop', 'slm', 'hmx'], # Change this to a list of ds_moodules
        # ['sgd', 'adamw'],
    ))[int(kwargs.config_idx)]

    tb_logger = L.pytorch.loggers.TensorBoardLogger(save_dir="/scratch/qx244/pl_tb_logs/", name=kwargs.date+ds)

    checkpoint_callback = ModelCheckpoint(
        monitor='net pick',  # The metric to monitor
        dirpath=f'checkpoints/{ds}/',  # Directory to save the checkpoints
        filename='{epoch:02d}-{net pick:.4f}',  # Naming convention for checkpoints
        save_top_k=3,  # Save only the best model based on the monitored metric
        mode='max',  # Mode to determine if a lower or higher metric is better
        save_last=True  # Optionally save the last epoch model as well
    )

    trainer = L.pytorch.Trainer(
        num_sanity_val_steps = -1,
        max_epochs = int(kwargs.total_epoch),
        devices = int(kwargs.num_device),
        accelerator = "gpu",
        accumulate_grad_batches = 16,
        logger = tb_logger,
        log_every_n_steps = 250,
        callbacks = [TQDMProgressBar(refresh_rate=1000),
                     checkpoint_callback]
    )

    ds_module_dict = dict(jsd = ssdm.jsd, slm = ssdm.slm, hmx = ssdm.hmx, rwcpop = ssdm.rwcpop)
    dm = PairDataModule(ds_module=ds_module_dict[ds], perf_margin=margin)
    lit_net = LitMultiModel()

    trainer.fit(lit_net, datamodule=dm)
    print('done without failure!')

