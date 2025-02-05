import ssdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
import wandb
import timm
import argparse
from typing import List

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor

DEFAULT_DS_MODULES = [ssdm.rwcpop, ssdm.jsd, ssdm.slm, ssdm.hmx]

class PairDataModule(L.LightningDataModule):
    def __init__(self, ds_modules=DEFAULT_DS_MODULES, perf_margin=0.05, train_split='train', val_split='val', predict_split='val', loaders=4):
        super().__init__()
        self.perf_margin = perf_margin
        self.ds_modules = ds_modules
        self.train_split = train_split
        self.val_split = val_split
        self.predict_split = predict_split
        self.loaders = loaders


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_ds_list = [ds_module.PairDS(split=self.train_split, perf_margin=self.perf_margin) for ds_module in self.ds_modules]
            self.val_ds_list = [ds_module.PairDS(split=self.val_split, perf_margin=self.perf_margin) for ds_module in self.ds_modules]
            self.train_ds = ConcatDataset(self.train_ds_list)
            self.val_ds = ConcatDataset(self.val_ds_list)

        if stage == 'validate':
            self.val_ds_list = [ds_module.PairDS(split=self.val_split, perf_margin=self.perf_margin) for ds_module in self.ds_modules]
            self.val_ds = ConcatDataset(self.val_ds_list)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds_list = [ds_module.InferDS(split='test') for ds_module in self.ds_modules]
            self.test_infer_ds = ConcatDataset(self.test_ds_list)

        if stage == "predict":
            self.predict_ds_list = [ds_module.InferDS(split=self.predict_split) for ds_module in self.ds_modules]
            self.predict_ds = ConcatDataset(self.predict_ds_list)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, shuffle=True,
            num_workers=self.loaders, pin_memory=True,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=None, shuffle=False,
            num_workers=self.loaders, pin_memory=True,
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_infer_ds, batch_size=None, shuffle=False,
            num_workers=self.loaders, pin_memory=True,
        )


    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, batch_size=None, shuffle=False,
            num_workers=self.loaders, pin_memory=True,
        )
    

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


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((24, 24))
        self.adapool_med = nn.AdaptiveMaxPool2d((12, 12))
        self.adapool_sm = nn.AdaptiveMaxPool2d((6, 6))
        self.Conv2d = InputSizeAwareConv2d

        self.convlayers1 = nn.Sequential(
            self.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers2 = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers3 = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers4 = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )
        self.convlayers5 = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU()
        )

        

        self.post_big_pool = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.maxpool,
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
            self.maxpool,
        )
        self.post_med_pool = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
            self.maxpool,
        )
        self.post_sm_pool = nn.Sequential(
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.SiLU(),
            self.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.SiLU(),
        )

        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(108, 32, bias=False),
            nn.SiLU(),
            nn.Linear(32, 1, bias=False)
        )
 
    def forward(self, x):
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

        return self.util_head(multires_embeddings)


class LitModule(L.LightningModule):
    def __init__(self, model='cnn'):
        super().__init__()
        self.expand_evecs = ExpandEvecs()

        if model == 'cnn':
            self.backbone = CNNModel()
        elif model == 'swin':
            self.backbone = timm.create_model(
                'swinv2_cr_tiny_224.untrained',
                pretrained=False,
                in_chans=16,
                num_classes=1,
                strict_img_size=False
            )
        self.loss_fn = nn.MarginRankingLoss(margin=1)
        self.save_hyperparameters()
    
    def forward(self, x):
        x = self.expand_evecs(x)
        return self.backbone(x)


    def on_fit_start(self):
        torch.set_float32_matmul_precision('medium')


    def training_step(self, batch, batch_idx):
        x = torch.cat([batch['x1'], batch['x2']], 0)
        y = batch['perf_gap'].sign()
        
        util_est = self(x)
        ranking_loss = self.loss_fn(util_est[0, None], util_est[1, None], y)
        self.log('train ranking loss', ranking_loss, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        return ranking_loss


    def validation_step(self, batch, batch_idx):
        x = torch.cat([batch['x1'], batch['x2']], 0)
        y = batch['perf_gap'].sign()
        
        util_est = self(x)
        ranking_loss = self.loss_fn(util_est[0, None], util_est[1, None], y)
        self.log('val ranking loss', ranking_loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        return ranking_loss


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
            base_lr=1e-6, max_lr=5e-3, 
            cycle_momentum=False, mode='triangular2', 
            step_size_up=18000
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 2}]


def train_model(loaders=4, wb_run_name=None, debug=False, ds_modules=DEFAULT_DS_MODULES, perf_margin=0.05, model='cnn'):
    net = LitModule(model=model)
    dm = PairDataModule(ds_modules=ds_modules, perf_margin=perf_margin,loaders=loaders)
    
    wb_logger = L.pytorch.loggers.WandbLogger(project='ssdm3', name=wb_run_name)
    wb_logger.watch(net, log='all')

    trainer = L.Trainer(
        max_epochs=20,
        accumulate_grad_batches=8, 
        logger=wb_logger, 
        callbacks=[
            TQDMProgressBar(refresh_rate=1000),
            LearningRateMonitor(logging_interval='step'), 
            ModelCheckpoint(dirpath=f'ckpts/{model}/', filename='{epoch}-{val ranking loss:.2f}',
                            monitor='val ranking loss', save_top_k=3, mode='min', save_last=True)
        ]
    )
    
    if debug:
        return trainer, net, dm

    trainer.fit(net, dm)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSDM model')
    parser.add_argument('--ds', type=str, default='hybrid',
                       help='Dataset modules to use')
    parser.add_argument('--name', type=str, default='test',
                       help='WandB run name')
    parser.add_argument('--perf_margin', type=float, default=0.05,
                       help='Performance margin')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'swin'], 
                       help='Model architecture')

    args = parser.parse_args()

    # Map dataset names to modules
    ds_map = {
        'rwcpop': [ssdm.rwcpop],
        'jsd': [ssdm.jsd],
        'slm': [ssdm.slm],
        'hmx': [ssdm.hmx],
        'hybrid': [ssdm.rwcpop, ssdm.jsd, ssdm.slm, ssdm.hmx]
    }

    train_model(
        wb_run_name='-'.join([args.name, args.model, args.ds]),
        ds_modules=ds_map[args.ds],
        perf_margin=args.perf_margin,
        model=args.model
    )
