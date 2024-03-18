import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ssdm

from tqdm import tqdm
import pandas as pd
import xarray as xr
import random

import numpy as np

# TiME MASK DATA AUG on sample, adapted from SpecAugment 
def time_mask(sample, T=40, num_masks=1, replace_with_zero=False, tau='rep'):
    rec_mat = sample['data']

    cloned = rec_mat.clone()
    len_rec_mat = cloned.shape[2]
    
    for _ in range(0, num_masks):
        t = min(random.randrange(0, T), len_rec_mat - 1)
        t_zero = random.randrange(0, len_rec_mat - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): break

        mask_end = random.randrange(t_zero, t_zero + t)

        if tau == 'rep':
            if (replace_with_zero): 
                cloned[0, 0][t_zero:mask_end, :] = 0
                cloned[0, 0][:, t_zero:mask_end] = 0
            else: 
                mat_mean = cloned.mean()
                cloned[0, 0][t_zero:mask_end, :] = mat_mean
                cloned[0, 0][:, t_zero:mask_end] = mat_mean
        elif tau == 'loc':
            if (replace_with_zero): cloned[0, 0][t_zero:mask_end] = 0
            else: cloned[0, 0][t_zero:mask_end] = cloned.mean()

    sample['data'] = cloned
    return sample


# Training loops:
# returns average epoch loss.
def train_epoch(ds_loader, net, criterion, optimizer, batch_size=8, lr_scheduler=None, device='cpu'):
    running_loss = 0.
    optimizer.zero_grad()
    net.train()
    for i, s in enumerate(ds_loader):
        data = s['data'].to(device)

        tau_hat = net(data)
        label = s['label'].to(device)
        loss = criterion(tau_hat, label)
        loss.backward()
        running_loss += loss
        
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    return (running_loss / len(ds_loader)).item()


# eval tools:
def net_eval(ds, net, criterion, device='cpu', verbose=False):
    # ds_loader just need to be a iterable of samples
    # make result DF
    result_df = pd.DataFrame(columns=('tau_hat', 'pred', 'label', 'loss'))
    ds_loader = DataLoader(ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            ds_loader = tqdm(ds_loader)
        for s in ds_loader:
            data = s['data'].to(device)
            tau_hat = net(data)
            label = s['label'].to(device)
            loss = criterion(tau_hat, label)
            result_df.loc['_'.join(s['info'])] = (tau_hat.item(), int(tau_hat >= 0.5), label.item(), loss.item())      
    return result_df.astype('float')


def net_infer(infer_ds, net, device='cpu', out_type='pd'):
    # out_type can be 'pd' or 'xr'
    result_df = pd.DataFrame(columns=(ssdm.AVAL_FEAT_TYPES), index=infer_ds.tids)
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        for s in tqdm(infer_loader):
            tid, feat = s['info'][:2]
            data = s['data'].to(device)
            tau_hat = net(data)
            result_df.loc[tid][feat] = tau_hat.item()

    if out_type == 'pd':
        return result_df.astype('float')
    
    elif out_type == 'xr':
        xr_da = xr.DataArray(
            result_df.sort_index(), 
            dims=['tid', 'f_type']
        )
        return xr_da


#####  MODELS ### MODELS #####       
class SmallRepOnly(nn.Module):
    def __init__(self):
        super(SmallRepOnly, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.maxpool = nn.AdaptiveMaxPool2d((7, 7))
        
        self.rep_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=False), nn.Sigmoid()
        )  
        
    def forward(self, x):
        x = self.convlayers(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


class LocOnly(nn.Module):
    def __init__(self):
        super(LocOnly, self).__init__()
        self.loc_conv_l1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=13, padding='same', bias=False), 
            nn.InstanceNorm1d(6, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.loc_conv_l2 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, padding='same', bias=False),
            nn.InstanceNorm1d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, padding='same', bias=False),
            nn.InstanceNorm1d(32, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool1d(7)

        self.loc_predictor = nn.Sequential(
            nn.Linear(32 * 7, 1, bias=True), nn.Sigmoid()
        )
        
    def forward(self, band):
        loc_emb_l1 = self.loc_conv_l1(band)
        loc_emb_l2 = self.loc_conv_l2(loc_emb_l1)
        loc_emb = self.loc_pool(loc_emb_l2)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))

        return loc_prob


class LocModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = TempSwish
        self.dropout = nn.Dropout(0.2)

        self.activation = nn.ReLU

        self.maxpool = nn.AdaptiveMaxPool1d(216)
        
        self.convlayers1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(8, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(12, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(12, eps=0.01), self.activation(), 
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv1d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(16, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(24, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(24, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(24, 36, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(36, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.rep_predictor = nn.Sequential(
            nn.Linear(36 * 6, 36),
            self.dropout, self.activation(),
            nn.Linear(36, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r
    

class TinyGPool(nn.Module):
    def __init__(self):
        super(TinyGPool, self).__init__()
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), nn.ReLU(inplace=True), 
        )

        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.convlayers3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(36, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )        
        self.rep_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 1), 
            nn.Sigmoid()
        )  
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.maxpool(x)
        x = self.convlayers2(x)
        x = self.convlayers3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


class RepDropout(TinyGPool):
    def __init__(self):
        super(RepDropout, self).__init__()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


class RepLessDropout(RepDropout):
    def __init__(self):
        super(RepLessDropout, self).__init__()

    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)
        # x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


class RepLessDropout2(RepLessDropout):
    def __init__(self):
        super(RepLessDropout2, self).__init__()
        self.dropout = nn.Dropout(0.15)


class RepDropout2(RepDropout):
    def __init__(self):
        super(RepDropout2, self).__init__()
        self.dropout = nn.Dropout(0.15)


class RepLessDropout2(RepLessDropout):
    def __init__(self):
        super(RepLessDropout2, self).__init__()
        self.dropout = nn.Dropout(0.15)


class RepLessDropout1(RepLessDropout):
    def __init__(self):
        super(RepLessDropout1, self).__init__()
        self.dropout = nn.Dropout(0.1)


class RepModel(nn.Module):
    def __init__(self):
        super(RepModel, self).__init__()

        self.activation = nn.ReLU

        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))
        
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(), 
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(36, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.rep_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 36),
            self.activation(),
            nn.Linear(36, 1), 
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


## SWISH LAYER:
class TempSwish(nn.Module):
    def __init__(self):
        super(TempSwish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(0, dtype=torch.float))

    def forward(self,x):
        x = x * nn.functional.sigmoid(self.beta * x)
        return x


class RepModelSwish(RepModel):
    def __init__(self):
        super(RepModelSwish, self).__init__()
        self.activation = TempSwish


class RepSwishDo2(RepModel):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.dropout = nn.Dropout(0.2)


class RepSD3(RepModel):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.dropout = nn.Dropout(0.2)

        self.rep_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 36),
            self.dropout, self.activation(),
            nn.Linear(36, 1), 
            nn.Sigmoid()
        )


class RepModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = TempSwish
        
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(), 
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(36, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.rep_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 72),
            self.activation(),
            nn.Dropout(0.25),
            nn.Linear(72, 1), 
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.convlayers1(x)
        x = nn.Dropout(0.15)(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = nn.Dropout(0.15)(x)

        x = self.convlayers3(x)
        x = nn.Dropout(0.15)(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        r = self.rep_predictor(x)
        return r


class RepNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.activation = TempSwish
        self.dropout = nn.Dropout(0.2)
        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))
        
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.rep_predictor = nn.Sequential(
            nn.Linear(24 * 6 * 6, 24),
            self.dropout,
            self.activation(),
            nn.Linear(24, 1), 
            nn.Sigmoid()
        )

          
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.dropout(x)
        r = self.rep_predictor(x)
        return r


class LocNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = TempSwish
        self.dropout = nn.Dropout(0.15)

        self.maxpool = nn.AdaptiveMaxPool1d(216)
        
        self.convlayers1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(8, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(8, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(12, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # nn.Conv1d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(12, eps=0.01), self.activation(), 
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv1d(12, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(16, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(16, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(24, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm1d(24, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm1d(24, eps=0.01), self.activation(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.rep_predictor = nn.Sequential(
            nn.Linear(24 * 6, 24),
            self.dropout, 
            self.activation(),
            nn.Linear(24, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.dropout(x)
        r = self.rep_predictor(x)
        return r


AVAL_MODELS = {
    'SmallRepOnly': SmallRepOnly,
    'LocOnly': LocOnly,
    # 'TinyRep': TinyRep,
    # 'GentlePool': GentlePool,
    # 'Gentle1Pool': Gentle1Pool,
    'TinyGPool': TinyGPool,
    # 'RepDropout': RepDropout,
    'RepDropout2': RepDropout2, 
    'RepModel': RepModel,
    'LocModel': LocModel,
    'RepModel2': RepModel2,
    'RepModelSwish': RepModelSwish,
    'RepLessDropout2': RepLessDropout2, # **
    'RepSwishDo2': RepSwishDo2,
    'RepSD3': RepSD3,
    'RepNet': RepNet,
    'LocNet': LocNet,
}

## https://github.com/scipy/scipy/blob/v1.11.3/scipy/sparse/csgraph/_laplacian.py#L524