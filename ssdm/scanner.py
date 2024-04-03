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

        if tau == 'rep' or tau == 'both':
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


def train_multi_loss(ds_loader, net, util_loss, nlvl_loss, optimizer, batch_size=8, lr_scheduler=None, device='cpu', loss_type='multi'):
    running_loss_util = 0.
    running_loss_nlvl = 0.
    optimizer.zero_grad()
    net.to(device)
    net.train()
    for i, s in enumerate(ds_loader):
        util, nlayer = net(s['data'])
        u_loss = util_loss(util, s['label'])
        nl_loss = nlvl_loss(nlayer, s['uniq_segs'])
        
        if loss_type == 'multi':
            loss = u_loss + (nl_loss / 10)
        elif loss_type == 'util':
            loss = u_loss
        elif loss_type == 'nlvl':
            loss = nl_loss
        loss.backward()
        
        running_loss_util += u_loss.item()
        running_loss_nlvl += nl_loss.item()
        
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    return (running_loss_util / len(ds_loader)), (running_loss_nlvl / len(ds_loader))

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


# eval tools:
def net_eval_multi_loss(ds, net, util_loss, nlvl_loss, device='cpu', verbose=False):
    # ds_loader just need to be a iterable of samples
    # make result DF
    result_df = pd.DataFrame(columns=('util', 'nlvl', 'u_loss', 'lvl_loss', 'loss', 'label', 'uniq_seg'))
    ds_loader = DataLoader(ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            ds_loader = tqdm(ds_loader)
        for s in ds_loader:
            util, nlayer = net(s['data'])
            u_loss = util_loss(util, s['label'])

            nl_loss = nlvl_loss(nlayer, s['uniq_segs'])
            best_nlvl = torch.argmax(nlayer)
            result_df.loc['_'.join(s['info'])] = (util.item(), best_nlvl.item(), u_loss.item(), nl_loss.item(),
                                                  u_loss.item() + nl_loss.item(), s['label'].item(), s['uniq_segs'].item())      
    return result_df.astype('float')


def net_infer(infer_ds, net, device='cpu', out_type='pd'):
    # out_type can be 'pd' or 'xr'
    assert infer_ds.infer # make sure it's the right mode
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
        return xr_da.sortby('tid')
    

# Need fixing
def net_infer_multi_loss(infer_ds=None, net=None, device='cpu', out_type='xr'):
    if out_type != 'xr':
        assert NotImplementedError
        # out_type can be 'xr'
    assert infer_ds.infer # make sure it's the right mode
    assert infer_ds.mode == 'both'

    result_coords = dict(
        tid=infer_ds.tids, 
        rep_ftype=ssdm.AVAL_FEAT_TYPES, loc_ftype=ssdm.AVAL_FEAT_TYPES,
        est_type=['util', 'nlvl']
    )
    result_xr = xr.DataArray(np.nan, coords=result_coords, dims=result_coords.keys())
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False)
    net.to(device)
    net.eval()
    with torch.no_grad():
        for s in tqdm(infer_loader):
            tid, rep_feat, loc_feat = s['info'][:3]
            data = s['data'].to(device)
            utility, nlvl = net(data)
            result_xr.loc[tid, rep_feat, loc_feat, 'util'] = utility.item()
            result_xr.loc[tid, rep_feat, loc_feat, 'nlvl'] =  torch.argmax(nlvl).item()
    return result_xr.sortby('tid')


#####  MODELS ### MODELS #####       
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


class ExpandEvecs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, evecs):
        with torch.set_grad_enabled(False):
            lras = []
            for lvl in range(evecs.shape[-1]):
                first_evecs = evecs[:, :, :, :lvl + 1]
                lras.append(torch.matmul(first_evecs, first_evecs.transpose(-1, -2)))

            cube = torch.cat(lras, dim=1)
        return cube


class EvecSQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))
        self.dropout = nn.Dropout(0.15)
        self.expand_evecs = ExpandEvecs()
    
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(20, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
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


        self.pre_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 72),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(72, 1), 
            nn.Sigmoid()
        ) 

        self.num_layer_head = nn.Linear(72, 11)
        
    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.pre_predictor(x)
        x = self.dropout(x)
        return self.utility_head(x), self.num_layer_head(x)


class EvecNetMulti2(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.15)
        self.activation = TempSwish

        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Conv2d(12, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        self.maxpool = nn.AdaptiveMaxPool2d((216, 20))

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Conv2d(12, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.pre_predictor = nn.Sequential(
            nn.Linear(24 * 6 * 5, 64),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(64, 1), 
            nn.Sigmoid()
        ) 

        self.num_layer_head = nn.Linear(64, 11)
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)
        x = self.convlayers3(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.dropout(x)
        x = self.pre_predictor(x)
        x = self.dropout(x)
        return self.utility_head(x), self.num_layer_head(x)
    

class EvecSQNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.maxpool = nn.AdaptiveMaxPool2d((216, 216))
        self.dropout = nn.Dropout(0.15)
        self.expand_evecs = ExpandEvecs()
    
        self.convlayers1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(9, 5, 5), padding='same'), nn.InstanceNorm2d(20, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(20, 6, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(6, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(6, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(), 
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

        self.convlayers4 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(36, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.pre_predictor = nn.Sequential(
            nn.Linear(36 * 6 * 6, 72),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(72, 1), 
            nn.Sigmoid()
        ) 

        self.num_layer_head = nn.Linear(72, 11)
        
    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        x = self.convlayers2(x)
        x = self.dropout(x)

        y = self.convlayers4(x)
        y = self.dropout(y)

        x = self.convlayers3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.pre_predictor(x)
        x = self.dropout(x)

        y = torch.flatten(y, 1)
        y = self.pre_predictor(y)
        y = self.dropout(y)
        return self.utility_head(x), self.num_layer_head(y)


class EvecSQNet3(EvecSQNet2):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, padding='same', groups=20), nn.InstanceNorm2d(20, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(20, 6, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(6, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(6, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(), 
        )

AVAL_MODELS = {
    'RepModel': RepModel,
    'RepModel2': RepModel2,
    'RepSD3': RepSD3,
    'RepNet': RepNet,
    'LocNet': LocNet,
    'EvecNetMulti2': EvecNetMulti2,
    'EvecSQNet': EvecSQNet,
    'EvecSQNet2': EvecSQNet2,
    'EvecSQNet3': EvecSQNet3,
}
