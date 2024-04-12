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


def train_multi_loss(ds_loader, net, util_loss, nlvl_loss, optimizer, batch_size=8, lr_scheduler=None, device='cpu', loss_type='multi', verbose=False):
    running_loss_util = 0.
    running_loss_nlvl = 0.
    running_nlvl_loss_count = 0
    optimizer.zero_grad()
    net.to(device)
    net.train()

    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator)

    for i, s in iterator:
        util, nlayer = net(s['data'])
        u_loss = util_loss(util, s['label'])
        nl_loss = nlvl_loss(nlayer, s['best_layer'])
        
        if loss_type == 'multi':
            if s['label'] == 0:
                loss = u_loss
            elif s['label'] == 1:
                loss = u_loss + nl_loss/10
        elif loss_type == 'util':
            loss = u_loss
        elif loss_type == 'nlvl':
            loss = nl_loss
        loss.backward()
        
        # For logging
        running_loss_util += u_loss.item()
        if s['label'] == 1:
            running_loss_nlvl += nl_loss.item()
            running_nlvl_loss_count += 1
        
        # Manual batching
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    return (running_loss_util / len(ds_loader)), (running_loss_nlvl / running_nlvl_loss_count)

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
    result_df = pd.DataFrame(columns=('util', 'nlvl', 'u_loss', 'lvl_loss', 'weighted_total_loss', 'label', 'best_layer'))
    ds_loader = DataLoader(ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            ds_loader = tqdm(ds_loader)
        for s in ds_loader:
            util, nlayer = net(s['data'])
            u_loss = util_loss(util, s['label'])

            nl_loss = nlvl_loss(nlayer, s['best_layer'])
            best_nlvl = torch.floor(nlayer)
            result_df.loc['_'.join(s['info'])] = (util.item(), best_nlvl.item(), u_loss.item(), nl_loss.item(),
                                                  u_loss.item() + nl_loss.item()/10, s['label'].item(), s['best_layer'].item())      
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
            result_xr.loc[tid, rep_feat, loc_feat, 'nlvl'] =  torch.floor(nlvl).item()
    return result_xr.sortby('tid')


#####  MODELS ### MODELS #####       
## SWISH LAYER:
class TempSwish(nn.Module):
    def __init__(self):
        super(TempSwish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(0, dtype=torch.float))

    def forward(self,x):
        x = x * nn.functional.sigmoid(self.beta * x)
        return x

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
    

# class MyMaxPool2d(nn.Module):
#     # Only apply maxpool when time dimension bigger than pool_thresh
#     def __init__(self, pool_thresh=17, **kwargs):
#         super().__init__()
#         self.maxpool = nn.MaxPool2d(**kwargs)
#         self.pool_thresh=pool_thresh
    
#     def forward(self, x):
#         if x.shape[-2] > self.pool_thresh:
#             return self.maxpool(x)
#         else:
#             return x


class EvecSQNet_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.dropout = nn.Dropout(0.15)
        self.expand_evecs = ExpandEvecs()
    
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=1024, kernel_size=2, stride=2),
            nn.Conv2d(16, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=512, kernel_size=2, stride=2),
            nn.Conv2d(12, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(), 
            MyMaxPool2d(pool_thresh=256, kernel_size=2, stride=2),
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=128, kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=64, kernel_size=2, stride=2),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=32, kernel_size=2, stride=2),
            nn.Conv2d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=16, kernel_size=2, stride=2),
            nn.Conv2d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.AdaptiveMaxPool2d((6, 6)),
        )

        self.utility_head = nn.Sequential(
            nn.Linear(24 * 6 * 6, 36, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(36, 1, bias=True), 
            nn.Sigmoid()
        ) 

        self.pre_num_layer_conv = nn.Sequential(
            # Goes after convlayers2
            nn.Conv2d(24, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=16, kernel_size=2, stride=2),
            nn.Conv2d(12, 6, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(6, eps=0.01), self.activation(),
            nn.AdaptiveMaxPool2d((16, 16)),
        )
        
        self.num_layer_head = nn.Sequential(
            nn.Linear(6 * 16 * 16, 36, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(36, 1, bias=True),
            nn.Softplus()
        )
        
    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.convlayers2(x)
        x = self.dropout(x)

        x_nlvl = self.pre_num_layer_conv(x)
        x_nlvl = torch.flatten(x_nlvl, 1)
        x_nlvl = self.dropout(x_nlvl)
        nlvl = self.num_layer_head(x_nlvl)

        x = self.convlayers3(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        util = self.utility_head(x)

        return util, nlvl


class EvecSQNetC(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device=torch.device('mps')
        else:
            self.device='cpu'

        self.activation = nn.ReLU
        self.dropout = nn.Dropout(0.15)
        self.expand_evecs = ExpandEvecs()
        
    
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pre_util_conv = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
        )

        self.utility_head = nn.Sequential(
            nn.Linear(25 * 6 * 6, 900, bias=False),
            self.activation(),
            nn.Linear(900, 1, bias=True), 
            nn.Sigmoid()
        ) 

        self.pre_num_layer_conv = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25), self.activation(),
        )
        
        self.num_layer_head = nn.Sequential(
            nn.Linear(25 * 6 * 6, 900, bias=False),
            self.activation(),
            nn.Linear(900, 1, bias=True),
            nn.Softplus()
        )

        self.to(self.device)
        self.adamaxpool = nn.AdaptiveMaxPool2d((96, 96)).to('cpu')
        
    def forward(self, x):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.dropout(x)

        x = self.adamaxpool(x.to('cpu'))
        x = self.convlayers2(x.to(self.device))
        x = self.dropout(x)

        x_nlvl = self.pre_num_layer_conv(x)
        x_nlvl = self.dropout(torch.flatten(x_nlvl, 1))
        nlvl = self.num_layer_head(x_nlvl) + 1

        x_util = self.pre_util_conv(x)
        x_util = self.dropout(torch.flatten(x_util, 1))
        util = self.utility_head(x_util)

        return util, nlvl


class EvecNetMulti3(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.15)
        self.activation = TempSwish

        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=1024, kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=512, kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=256, kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=128, kernel_size=(2, 1), stride=(2, 1)),
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=64, kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=32, kernel_size=(2, 1), stride=(2, 1)),
        )

        self.conv_util = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=16, kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=8, kernel_size=2, stride=2),
            nn.Conv2d(24, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.AdaptiveMaxPool2d((4, 10))
        )
        
        self.util_embed = nn.Sequential(
            nn.Linear(24 * 4 * 10, 32),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(32, 1), 
            nn.Sigmoid()
        ) 

        self.conv_nlvl = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, padding='same'), nn.InstanceNorm2d(8, eps=0.01), self.activation(),
            MyMaxPool2d(pool_thresh=16, kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(8, 4, kernel_size=5, padding='same'), nn.InstanceNorm2d(4, eps=0.01), self.activation(),
            nn.AdaptiveMaxPool2d((8, 20)),
            nn.Conv2d(4, 4, kernel_size=(8, 1), padding='valid'), nn.InstanceNorm2d(4, eps=0.01), self.activation(), # Full height
        )

        self.nlvl_embed = nn.Sequential(
            nn.Linear(4 * 1 * 20, 20),
            nn.ReLU(inplace=True)
        )

        self.num_layer_head = nn.Sequential(nn.Linear(20, 1), nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.convlayers2(x)
        x = self.dropout(x)

        x_nlvl = self.conv_nlvl(x)
        x_nlvl = torch.flatten(x_nlvl, 1)
        # x_nlvl = self.dropout(x_nlvl)
        x_nlvl = self.nlvl_embed(x_nlvl)
        x_nlvl = self.dropout(x_nlvl)
        nlvl = self.num_layer_head(x_nlvl)

        x_util = self.conv_util(x)
        x_util = torch.flatten(x_util, 1)
        # x_util = self.dropout(x_util)
        x_util = self.util_embed(x_util)
        x_util = self.dropout(x_util)
        util = self.utility_head(x_util)

        return util, nlvl


class EvecNetMulti2(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.15)
        self.activation = TempSwish

        self.convlayers1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 12, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.maxpool = nn.AdaptiveMaxPool2d((54, 20))

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(12, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.conv_util = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.util_embed = nn.Sequential(
            nn.Linear(24 * 3 * 5, 32),
            nn.ReLU(inplace=True)
        )

        self.utility_head = nn.Sequential(
            nn.Linear(32, 1), 
            nn.Sigmoid()
        ) 

        self.conv_nlvl = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, padding='same'), nn.InstanceNorm2d(8, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(8, 4, kernel_size=5, padding='same'), nn.InstanceNorm2d(4, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(4, 4, kernel_size=(3, 1), padding='valid'), nn.InstanceNorm2d(4, eps=0.01), self.activation(), # Full height
        )

        self.nlvl_embed = nn.Sequential(
            nn.Linear(4 * 1 * 20, 20),
            nn.ReLU(inplace=True)
        )

        self.num_layer_head = nn.Sequential(
            nn.Linear(20, 1, bias=True),
        )
        
    def forward(self, x):
        x = self.convlayers1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.convlayers2(x)
        x = self.dropout(x)

        x_nlvl = self.conv_nlvl(x)
        x_nlvl = torch.flatten(x_nlvl, 1)
        # x_nlvl = self.dropout(x_nlvl)
        x_nlvl = self.nlvl_embed(x_nlvl)
        x_nlvl = self.dropout(x_nlvl)
        nlvl = self.num_layer_head(x_nlvl)

        x_util = self.conv_util(x)
        x_util = torch.flatten(x_util, 1)
        # x_util = self.dropout(x_util)
        x_util = self.util_embed(x_util)
        x_util = self.dropout(x_util)
        util = self.utility_head(x_util)

        return util, nlvl
    

class EvecSQNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = TempSwish
        self.maxpool = nn.AdaptiveMaxPool2d((108, 108))
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
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.convlayers4 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(24, eps=0.01), self.activation(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 16, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.utility_head = nn.Sequential(
            nn.Linear(16 * 6 * 6, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 1), 
            nn.Sigmoid()
        ) 

        self.num_layer_head = nn.Sequential(
            nn.Linear(16 * 6 * 6, 36),
            nn.ReLU(inplace=True),
            nn.Linear(36, 1), 
            nn.Softplus()
        ) 
        
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

        x = torch.flatten(x, 1)
        y = torch.flatten(y, 1)
        return self.utility_head(x), self.num_layer_head(y)


class EvecSQNet3(EvecSQNet2):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', groups=16), nn.InstanceNorm2d(16), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 6, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(6), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(6, 12, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(12), self.activation(), 
        )


AVAL_MODELS = {
    'EvecNetMulti2': EvecNetMulti2,
    'EvecNetMulti3': EvecNetMulti3,
    'EvecSQNetC': EvecSQNetC,
    'EvecSQNet2': EvecSQNet2,
    'EvecSQNet3': EvecSQNet3,
}
