import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ssdm

from tqdm import tqdm
import pandas as pd
import xarray as xr
import random
import torchvision

import numpy as np

class NLvlLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        diviation = torch.abs(targets - targets.max())
        return torch.mean(torch.linalg.vecdot(predictions, diviation))
    

def perm_layer(samp):
    new_samp = samp.copy()
    perm_order = torch.randperm(16)
    new_samp['data'] = new_samp['data'].clone()[:, perm_order, :, :]
    new_samp['layer_score'] = new_samp['layer_score'].clone()[:, perm_order]
    new_samp['perm_order'] = perm_order
    if 'best_layer' in new_samp:
        del new_samp['best_layer']
    return new_samp

def quant_nlvl_target(samp, new_lvl_idx=[2, 4, 7, 15]):
    samp = samp.copy()
    samp['layer_score'] = samp['layer_score'][:, new_lvl_idx]
    return samp


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


def train_multi_loss(ds_loader, net, util_loss, nlvl_loss, optimizer, batch_size=8, lr_scheduler=None, device='cuda', loss_type='multi', pos_mask=True, entro_pen=0, verbose=False):
    running_loss_util = 0.
    running_loss_nlvl = 0.
    running_nlvl_loss_count = 0
    optimizer.zero_grad()
    net.to(device)
    net.train()

    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    for i, s in iterator:
        tid, repf, locf = s['info'].split('_')
        repf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(repf)).to(device)
        locf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(locf)).to(device)

        util, nlayer = net(s['data'].to(device), repf_idx, locf_idx)
        u_loss = util_loss(util, s['label'].to(device))
        nl_loss = nlvl_loss(nlayer, s['layer_score'].to(device))
        
        if loss_type == 'multi':
            if s['label'] == 0 and pos_mask:
                loss = u_loss + nl_loss * 0
            else:
                loss = u_loss + nl_loss
        elif loss_type == 'util':
            loss = u_loss
        elif loss_type == 'nlvl':
            loss = nl_loss
            # Do I want to mask negative examples here as well?
            # if s['label'] == 0:
            #     loss = 0 * loss

        if entro_pen != 0:
            logp = torch.log(nlayer)
            loss += torch.sum(nlayer * logp) * entro_pen
        try:
            loss.backward()
        except RuntimeError:
            print('loss.backward() failed...')
            print(util.item())
            print(u_loss.item())
            print(s['info'])

        # For logging
        running_loss_util += u_loss.item()
        if s['label'] == 1 or not pos_mask:
            running_loss_nlvl += nl_loss.item()
            running_nlvl_loss_count += 1
        
        # Manual batching
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    return (running_loss_util / len(ds_loader)), (running_loss_nlvl / running_nlvl_loss_count)

# eval tools:
def net_eval_multi_loss(ds, net, util_loss, nlvl_loss, device='cuda', num_workers=4, verbose=False):
    # ds_loader just need to be a iterable of samples
    # make result DF
    result_df = pd.DataFrame(columns=('util', 'nlvl', 'u_loss', 'lvl_loss', 'single_pick_lvl_loss', 'label'))
    nlvl_output = pd.DataFrame(columns=[str(i) for i in range(16)])
    ds_loader = DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)

    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            ds_loader = tqdm(ds_loader)
        for s in ds_loader:
            tid, repf, locf = s['info'].split('_')
            repf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(repf)).to(device)
            locf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(locf)).to(device)
            util, nlayer = net(s['data'].to(device), repf_idx, locf_idx)
            # util, nlayer = net(s['data'].to(device))
            u_loss = util_loss(util, s['label'].to(device))
            nl_loss = nlvl_loss(nlayer, s['layer_score'].to(device))

            sp_nlayer = torch.zeros((1,16)).to(device)
            max_index = nlayer.argmax().item()
            max_index_tuple = np.unravel_index(max_index, sp_nlayer.shape)
            sp_nlayer[max_index_tuple] = 1
            sp_nl_loss = nlvl_loss(sp_nlayer, s['layer_score'].to(device))
            
            if type(s['info']) is str:
                idx = s['info']
            else:
                idx = '_'.join(s['info']) 
            nlvl_output.loc[idx] = nlayer.detach().cpu().numpy().squeeze()
            result_df.loc[idx] = (util.item(), nlayer.argmax().item(), u_loss.item(), nl_loss.item(),
                                  sp_nl_loss.item(), s['label'].item())
    return result_df.astype('float'), nlvl_output.astype('float')

# 
def net_infer_multi_loss(infer_ds=None, net=None, device='cuda', num_workers=4, verbose=False): 
    full_tids = [infer_ds.name + tid for tid in infer_ds.tids]
    result_coords = dict(
        tid=full_tids, 
        rep_ftype=ssdm.AVAL_FEAT_TYPES, loc_ftype=ssdm.AVAL_FEAT_TYPES,
        est_type=['util', 'nlvl']
    )
    result_xr = xr.DataArray(np.nan, coords=result_coords, dims=result_coords.keys())
    
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)
    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            infer_loader = tqdm(infer_loader)
        for s in infer_loader:
            tid, rep_feat, loc_feat = s['info'].split('_')
            # repf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(rep_feat)).to(device)
            # locf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(loc_feat)).to(device)
            data = s['data'].to(device)
            try:
                utility, nlvl = net(data)
            except:
                print(s['info'])
                raise AssertionError
            result_xr.loc[infer_ds.name+tid, rep_feat, loc_feat, 'util'] = utility.item()
            result_xr.loc[infer_ds.name+tid, rep_feat, loc_feat, 'nlvl'] = nlvl.argmax().item()
    return result_xr.sortby('tid')


def net_infer_nlvl_only(infer_ds=None, net=None, device='cuda', num_workers=4, verbose=False): 
    full_tids = [infer_ds.name + tid for tid in infer_ds.tids]
    result_coords = dict(
        tid=full_tids, 
        rep_ftype=ssdm.AVAL_FEAT_TYPES, loc_ftype=ssdm.AVAL_FEAT_TYPES,
        layer=list(range(1, 17))
    )
    result_xr = xr.DataArray(np.nan, coords=result_coords, dims=result_coords.keys())
    
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)
    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            infer_loader = tqdm(infer_loader)
        for s in infer_loader:
            tid, rep_feat, loc_feat = s['info'].split('_')
            # repf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(rep_feat)).to(device)
            # locf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(loc_feat)).to(device)
            data = s['data'].to(device)
            try:
                utility, nlvl = net(data)
            except:
                print(s['info'])
                raise AssertionError
            result_xr.loc[infer_ds.name+tid, rep_feat, loc_feat] = nlvl.detach().cpu().numpy().squeeze()
    return result_xr.sortby('tid')


def net_infer_util_only(infer_ds=None, net=None, device='cuda', num_workers=4, verbose=False): 
    full_tids = [infer_ds.name + tid for tid in infer_ds.tids]
    result_coords = dict(
        tid=full_tids, 
        rep_ftype=ssdm.AVAL_FEAT_TYPES, loc_ftype=ssdm.AVAL_FEAT_TYPES,
    )
    result_xr = xr.DataArray(np.nan, coords=result_coords, dims=result_coords.keys())
    
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True)
    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            infer_loader = tqdm(infer_loader)
        for s in infer_loader:
            tid, rep_feat, loc_feat = s['info'].split('_')
            # repf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(rep_feat)).to(device)
            # locf_idx = torch.tensor(ssdm.AVAL_FEAT_TYPES.index(loc_feat)).to(device)
            data = s['data'].to(device)
            try:
                net_out = net(data)
            except:
                print(s['info'])
                raise AssertionError
            try:
                result_xr.loc[infer_ds.name+tid, rep_feat, loc_feat] = net_out.item()
            except:
                result_xr.loc[infer_ds.name+tid, rep_feat, loc_feat] = net_out[0].item()
    return result_xr.sortby('tid')


#####  MODELS ### MODELS #####       
## SWISH LAYER:
class TempSwish(nn.Module):
    def __init__(self):
        super(TempSwish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(1, dtype=torch.float))

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


class EvecSQNetC(nn.Module):
    def __init__(self, output_bias=False):
        super().__init__()
        self.activation = TempSwish
        self.dropout = nn.Dropout(0.2)
        self.expand_evecs = ExpandEvecs()
        self.adamaxpool = nn.AdaptiveMaxPool2d((96, 96))
    
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pre_util_conv = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
        )

        self.utility_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 6 * 6, 100, bias=False),
            self.activation(),
            nn.Linear(100, 1, bias=output_bias), 
            nn.Sigmoid()
        ) 

        self.pre_num_layer_conv = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
        )
        
        self.num_layer_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 6 * 6, 100, bias=False),
            self.activation(),
            nn.Linear(100, 16, bias=output_bias), 
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.adamaxpool(x)
        x = self.convlayers2(x)

        x_nlvl = self.pre_num_layer_conv(x)
        x_nlvl = torch.flatten(x_nlvl, 1)
        nlvl = self.num_layer_head(x_nlvl)

        x_util = self.pre_util_conv(x)
        x_util = torch.flatten(x_util, 1)
        util = self.utility_head(x_util)
        return util, nlvl


class EvecSQNetD(EvecSQNetC):
    def __init__(self):
        super().__init__()
        self.utility_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 6 * 6, 36, bias=False),
            self.activation(),
            nn.Linear(36, 1, bias=True), 
            nn.Sigmoid()
        ) 
        self.num_layer_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 6 * 6, 36, bias=False),
            self.activation(),
            nn.Linear(36, 1, bias=True), 
            nn.Softplus()
        )


class SQSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU
        self.dropout = nn.Dropout(0.2)
        self.expand_evecs = ExpandEvecs()
        self.adamaxpool = nn.AdaptiveMaxPool2d((81, 81))
    
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(16, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(), 
        )

        self.convlayers2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
        )

        self.convlayers3 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
            nn.Conv2d(25, 25, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(25, affine=True), self.activation(),
        )
        
        self.utility_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 3 * 3, 100, bias=False),
            self.activation(),
            nn.Linear(100, 1, bias=True), 
            nn.Sigmoid()
        ) 

        self.num_layer_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 3 * 3, 100, bias=False),
            self.activation(),
            nn.Linear(100, 1, bias=True), 
            nn.Softplus()
        )
        
    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.adamaxpool(x)
        x = self.convlayers2(x)
        x = x + self.convlayers3(x)

        x_flat = torch.flatten(x, 1)
        nlvl = self.num_layer_head(x_flat) + 1
        util = self.utility_head(x_flat)
        return util, nlvl


class NewSQSmall(SQSmall):
    def __init__(self):
        super().__init__()
        self.num_layer_head = nn.Sequential(
            self.dropout,
            nn.Linear(25 * 3 * 3, 16, bias=True),
            # self.activation(),
            # nn.Linear(100, 16, bias=True), 
            nn.Softmax(dim=-1)
        )

    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.convlayers1(x)
        x = self.adamaxpool(x)
        x = self.convlayers2(x)
        x = x + self.convlayers3(x)

        x_flat = torch.flatten(x, 1)
        nlvl = self.num_layer_head(x_flat)
        util = self.utility_head(x_flat)
        return util, nlvl


class MultiRes(nn.Module):
    def __init__(self, num_lvl=16):
        super().__init__()
        self.expand_evecs = ExpandEvecs()
        self.convlayers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(), 
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), nn.ReLU(), 
        )

        self.adapool_big = nn.AdaptiveMaxPool2d((81, 81))
        self.adapool_med = nn.AdaptiveMaxPool2d((9, 9))
        self.adapool_sm = nn.AdaptiveMaxPool2d((3, 3))

        self.util_head = nn.Sequential(
            nn.Linear(6651, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 1, bias=False), 
            # nn.Sigmoid()
        ) 

        self.nlvl_head = nn.Sequential(
            nn.Linear(106416, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, num_lvl, bias=True), 
            nn.Softmax(dim=-1)
        )


    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.convlayers(x)
        
        x_sm = self.adapool_sm(x)
        x_med = self.adapool_med(x)
        x_big = self.adapool_big(x)

        x_sm_ch_max = torch.max(x_sm, 1).values
        x_med_ch_max = torch.max(x_med, 1).values
        x_big_ch_max = torch.max(x_big, 1).values

        pre_util_head = torch.cat(
            [torch.flatten(x_sm_ch_max, 1), torch.flatten(x_med_ch_max, 1), torch.flatten(x_big_ch_max, 1)], 
            1
        )

        pre_nlvl_head = torch.cat(
            [torch.flatten(x_sm, 1), torch.flatten(x_med, 1), torch.flatten(x_big, 1)], 
            1
        )

        return self.util_head(pre_util_head), self.nlvl_head(pre_nlvl_head)


class MultiResSoftmax(MultiRes):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.convlayers(x)
        
        x_sm = self.adapool_sm(x)
        x_med = self.adapool_med(x)
        x_big = self.adapool_big(x)

        x_sm_ch_softmax = (x_sm.softmax(1) * x_sm).sum(1)
        x_med_ch_softmax = (x_med.softmax(1) * x_med).sum(1)
        x_big_ch_softmax = (x_big.softmax(1) * x_big).sum(1)

        pre_util_head = torch.cat(
            [torch.flatten(x_sm_ch_softmax, 1), torch.flatten(x_med_ch_softmax, 1), torch.flatten(x_big_ch_softmax, 1)], 
            1
        )

        pre_nlvl_head = torch.cat(
            [torch.flatten(x_sm, 1), torch.flatten(x_med, 1), torch.flatten(x_big, 1)], 
            1
        )

        return self.util_head(pre_util_head), self.nlvl_head(pre_nlvl_head)


class MultiResSoftmaxB(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand_evecs = ExpandEvecs()
        
        self.convlayers1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), TempSwish()
        )
        self.convlayers2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), TempSwish()
        )
        self.convlayers3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), TempSwish()
        )
        self.convlayers4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(32, affine=True), TempSwish(),
            nn.Conv2d(32, 16, kernel_size=5, padding='same', groups=16, bias=False), nn.InstanceNorm2d(16, affine=True), TempSwish()
        )

        self.maxpool = ConditionalMaxPool2d(kernel_size=2, stride=2)
        self.adapool_big = nn.AdaptiveMaxPool2d((36, 36))
        self.adapool_med = nn.AdaptiveMaxPool2d((9, 9))
        self.adapool_sm = nn.AdaptiveMaxPool2d((3, 3))

        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(1386, 128, bias=False),
            TempSwish(),
            nn.Linear(128, 1, bias=False)
        ) 

        self.nlvl_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(22176, 128, bias=False),
            TempSwish(),
            nn.Linear(128, 16, bias=True), 
            nn.Softmax(dim=-1)
        )


    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
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

        pre_util_head = torch.cat(
            [torch.flatten(x_sm_ch_softmax, 1), torch.flatten(x_med_ch_softmax, 1), torch.flatten(x_big_ch_softmax, 1)], 
            1
        )

        pre_nlvl_head = torch.cat(
            [torch.flatten(x_sm, 1), torch.flatten(x_med, 1), torch.flatten(x_big, 1)], 
            1
        )

        return self.util_head(pre_util_head), self.nlvl_head(pre_nlvl_head)


class MultiResSoftmaxUtil(nn.Module):
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

        # self.nlvl_head = nn.Sequential(
        #     nn.Dropout(0.2, inplace=False),
        #     nn.Linear(22176, 128, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(128, 16, bias=True), 
        #     nn.Softmax(dim=-1)
        # )


    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
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

        pre_util_head = torch.cat(
            [torch.flatten(x_sm_ch_softmax, 1), torch.flatten(x_med_ch_softmax, 1), torch.flatten(x_big_ch_softmax, 1)], 
            1
        )

        # pre_nlvl_head = torch.cat(
        #     [torch.flatten(x_sm, 1), torch.flatten(x_med, 1), torch.flatten(x_big, 1)], 
        #     1
        # )

        return self.util_head(pre_util_head)





# # Produced by chatgpt-4o
# def replace_batchnorm_with_instancenorm(model):
#     for name, module in model.named_children():
#         if isinstance(module, nn.BatchNorm2d):
#             num_features = module.num_features
#             new_module = nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True)
#             setattr(model, name, new_module)
#         else:
#             replace_batchnorm_with_instancenorm(module)


class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        effi_net = torchvision.models.efficientnet_b0()
        effi_net.features[0][0] = torch.nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.expand_evecs = ExpandEvecs()
        self.backbone = nn.Sequential(effi_net.features, effi_net.avgpool) # output size : 1280 * 1
        self.util_head = nn.Sequential(
            nn.Dropout(0.2, inplace=False),
            nn.Linear(1280, 1, bias=False),
        ) 
        # self.nlvl_head = nn.Sequential(
        #     nn.Dropout(0.2, inplace=False),
        #     nn.Linear(1280, 16, bias=False),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, x, rep_f_idx=None, loc_f_idx=None):
        x = self.expand_evecs(x)
        x = self.backbone(x)
        emb = torch.flatten(x, 1) # 1280 * 1
        return self.util_head(emb)#, self.nlvl_head(emb)


AVAL_MODELS = {
    'EvecSQNetC': EvecSQNetC,
    'EvecSQNetD': EvecSQNetD,
    'SQSmall': SQSmall,
    'NewSQSmall': NewSQSmall,
    'MultiRes': MultiRes,
    'MultiResSoftmaxUtil': MultiResSoftmaxUtil,
    'MultiResSoftmaxB': MultiResSoftmaxB,
    'EfficientNetB0': EfficientNetB0,
}
