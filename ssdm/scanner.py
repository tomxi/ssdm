import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import ssdm

from tqdm import tqdm
import pandas as pd
import xarray as xr
import random

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
    
# class NLvlRegressionLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def pred2weight(self,
#             pred, sigmas=torch.tensor([0.5] * 16, device='cuda:0'), 
#             mus=torch.tensor(list(range(16)), device='cuda:0')
#         ):
#         deviations = -(pred - mus) ** 2 / sigmas ** 2
#         weight = torch.exp(deviations)
#         return weight / weight.sum()

#     def forward(self, pred, targets):
#         score_gap = (targets - targets.max()) ** 2
#         loss_weight = self.pred2weight(pred)
#         return torch.mean(torch.linalg.vecdot(loss_weight, score_gap))



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
# def train_epoch(ds_loader, net, criterion, optimizer, batch_size=8, lr_scheduler=None, device='cpu'):
#     running_loss = 0.
#     optimizer.zero_grad()
#     net.train()
#     for i, s in enumerate(ds_loader):
#         data = s['data'].to(device)

#         tau_hat = net(data)
#         label = s['label'].to(device)
#         loss = criterion(tau_hat, label)
#         loss.backward()
#         running_loss += loss
        
#         if i % batch_size == (batch_size - 1):
#             # take back prop step
#             optimizer.step()
#             optimizer.zero_grad()
#             if lr_scheduler is not None:
#                 lr_scheduler.step()
#     return (running_loss / len(ds_loader)).item()


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
        nl_loss = nlvl_loss(nlayer, s['layer_score'])
        
        if loss_type == 'multi':
            if s['label'] == 0:
                loss = u_loss + nl_loss * 0
            elif s['label'] == 1:
                loss = u_loss + nl_loss
        elif loss_type == 'util':
            loss = u_loss
        elif loss_type == 'nlvl':
            loss = nl_loss
            # Do I want to mask negative examples here as well?
            # if s['label'] == 0:
            #     loss = 0 * loss

        try:
            loss.backward()
        except RuntimeError:
            print('loss.backward() failed...')
            print(util.item())
            print(u_loss.item())
            print(s['info'])

        
        # For logging
        running_loss_util += u_loss.item()
        if s['label'] == 1:
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
# def net_eval(ds, net, criterion, device='cpu', verbose=False):
#     # ds_loader just need to be a iterable of samples
#     # make result DF
#     result_df = pd.DataFrame(columns=('tau_hat', 'pred', 'label', 'loss'))
#     ds_loader = DataLoader(ds, batch_size=None, shuffle=False)

#     net.to(device)
#     net.eval()
#     with torch.no_grad():
#         if verbose:
#             ds_loader = tqdm(ds_loader)
#         for s in ds_loader:
#             data = s['data'].to(device)
#             tau_hat = net(data)
#             label = s['label'].to(device)
#             loss = criterion(tau_hat, label)
#             if type(s['info']) is str:
#                 idx = s['info']
#             else:
#                 idx = '_'.join(s['info']) 
#             result_df.loc[idx] = (tau_hat.item(), int(tau_hat >= 0.5), label.item(), loss.item())      
#     return result_df.astype('float')


# eval tools:
def net_eval_multi_loss(ds, net, util_loss, nlvl_loss, device='cpu', verbose=False):
    # ds_loader just need to be a iterable of samples
    # make result DF
    result_df = pd.DataFrame(columns=('util', 'nlvl', 'u_loss', 'lvl_loss', 'label'))
    nlvl_output = pd.DataFrame(columns=[str(i) for i in range(16)])
    ds_loader = DataLoader(ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        if verbose:
            ds_loader = tqdm(ds_loader)
        for s in ds_loader:
            util, nlayer = net(s['data'])
            u_loss = util_loss(util, s['label'])
            nl_loss = nlvl_loss(nlayer, s['layer_score'])
            
            if type(s['info']) is str:
                idx = s['info']
            else:
                idx = '_'.join(s['info']) 
            nlvl_output.loc[idx] = nlayer.detach().cpu().numpy().squeeze()
            result_df.loc[idx] = (util.item(), nlayer.argmax().item(), u_loss.item(), nl_loss.item(),
                                  s['label'].item())
    return result_df.astype('float'), nlvl_output.astype('float')

# 
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
            tid, rep_feat, loc_feat = s['info'].split('_')
            data = s['data'].to(device)
            utility, nlvl = net(data)
            result_xr.loc[tid, rep_feat, loc_feat, 'util'] = utility.item()
            result_xr.loc[tid, rep_feat, loc_feat, 'nlvl'] =  nlvl.item()
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
        
    def forward(self, x):
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
        
    def forward(self, x):
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

    def forward(self, x):
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
    def __init__(self, output_bias=False):
        super().__init__()
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
            nn.Dropout(0.2),
            nn.Linear(6651, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 1, bias=output_bias), 
            nn.Sigmoid()
        ) 

        self.nlvl_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(106416, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 16, bias=output_bias), 
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
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


AVAL_MODELS = {
    'EvecSQNetC': EvecSQNetC,
    'EvecSQNetD': EvecSQNetD,
    'SQSmall': SQSmall,
    'NewSQSmall': NewSQSmall,
    'MultiRes': MultiRes,
}
