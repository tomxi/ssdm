import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools

# import librosa
import ssdm
from ssdm import harmonix as hmx
from scipy import stats

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr

class SlmDS(Dataset):
    """ split='train',
        mode='rep', # {'rep', 'loc'}
        infer=False,
        drop_features=[],
        precomputed_tau_fp = '/home/qx244/scanning-ssm/ssdm/taus_1107.nc'
    """
    def __init__(self, 
                 split='train',
                 mode='rep', # {'rep', 'loc'}
                 infer=False,
                 drop_features=[],
                 precomputed_tau_fp = '/home/qx244/scanning-ssm/ssdm/taus_1107.nc',
                ):
        if mode not in ('rep', 'loc'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.split = split
        # load precomputed taus, and drop feature and select tau type
        taus_full = xr.open_dataarray(precomputed_tau_fp)
        self.tau_scores = taus_full.drop_sel(f_type=drop_features).sel(tau_type=mode)

        # Get the threshold for upper and lower quartiles, 
        # and use them as positive and negative traning examples respectively
        tau_percentile_flat = stats.percentileofscore(self.tau_scores.values.flatten(), self.tau_scores.values.flatten())
        self.tau_percentile = self.tau_scores.copy(data=tau_percentile_flat.reshape(self.tau_scores.shape))
        self.tau_thresh = [np.percentile(self.tau_scores, 25), np.percentile(self.tau_scores, 75)]
        
        tau_series = self.tau_scores.to_series()
        split_ids = ssdm.get_ids(split, out_type='set')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.infer = infer
        if infer:
            # just get all the combos, returns the percentile of the sample tau value [0~1]
            self.samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in tau_series.index.to_flat_index().values \
                            if pair[0] in split_ids}
        else:
            # use threshold to prepare training data
            # calculate threshold from all taus
            neg_tau_series = tau_series[tau_series < self.tau_thresh[0]]
            self.all_neg_samples = neg_tau_series.index.to_flat_index().values
            pos_tau_series = tau_series[tau_series > self.tau_thresh[1]]
            self.all_pos_samples = pos_tau_series.index.to_flat_index().values

            # use set to select only the ones in the split
            neg_samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in self.all_neg_samples if pair[0] in split_ids}
            pos_samples = {pair: self.tau_percentile.sel(tid=pair[0], f_type=pair[1]).item() / 100 \
                            for pair in self.all_pos_samples if pair[0] in split_ids}

            self.samples = pos_samples.copy()
            self.samples.update(neg_samples)
        self.ordered_keys = list(self.samples.keys())
        self.tids = list(split_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, feat = self.ordered_keys[idx]
        track = ssdm.Track(tid)
        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        if self.mode == 'rep':
            rep_ssm = track.ssm(feature=feat, 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feat]
                                )
            tau_percent = self.samples[(tid, feat)]
            return {'data': torch.tensor(rep_ssm[None, None, :], dtype=torch.float32, device=self.device),
                    'label': torch.tensor([tau_percent > 0.5], dtype=torch.float32, device=self.device)[None, :],
                    'tau_percent': torch.tensor(tau_percent, dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
        
        elif self.mode == 'loc':
            path_sim = track.path_sim(feature=feat, 
                                      distance=config['loc_metric'],
                                      **ssdm.LOC_FEAT_CONFIG[feat])

            loc_label = self.samples[(tid, feat)]
            return {'data': torch.tensor(path_sim[None, None, :], dtype=torch.float32, device=self.device),
                    'label': torch.tensor([loc_label], dtype=torch.float32)[None, :],
                    'tau_percent': torch.tensor(tau_percent, dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
       
        else:
            assert KeyError


class HmxDS(Dataset):
    """ 
    mode='rep', # {'rep', 'loc'}
    """
    def __init__(self, mode='rep'):
        if mode not in ('rep', 'loc'):
            raise AssertionError('bad dataset mode, can only be rep or loc')
        self.mode = mode
        self.tids = ssdm.get_ids('harmonix', out_type='list')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.samples = list(itertools.product(self.tids, ssdm.AVAL_FEAT_TYPES))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tid, feat = self.samples[idx]
        track = hmx.Track(tid)

        config = ssdm.DEFAULT_LSD_CONFIG.copy()

        if self.mode == 'rep':
            rep_ssm = track.ssm(feature=feat, 
                                distance=config['rep_metric'],
                                width=config['rec_width'],
                                full=config['rec_full'],
                                **ssdm.REP_FEAT_CONFIG[feat]
                                )

            return {'data': torch.tensor(rep_ssm[None, None, :], dtype=torch.float32, device=self.device),
                    'info': (tid, feat, self.mode),
                    }
        
        elif self.mode == 'loc':
            path_sim = track.path_sim(feature=feat, 
                                      distance=config['loc_metric'],
                                      **ssdm.LOC_FEAT_CONFIG[feat])

            return {'data': torch.tensor(path_sim[None, None, :], dtype=torch.float32, device=self.device),
                    'info': ('harmonix', tid, feat, self.mode),
                    }
       
        else:
            assert KeyError


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
    result_df = pd.DataFrame(columns=('tau_hat', 'pred', 'tau_percent', 'label', 'loss'))
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
            result_df.loc['_'.join(s['info'])] = (tau_hat.item(), int(tau_hat >= 0.5), s['tau_percent'].item(), label.item(), loss.item())      
    return result_df.astype('float')


def net_infer(infer_ds, net, device='cpu'):
    result_df = pd.DataFrame(columns=(ssdm.AVAL_FEAT_TYPES), index=infer_ds.tids)
    infer_loader = DataLoader(infer_ds, batch_size=None, shuffle=False)

    net.to(device)
    net.eval()
    with torch.no_grad():
        for s in tqdm(infer_loader):
            tid, feat, _= s['info']
            data = s['data'].to(device)
            tau_hat = net(data)
            result_df.loc[tid][feat] = tau_hat.item()
    return result_df.astype('float')

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
            nn.Conv1d(1, 4, kernel_size=13, padding='same', bias=False), 
            nn.InstanceNorm1d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.loc_conv_l2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=7, padding='same', bias=False),
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 8, kernel_size=5, padding='same', bias=False),
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool1d(7)

        self.loc_predictor = nn.Sequential(
            nn.Linear(8 * 7, 1, bias=True), nn.Sigmoid()
        )
        
    def forward(self, band):
        loc_emb_l1 = self.loc_conv_l1(band)
        loc_emb_l2 = self.loc_conv_l2(loc_emb_l1)
        loc_emb = self.loc_pool(loc_emb_l2)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))

        return loc_prob


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


AVAL_MODELS = {
    'SmallRepOnly': SmallRepOnly,
    'LocOnly': LocOnly,
    # 'TinyRep': TinyRep,
    # 'GentlePool': GentlePool,
    # 'Gentle1Pool': Gentle1Pool,
    'TinyGPool': TinyGPool,
    # 'RepDropout': RepDropout,
    'RepDropout2': RepDropout2,
    # 'RepLessDropout': RepLessDropout,
    'RepLessDropout2': RepLessDropout2,
}

## https://github.com/scipy/scipy/blob/v1.11.3/scipy/sparse/csgraph/_laplacian.py#L524