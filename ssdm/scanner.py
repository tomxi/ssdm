import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from scipy.spatial.distance import pdist, squareform

from ssdm import salami
import ssdm

import librosa

from tqdm import tqdm
import numpy as np
import pandas as pd


# some helper functions from Brian
def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


def trieye_like(tensor, band=1):
    e = eye_like(tensor)
    # set above and below diagonals to 1
    for k in range(1, band + 1):
        torch.diagonal(e, offset=k).fill_(1)
        torch.diagonal(e, offset=-k).fill_(1)
    return e

        

class SalamiDataset(Dataset):
    def __init__(self, 
                 track_ids,
                 loc_label_path='/home/qx244/scanning-ssm/revive/loc_label.pkl',
                 rep_label_path='/home/qx244/scanning-ssm/revive/rep_label.pkl',
                 mode='rep' # {'rep', 'both', 'loc'}
                ):
        self.track_ids = list(track_ids)
        self.track_ids.sort()
        # self.device = device
        # get labels from pkl file
        self.loc_df = pd.read_pickle(loc_label_path)
        self.rep_df = pd.read_pickle(rep_label_path)

        self.mode=mode
        
        # build
        if mode == 'both':
            self.rl_labels = {}
            for tid in self.track_ids:
                for feat in ssdm.AVAL_FEAT_TYPES:
                    loc_label = self.loc_df.loc[tid][feat]
                    rep_label = self.rep_df.loc[tid][feat]
                    if rep_label**2 + loc_label**2 > 0 :
                        self.rl_labels[f'{tid}_{feat}'] = (np.asarray(rep_label), np.asarray(loc_label))
            # store keys as a list
            self.rl_samples = list(self.rl_labels.keys())


        elif mode == 'rep':
            # build self.labels: {tid_feat: 1/0}
            rep_sample_position = self.rep_df.to_numpy().nonzero()
            self.rep_labels = {}
            for x, y in zip(rep_sample_position[0], rep_sample_position[1]):
                tid = self.rep_df.index[x]
                # only collect if there it's in track_ids
                if tid in self.track_ids:    
                    feat = self.rep_df.columns[y]
                    self.rep_labels[f'{tid}_{feat}'] = self.rep_df.loc[tid][feat]
            # store keys as a list
            self.rep_samples = list(self.rep_labels.keys())
            
        elif mode == 'loc':
            # build self.labels: {tid_feat: 1/0}
            loc_sample_position = self.loc_df.to_numpy().nonzero()
            self.loc_labels = {}
            for x, y in zip(loc_sample_position[0], loc_sample_position[1]):
                tid = self.loc_df.index[x]
                # only collect if there it's in track_ids
                if tid in self.track_ids:    
                    feat = self.loc_df.columns[y]
                    self.loc_labels[f'{tid}_{feat}'] = self.loc_df.loc[tid][feat]
            # store keys as a list
            self.loc_samples = list(self.loc_labels.keys())


    def __len__(self):
        if self.mode == 'both':
            return len(self.rl_samples)
        elif self.mode == 'rep':
            return len(self.rep_samples)
        elif self.mode == 'loc':
            return len(self.loc_samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'rep':
            sample_key = self.rep_samples[idx]
        if self.mode == 'loc':
            sample_key = self.loc_samples[idx]
        elif self.mode == 'both':
            sample_key = self.rl_samples[idx]
        sdm_path = f'/scratch/qx244/data/salami/sdms/{sample_key}_cosine.npy'
        with open(sdm_path, 'rb') as f:
            dmat = np.load(f, allow_pickle=True)

        if self.mode == 'rep':
            rep_label = (self.rep_labels[sample_key] + 1) * 0.5 # rep label was -1, 0, or 1,  now 0, 0.5 ,1
            rep_label = np.array([rep_label])
            datum = {'dmat': torch.tensor(dmat[None, None, :], dtype=torch.float32), 
                     'rep_label':torch.tensor(rep_label[None, :], dtype=torch.float32),
                     'info': sample_key,
                    }
        if self.mode == 'loc':
            utri_dmat = np.triu(dmat)
            band = librosa.util.shear(utri_dmat, factor=-1, axis=0)[:, 1:14].T
            loc_label = (self.loc_labels[sample_key] + 1) * 0.5 # loc label was -1, 0, or 1,  now 0, 0.5 ,1
            loc_label = np.array([loc_label])
            datum = {'band': torch.tensor(band[None, :], dtype=torch.float32),
                     'loc_label':torch.tensor(loc_label[None, :], dtype=torch.float32),
                     'info': sample_key,
                    }
        elif self.mode == 'both':
            rep_label, loc_label = (self.rl_labels[sample_key])
            rep_label = np.array([(rep_label + 1) * 0.5])
            loc_label = np.array([(loc_label + 1) * 0.5])
            # shear pull and zero pad
            utri_dmat = np.triu(dmat)
            band = librosa.util.shear(utri_dmat, factor=-1, axis=0)[:, 1:14].T
            datum = {'dmat': torch.tensor(dmat[None, None, :], dtype=torch.float32),
                     'band': torch.tensor(band[None, None, :], dtype=torch.float32),
                     'rep_label':torch.tensor(rep_label[None, :], dtype=torch.float32),
                     'loc_label':torch.tensor(loc_label[None, :], dtype=torch.float32),
                     'info': sample_key,
                    }   
        return datum


class NewSalamiDS(SalamiDataset):
    def __init__(self, 
                 track_ids,
                 mode='both'
                ):
        super().__init__(track_ids, mode=mode)

    def __getitem__(self, idx):
        if self.mode == 'both':
            sample_key = self.rl_samples[idx]
        elif self.mode == 'loc' or self.mode == 'loc1':
            sample_key = self.loc_samples[idx]
        sdm_path = f'/scratch/qx244/data/salami/sdms/{sample_key}_cosine.npy'
        with open(sdm_path, 'rb') as f:
            dmat = np.load(f, allow_pickle=True)
        # mask with band:
        dmat_tensor = torch.tensor(dmat, dtype=torch.float32)
        band = dmat_tensor * trieye_like(dmat_tensor, band=13)

        if self.mode == 'both':
            rep_label, loc_label = (self.rl_labels[sample_key])
            # convert from -1, 0, 1 to 0, 0.5, 1
            rep_label = np.array([(rep_label + 1) * 0.5])
            loc_label = np.array([(loc_label + 1) * 0.5])
            datum = {'dmat': torch.tensor(dmat[None, None, :], dtype=torch.float32),
                     'band': band[None, None, :].to(torch.float32),
                     'rep_label':torch.tensor(rep_label[None, :], dtype=torch.float32),
                     'loc_label':torch.tensor(loc_label[None, :], dtype=torch.float32),
                     'info': sample_key,
                    }   

        elif self.mode == 'loc':
            loc_label = (self.loc_labels[sample_key])[0]
            loc_label = np.array([(loc_label + 1) * 0.5])

            datum = {'band': band[None, None, :].to(torch.float32),
                     'loc_label':torch.tensor(loc_label[None, :], dtype=torch.float32),
                     'info': sample_key,
                    }

        elif self.mode == 'loc1':
            loc_label = (self.loc_labels[sample_key])[0]
            loc_label = np.array([(loc_label + 1) * 0.5])

            datum = {'band': band[None, None, :].to(torch.float32),
                     'loc_label':torch.tensor(loc_label[None, None, :], dtype=torch.float32),
                     'info': sample_key,
                    }
        return datum


# add logging for lr_sched
def train_epoch(ds_loader, net, criterion, optimizer, batch_size=8, lr_scheduler=None, device='cpu'):
    running_loss = 0.
    optimizer.zero_grad()
    net.train()
    for i, s in enumerate(ds_loader):
        dmat = s['dmat'].to(device)

        out_rep = net(dmat)
        rep_label = s['rep_label'].to(device)
        loss = criterion(out_rep, rep_label)
        loss.backward()
        running_loss += loss
        
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    return running_loss / len(ds_loader)


def train_epoch_loc(ds_loader, net, criterion, optimizer, batch_size=8, lr_scheduler=None, writer=None, device='cpu', writer_step=0):
    running_loss = 0.
    optimizer.zero_grad()
    net.train()
    for i, s in enumerate(ds_loader):
        band = s['band'].to(device)

        out_loc = net(band)
        loc_label = s['loc_label'].to(device)
        loss = criterion(out_loc, loc_label)
        loss.backward()
        running_loss += loss
        
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            writer_step += 1
            if lr_scheduler is not None:
                lr_scheduler.step()  
            writer.add_scalar('lr',
                              lr_scheduler.get_last_lr()[0],
                              writer_step
                              )
    return running_loss / len(ds_loader), writer_step


# now with band input
def train_epoch_duo(ds_loader, net, criterion, optimizer, batch_size=8, lr_scheduler=None, writer=None, device='cpu', writer_step=0):
    past_running_loss = 0.
    running_loss = 0.

    net.train()
    optimizer.zero_grad()
    for i, s in enumerate(ds_loader):
        dmat = s['dmat'].to(device)
        band = s['band'].to(device)
        rep_label = s['rep_label'].to(device)
        loc_label = s['loc_label'].to(device)

        out_rep, out_loc = net(dmat, band)
        r_mask = (rep_label.clone().detach() != 0.5).to(torch.float32)
        l_mask = (loc_label.clone().detach() != 0.5).to(torch.float32)
        rep_loss = criterion(out_rep, rep_label) * r_mask
        loc_loss = criterion(out_loc, loc_label) * l_mask
        loss = (rep_loss + loc_loss) / (r_mask + l_mask) # divide by sum of the mask
        loss.backward()
        running_loss += loss.item()
        
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = (running_loss - past_running_loss) / batch_size
            past_running_loss = running_loss
            writer_step += 1
            if lr_scheduler is not None:
                lr_scheduler.step()
                writer.add_scalar('Batch Loss',
                                   batch_loss,
                                   writer_step)
                writer.add_scalar('lr',
                                   lr_scheduler.get_last_lr()[0],
                                   writer_step)
    return running_loss / len(ds_loader), writer_step


# Generate DF instead
def net_eval_df(ds_loader, net, criterion, mode='rep', device='cpu'):
    # make result DF
    result_df = pd.DataFrame(columns=('pred_r', 'label_r', 'pred_l', 'label_l', 'loss'))

    net.eval()
    with torch.no_grad():
        for s in ds_loader:
            if mode == 'rep':
                dmat = s['dmat'].to(device)
                rep_label = s['rep_label'].to(device)

                out_rep = net(dmat)
                vloss = criterion(out_rep, rep_label)
                result_df.loc[s['info']] = (out_rep.item(), rep_label.item(), 0, 0, vloss.item())
            elif mode == 'loc':
                band = s['band'].to(device)
                loc_label = s['loc_label'].to(device)

                out_loc = net(band)
                vloss = criterion(out_loc, loc_label)
                result_df.loc[s['info']] = (out_loc.item(), loc_label.item(), 0, 0, vloss.item())
            elif mode == 'both':
                dmat = s['dmat'].to(device)
                band = s['band'].to(device)
                rep_label = s['rep_label'].to(device)
                loc_label = s['loc_label'].to(device)

                out_rep, out_loc = net(dmat, band)
                if out_loc.shape == torch.Size([1]):
                    print(s['info'])
                    out_loc = out_loc[None, :]
                if out_rep.shape == torch.Size([1]):
                    print(s['info'])
                    out_rep = out_rep[None, :]
                r_mask = int(rep_label != 0.5)
                l_mask = int(loc_label != 0.5)
                rep_loss = criterion(out_rep, rep_label) * r_mask
                loc_loss = criterion(out_loc, loc_label) * l_mask
                vloss = (rep_loss + loc_loss) / (r_mask + l_mask) # divide by sum of the mask
                result_df.loc[s['info']] = (out_rep.item(), rep_label.item(), out_loc.item(), loc_label.item(), vloss.item())
    return result_df

def process_duo_eval_df(result_df):
    pred_r = result_df.loc[result_df['label_r'] != 0.5]['pred_r'].round()
    label_r = result_df.loc[result_df['label_r'] != 0.5]['label_r']

    rep_accu = sum(label_r == pred_r) / len(pred_r)

    pred_l = result_df.loc[result_df['label_l'] != 0.5]['pred_l'].round()
    label_l = result_df.loc[result_df['label_l'] != 0.5]['label_l']

    loc_accu = sum(label_l == pred_l) / len(pred_l)

    loss = sum(result_df['loss']) / len(result_df['loss'])

    return {'rep_accu': rep_accu, 'loc_accu': loc_accu, 'loss': loss}


# Legacy
def evaluate_net(val_loader, net, criterion, mode='rep', device='cpu'):
    net.eval()
    with torch.no_grad():
        running_vloss = 0.0
        for i, s in enumerate(val_loader):
            if mode == 'rep':
                out_rep = net(s['dmat'].to(device))
                vloss = criterion(out_rep, s['rep_label'].to(device))
            elif mode == 'loc':
                out_loc = net(s['band'].to(device))
                vloss = criterion(out_loc, s['loc_label'].to(device))
            elif mode == 'both':
                dmat = s['dmat'].to(device)
                band = s['band'].to(device)
                rep_label = s['rep_label'].to(device)
                loc_label = s['loc_label'].to(device)

                out_rep, out_loc = net(dmat, band)
                if out_loc.shape == torch.Size([1]):
                    out_loc = out_loc[None, :]
                if out_rep.shape == torch.Size([1]):
                    out_rep = out_rep[None, :]
                r_mask = int(rep_label != 0.5)
                l_mask = int(loc_label != 0.5)
                rep_loss = criterion(out_rep, rep_label) * r_mask
                loc_loss = criterion(out_loc, loc_label) * l_mask
                vloss = (rep_loss + loc_loss) / (r_mask + l_mask) # divide by sum of the mask
            running_vloss += vloss.item()

    return running_vloss / len(val_loader)


# Depre after fixing evaluate_net
def net_accuracy(dataloader, net, mode='rep', device='cpu'):
    net.eval()
    with torch.no_grad():
        num_rep_correct = 0
        num_rep_total = 0
        num_loc_correct = 0
        num_loc_total = 0
        for i, s in enumerate(dataloader):
            
            if mode == 'rep':
                rep_label = s['rep_label'].to(device)
                rep_pred = int(net(s['dmat'].to(device)) > 0.5)
            elif mode =='loc':
                loc_label = s['loc_label'].to(device)
                loc_pred = int(net(s['band'].to(device)) > 0.5)

            elif mode == 'both':
                dmat = s['dmat'].to(device)
                band = s['band'].to(device)
                rep_label = s['rep_label'].to(device)
                loc_label = s['loc_label'].to(device)

                rep_prob, loc_prob = net(dmat, band)
                rep_pred = int(rep_prob > 0.5)
                loc_pred = int(loc_prob > 0.5)
            
            if mode == 'rep' or mode == 'both':
                if rep_label == 0.5:
                    # don't count
                    pass
                elif rep_label == rep_pred:
                    # correct
                    num_rep_correct += 1
                    num_rep_total += 1
                else:
                    # wrong
                    num_rep_total += 1
            
            elif mode == 'loc' or mode == 'both':
                if loc_label == 0.5:
                    # don't count
                    pass
                elif loc_label == loc_pred:
                    # correct
                    num_loc_correct += 1
                    num_loc_total += 1
                else:
                    # wrong
                    num_loc_total += 1

        if mode == 'rep':
            return num_rep_correct / num_rep_total
        elif mode == 'loc':
            return num_loc_correct / num_loc_total
        elif mode == 'both':
            return num_rep_correct / num_rep_total, num_loc_correct / num_loc_total 


##### REP ONLY MODELS ### REP ONLY MODELS #####
# BEST with .83 acc        
class SmallRepOnly(nn.Module):
    def __init__(self):
        super(SmallRepOnly, self).__init__()
        # Heavily influenced by VGG
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


##### OLD ARCHI MODELS ### OLD ARCHI MODELS #####
class DuoHead(nn.Module):
    def __init__(self):
        super(DuoHead, self).__init__()

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
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )  
        
        self.loc_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )


    def forward(self, x):
        x = self.convlayers(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        r = self.rep_predictor(x)
        l = self.loc_predictor(x)
        return r, l


class BandDuo(nn.Module):
    def __init__(self):
        super(BandDuo, self).__init__()
        self.rep_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding='same', bias=False),
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False),
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True)
        )

        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(32, 64, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(64, 64, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
        )

        self.ada_pool = nn.AdaptiveMaxPool2d((7, 7))

        self.loc_predictor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64 * 7 * 7, bias=False), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )
        self.rep_predictor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64 * 7 * 7, bias=False), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

        

    def forward(self, x, xband):
        rep_x = self.rep_conv(x)
        rep_emb = torch.flatten(self.ada_pool(rep_x), 1)
        rep_prob = self.rep_predictor(rep_emb)

        loc_x = self.loc_conv(xband)
        loc_emb = torch.flatten(self.ada_pool(loc_x), 1)
        loc_prob = self.loc_predictor(loc_emb)

        return rep_prob, loc_prob

    
class BandDuo2(nn.Module):
    def __init__(self):
        super(BandDuo2, self).__init__()
        self.rep_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, padding='same', bias=False),
            nn.Dropout(p=0.15, inplace=True),
            nn.InstanceNorm2d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=7, padding='same', bias=False),
            nn.Dropout(p=0.15, inplace=True),
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding='same', bias=False), 
            nn.Dropout(p=0.15, inplace=True),
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, padding='same', bias=False),
            nn.Dropout(p=0.15, inplace=True), 
            nn.InstanceNorm2d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=7, padding='same', bias=False),
            nn.Dropout(p=0.15, inplace=True), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding='same', bias=False),
            nn.Dropout(p=0.15, inplace=True),
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.maxpool2d = nn.AdaptiveMaxPool2d((7, 7))
        
        self.rep_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )
        self.loc_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, x, xband):
        rep_x = self.rep_conv(x)
        rep_emb = torch.flatten(self.maxpool2d(rep_x), 1)
        rep_prob = self.rep_predictor(rep_emb)

        loc_x = self.loc_conv(xband)
        loc_emb = torch.flatten(self.maxpool2d(loc_x), 1)
        loc_prob = self.loc_predictor(loc_emb)

        return rep_prob, loc_prob
    
    
class BandDuo3(nn.Module):
    def __init__(self):
        super(BandDuo3, self).__init__()
        self.rep_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_conv_l1 = nn.Sequential(
            nn.Conv1d(13, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.loc_conv_l2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=7, padding='same', bias=False),
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.loc_conv_l3 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm1d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.Conv1d(4, 4, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.rep_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_pool = nn.AdaptiveMaxPool1d(7)

        self.rep_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )
        self.loc_predictor = nn.Sequential(
            nn.Linear(4 * 7, 1, bias=True), nn.Sigmoid()
        )
        
    def forward(self, dmat, band):
        loc_emb_l1 = self.loc_conv_l1(band)
        loc_emb_l2 = self.loc_conv_l2(loc_emb_l1)
        loc_emb_l3 = self.loc_conv_l3(loc_emb_l2)

        loc_emb = self.loc_pool(loc_emb_l3)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        
        rep_emb_big = self.rep_conv(dmat)
        rep_emb = self.rep_pool(rep_emb_big)
        rep_prob = self.rep_predictor(rep_emb.flatten(1))
        return rep_prob, loc_prob
    

class BandDuo4(nn.Module):
    def __init__(self):
        super(BandDuo4, self).__init__()
        self.rep_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_conv_l1 = nn.Sequential(
            nn.Conv1d(13, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.loc_conv_l2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=7, padding='same', bias=False),
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.loc_conv_l3 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm1d(4, eps=0.01), nn.ReLU(inplace=True),
            # nn.Conv1d(4, 4, kernel_size=5, padding='same', bias=False), 
            # nn.InstanceNorm2d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.rep_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_pool = nn.AdaptiveMaxPool1d(7)

        self.rep_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )
        self.loc_predictor = nn.Sequential(
            nn.Linear(4 * 7, 1, bias=True), nn.Sigmoid()
        )
        
    def forward(self, dmat, band):
        loc_emb_l1 = self.loc_conv_l1(band)
        loc_emb_l2 = self.loc_conv_l2(loc_emb_l1)
        loc_emb_l3 = self.loc_conv_l3(loc_emb_l2)

        loc_emb = self.loc_pool(loc_emb_l3)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        
        rep_emb_big = self.rep_conv(dmat)
        rep_emb = self.rep_pool(rep_emb_big)
        rep_prob = self.rep_predictor(rep_emb.flatten(1))
        return rep_prob, loc_prob


class BandDuo5(nn.Module):
    def __init__(self):
        super(BandDuo5, self).__init__()
        self.rep_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_conv_l1 = nn.Sequential(
            nn.Conv1d(13, 8, kernel_size=13, padding='same', bias=False), 
            nn.InstanceNorm1d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.loc_conv_l2 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=7, padding='same', bias=False),
            nn.InstanceNorm1d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 4, kernel_size=5, padding='same', bias=False),
            nn.InstanceNorm1d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.rep_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_pool = nn.AdaptiveMaxPool1d(7)

        self.rep_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )
        self.loc_predictor = nn.Sequential(
            nn.Linear(4 * 7, 1, bias=True), nn.Sigmoid()
        )
        
    def forward(self, dmat, band):
        loc_emb_l1 = self.loc_conv_l1(band)
        loc_emb_l2 = self.loc_conv_l2(loc_emb_l1)
        loc_emb = self.loc_pool(loc_emb_l2)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        
        rep_emb_big = self.rep_conv(dmat)
        rep_emb = self.rep_pool(rep_emb_big)
        rep_prob = self.rep_predictor(rep_emb.flatten(1))
        return rep_prob, loc_prob
    

class LocOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            # nn.Linear(64 * 7 * 7, 64 * 7 * 7, bias=True),
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob

class BandLO(nn.Module):
    # Consumes band input from SalamiDataset + 1 axis, and do 2d conv on them as opposed to 1dconv
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(4, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(4, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(8, 16, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(32, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 32 * 7 * 7, bias=True), nn.ReLU(inplace=True),
            nn.Linear(32 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        band = band[None, :]
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob


class BandLOD(nn.Module):
    # Consumes band input from SalamiDataset + 1 axis, and do 2d conv on them as opposed to 1dconv
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(32, 64, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64 * 7 * 7, bias=False), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        band = band[None, :]
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob


class BandLOD2(nn.Module):
    # Consumes band input from SalamiDataset + 1 axis, and do 2d conv on them as opposed to 1dconv
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(12, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(12, 24, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(24, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.Conv2d(24, 48, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(48, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(48 * 7 * 7, 48 * 7 * 7, bias=False), nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(48 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        band = band[None, :]
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob


class BigLocOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=13, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob


class BiggerLocOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=13, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(128, eps=0.01), nn.ReLU(inplace=True),
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob

class LocDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(8, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=7, padding='same', bias=False), 
            nn.InstanceNorm2d(16, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(32, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same', bias=False), 
            nn.InstanceNorm2d(64, eps=0.01), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.loc_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.loc_predictor = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64 * 7 * 7, bias=True),
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(64 * 7 * 7, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, band):
        square_emb = self.loc_conv(band)
        loc_emb = self.loc_pool(square_emb)
        loc_prob = self.loc_predictor(loc_emb.flatten(1))
        return loc_prob


AVAL_MODELS = {
    'SmallRepOnly': SmallRepOnly,
    'DuoHead': DuoHead,
    'BandDuo': BandDuo,
    'BandDuo5': BandDuo5,
    'LocOnly': LocOnly,
    'LocDropout': LocDropout,
    'BigLocOnly': BigLocOnly,
    'BiggerLocOnly': BiggerLocOnly,
    'BandLO': BandLO,
    'BandLOD': BandLOD,
    'BandLOD2': BandLOD2
}