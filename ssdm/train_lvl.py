import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from tqdm import tqdm
import json, argparse, itertools
import numpy as np

import ssdm
import ssdm.scanner as scn


def main(MODEL_ID='MultiResSoftmaxB', EPOCH=5, DATE='YYMMDD', DS='rwcpop', OPT='sgd', ETP=0.01, net=None):
    """
    """
    short_xid_str = f'{MODEL_ID}-{DS}-{OPT}-vmeasure_nlvl_only'
    experiment_id_str = f'{DATE}_{EPOCH}_{short_xid_str}'
    print(experiment_id_str)
    
    # setup device
    # torch.multiprocessing.set_start_method('spawn')
    device = torch.device(torch.device('cuda') if torch.cuda.is_available() else "cpu")
    print(device)

    if net is None:
        # Initialize network based on model_id:
        net = scn.AVAL_MODELS[MODEL_ID]()
    net.to(device)
    
    train_loader, val_datasets = setup_dataset(DS)
    
    if OPT == 'adamw':
        optimizer, lr_scheduler = setup_optimizer_adamw(net)
    elif OPT == 'sgd':
        optimizer, lr_scheduler = setup_optimizer_sgd(net, step_size=250, init_lr=1e-3)
    else:
        assert False

    ### Train loop:
    # pretrain check-up and setup baseline:
    net_pick_score_val, orc_score_val = eval_net_picking(val_datasets, net, device=device, verbose=True)
    best_net_pick_score = net_pick_score_val.mean().item()
    best_net_pick_score_orclvl = orc_score_val.mean().item()
    print('init netpick_score:', best_net_pick_score, 'with oracle lvl choice:', best_net_pick_score_orclvl)

    # simple logging
    train_losses = []
    val_perfs = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_lvl_epoch(
            train_loader, net, optimizer, 
            lr_scheduler=lr_scheduler, entro_pen=float(ETP), 
            device=device, verbose=False
        )
        net_pick_score_val, orc_score_val = eval_net_picking(
            val_datasets, net, device=device, verbose=True
        )
        val_perf = net_pick_score_val.mean().item()
        val_orc = orc_score_val.mean().item()
        
        print(f'\n {short_xid_str} Post Epoch {epoch + 1}:'), 
        print(f'\t Train: nlvl Loss {training_loss[0]:.4f}, entropy loss: {training_loss[1]:.4f}')
        print(f'\t Valid: Net Pick Performance {val_perf:.4f} oracle lvl choice {val_orc:.4f}')
        
        train_losses.append(training_loss)
        val_perfs.append((val_perf, val_orc))

        if val_perf > best_net_pick_score:
            # update best_loss and save model
            best_net_pick_score = val_perf
            best_state = net.state_dict()
            torch.save(best_state, f'{short_xid_str}_best_netpick')

        if epoch % 5 == 0:
            # save every 5 epoch regardless
            torch.save(net.state_dict(), f'{short_xid_str}_e{epoch}_statedict')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                          'val_perf': val_perfs
                          }
        with open(f'{short_xid_str}.json', 'w') as file:
            json.dump(trainning_info, file)
    
    return net

def train_lvl_epoch(ds_loader, net, optimizer, lr_scheduler=None, entro_pen=0.01, device='cuda', verbose=False):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_nlvl = 0
    running_ent_loss = 0
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    nlvl_loss_fn = NLvlLoss(scale_by_target=False)

    for i, s in iterator:

        x = s['x'].to(device)
        vmeasures = s['lvl_score'].to(device)
        
        util_pred, nlvl_softmax = net(x)
        
        nlvl_loss, entropy = nlvl_loss_fn(nlvl_softmax, vmeasures)
        ent_loss = -entropy * entro_pen
        
        loss = nlvl_loss + ent_loss
        loss.backward()

        # logging
        running_loss_nlvl += nlvl_loss.item()
        running_ent_loss += ent_loss.item()

        # Manual batching
        batch_size = 16
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    mean_nlvl_loss = running_loss_nlvl / len(ds_loader)
    mean_nlvl_entropy = running_ent_loss / len(ds_loader)
    return mean_nlvl_loss, mean_nlvl_entropy


def eval_net_picking(val_datasets, net, device='cuda', verbose=True):
    all_ds_net_pick = []
    all_ds_orc = []
    for val_ds in val_datasets:
        perf = ssdm.net_pick_performance(val_ds, net, device=device)
        net_lvl_orc_feat = perf['orc_feat_net_lvl']
        orc = perf['orc']

        new_tids = [val_ds.name + str(i) for i in net_lvl_orc_feat['tid'].values]
        all_ds_net_pick.append(net_lvl_orc_feat.assign_coords(tid=new_tids))
        all_ds_orc.append(orc.assign_coords(tid=new_tids))
        if verbose:
            print(f'net pick v-measure on {val_ds}:')
            print('\tnet pick lvl with orc feats / orc picks' , net_lvl_orc_feat.mean('tid').item(), orc.mean().item())
            print('__________')
        # break
    all_ds_net_pick = ssdm.xr.concat(all_ds_net_pick, 'tid')
    all_ds_orc = ssdm.xr.concat(all_ds_orc, 'tid')
    if verbose:
        print(f'net pick v-measure on all val ds combined:')
        print('\tnet pick lvl with orc feats / orc picks\n\t' , all_ds_net_pick.mean('tid').item(), all_ds_orc.mean('tid').item())
        print('===================')

    return all_ds_net_pick, all_ds_orc


def setup_optimizer_adamw(net):
    # training details
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": 2,
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(grouped_parameters)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=2e-7, max_lr=1e-4, cycle_momentum=False, mode='triangular2', step_size_up=3000
    )
    return optimizer, lr_scheduler


def setup_optimizer_sgd(net, step_size=1250, init_lr=1e-3):
    # training details
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": 0.001,	
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(grouped_parameters, lr=init_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.95) # goes down by ~1k every 135 steps
    return optimizer, lr_scheduler


def setup_dataset(option='all'):
    if option == 'hmx':
        train_ds = ssdm.hmx.LvlDS(split='train')
        val_datasets = [ssdm.hmx.InferDS(split='val')]
    elif option == 'slm':
        train_ds = ssdm.slm.LvlDS(split='train')
        val_datasets = [ssdm.slm.InferDS(split='val')]
    elif option == 'jsd':
        train_ds = ssdm.jsd.LvlDS(split='train')
        val_datasets = [ssdm.jsd.InferDS(split='val')]
    elif option == 'rwcpop':
        train_ds = ssdm.rwcpop.LvlDS(split='train')
        val_datasets = [ssdm.rwcpop.InferDS(split='val')]
    elif option == 'all':
        train_ds = ConcatDataset(
            [ssdm.hmx.LvlDS(split='train'), 
             ssdm.slm.LvlDS(split='train'),
             ssdm.jsd.LvlDS(split='train'), 
             ssdm.rwcpop.LvlDS(split='train')]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
    elif option == 'all-but-hmx':
        train_ds = ConcatDataset(
            [ssdm.slm.LvlDS(split='train'),
             ssdm.jsd.LvlDS(split='train'), 
             ssdm.rwcpop.LvlDS(split='train')]
        )
        val_datasets = [ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-slm':
        train_ds = ConcatDataset(
            [ssdm.hmx.LvlDS(split='train'), 
             ssdm.jsd.LvlDS(split='train'), 
             ssdm.rwcpop.LvlDS(split='train')]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-jsd':
        train_ds = ConcatDataset(
            [ssdm.hmx.LvlDS(split='train'), 
             ssdm.slm.LvlDS(split='train'),
             ssdm.rwcpop.LvlDS(split='train')])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-rwcpop':
        train_ds = ConcatDataset(
            [ssdm.hmx.LvlDS(split='train'), 
             ssdm.slm.LvlDS(split='train'),
             ssdm.jsd.LvlDS(split='train')])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val')]
    else:
        assert False

    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=PermutationSampler(train_ds, 4000),
        num_workers=4, pin_memory=True,
    )
    
    return train_loader, val_datasets


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


class PermutationSampler(Sampler):
    def __init__(self, data_source, epoch_length=20000):
        self.data_source = data_source
        self.epoch_length = epoch_length

    def __iter__(self):
        n = len(self.data_source)
        indices = np.random.permutation(n)
        for i in range(0, self.epoch_length):
            yield indices[i % n]

    def __len__(self):
        return self.epoch_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    # parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    # parser.add_argument('optimizer', help='Which sgd or adamw')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date
    opt = 'sgd'
    etp = 0.01

    model_id, ds = list(itertools.product(
        ['MultiResSoftmaxB', 'EfficientNetB0'],
        ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-jsd', 'all-but-rwcpop', 'all-but-slm', 'all-but-hmx'],
    ))[int(kwargs.config_idx)]

    main(MODEL_ID=model_id, EPOCH=total_epoch, DATE=date, DS=ds, OPT=opt, ETP=0.01, net=None)
    print('done without failure!')

