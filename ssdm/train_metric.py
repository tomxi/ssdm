import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from tqdm import tqdm
import json, argparse, itertools
import numpy as np

import ssdm
import ssdm.scanner as scn


def main(MODEL_ID='MultiResSoftmaxB', EPOCH=7, DATE='YYMMDD', DS='slm', OPT='sgd', ETP=0.01, MARGIN=0.001, net=None):
    """
    """
    short_xid_str = f'{MODEL_ID}-{DS}-{OPT}-etp{ETP}-vmeasure{MARGIN}'
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
    
    train_loader, val_datasets = setup_dataset(DS, epoch_length=2048 * 2)
    
    if OPT == 'adamw':
        optimizer, lr_scheduler = setup_optimizer_adamw(net)
    elif OPT == 'sgd':
        optimizer, lr_scheduler = setup_optimizer_sgd(net, step_size=256, init_lr=1e-4)
    else:
        assert False

    ### Train loop:
    # pretrain check-up and setup baseline:
    net_score_val = eval_net_picking(val_datasets, net, device=device, verbose=True)
    best_net_pick_score = net_score_val['net_pick'].mean().item()
    # print('init netpick_score:', best_net_pick_score, 'with oracle lvl choice:', best_net_pick_score_orclvl)

    # simple logging
    train_losses = []
    val_perfs = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_contrastive_epoch(
            train_loader, net, optimizer, 
            lr_scheduler=lr_scheduler, entro_pen=float(ETP), 
            device=device, verbose=False
        )
        net_score_val = eval_net_picking(
            val_datasets, net, device=device, verbose=True
        )
        val_perf = net_score_val['net_pick'].mean().item()
        
        train_losses.append(training_loss)
        val_perfs.append({k: net_score_val[k].mean().item() for k in net_score_val})

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


def train_contrastive_epoch(ds_loader, net, optimizer, lr_scheduler=None, entro_pen=0.01, device='cuda', verbose=False):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_ranking = []
    running_loss_nlvl = []
    running_ent_loss = []
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    rank_loss_fn = torch.nn.MarginRankingLoss(margin=1)
    nlvl_loss_fn = NLvlLoss(scale_by_target=False)

    for i, s in iterator:

        x = torch.cat([s['x1'], s['x2']], 0).to(device)
        vmeasures = torch.cat([s['x1_vmeasure'], s['x2_vmeasure']], 0).to(device)
        
        util_pred, nlvl_softmax = net(x)
        
        contrast_label = s['perf_gap'].to(device).sign()
        rank_loss = rank_loss_fn(util_pred[0, None], util_pred[1, None], contrast_label)

        if (s['x1_rank'] <= 8) or (s['x2_rank'] <= 8):
            if s['x1_rank'] > 8:
                nlvl_softmax = nlvl_softmax[1, None]
                vmeasures = vmeasures[1, None]
            elif s['x2_rank'] > 8:
                nlvl_softmax = nlvl_softmax[0, None]
                vmeasures = vmeasures[0, None]


            nlvl_loss, entropy = nlvl_loss_fn(nlvl_softmax, vmeasures)
            ent_loss = -entropy * entro_pen
            loss = rank_loss + nlvl_loss + ent_loss

            # logging
            running_loss_nlvl.append(nlvl_loss.item())
            running_ent_loss.append(ent_loss.item())
            running_loss_ranking.append(rank_loss.item())
        else:
            loss = rank_loss
            # logging
            running_loss_ranking.append(rank_loss.item())
        
        loss.backward()

        # Manual batching
        batch_size = 16
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    out = dict(rank = np.array(running_loss_ranking).mean(), 
               lvl = np.array(running_loss_nlvl).mean(),
               entropy = np.array(running_ent_loss).mean())

    return out


def eval_net_picking(val_datasets, net, device='cuda', verbose=False):
    all_ds_net_pick_both = []
    all_ds_net_pick_lvl = []
    all_ds_net_pick_feat = []
    all_ds_orc = []
    all_ds_scores = []

    for val_ds in val_datasets:
        perf = ssdm.net_pick_performance(val_ds, net, device=device)
        net_pick = perf['net_pick']
        net_lvl_orc_feat = perf['orc_feat_net_lvl']
        net_feat_orc_lvl = perf['net_feat_orc_lvl']
        orc = perf['orc']

        

        new_tids = [val_ds.name + str(i) for i in net_lvl_orc_feat['tid'].values]
        all_ds_net_pick_both.append(net_pick.assign_coords(tid=new_tids))
        all_ds_net_pick_lvl.append(net_lvl_orc_feat.assign_coords(tid=new_tids))
        all_ds_net_pick_feat.append(net_feat_orc_lvl.assign_coords(tid=new_tids))
        all_ds_orc.append(orc.assign_coords(tid=new_tids))
        all_ds_scores.append(val_ds.scores.copy().assign_coords(tid=new_tids))

    

    all_ds_scores = ssdm.xr.concat(all_ds_scores, 'tid')
    all_ds_net_pick_both = ssdm.xr.concat(all_ds_net_pick_both, 'tid')
    all_ds_net_pick_lvl = ssdm.xr.concat(all_ds_net_pick_lvl, 'tid')
    all_ds_net_pick_feat = ssdm.xr.concat(all_ds_net_pick_feat, 'tid')
    all_ds_orc = ssdm.xr.concat(all_ds_orc, 'tid')
    
    all_ds_boa_choice = all_ds_scores.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype', 'layer'])
    all_ds_boa_scores = all_ds_scores.isel(all_ds_boa_choice)

    if verbose:
        print()
        print(f'net pick v-measure on all val ds combined:')
        print('\toracle score:',all_ds_orc.mean('tid').item())
        print('\tlvl pick gap:', all_ds_orc.mean('tid').item() - all_ds_net_pick_lvl.mean('tid').item())
        print('\tfeat pick gap:', all_ds_orc.mean('tid').item() - all_ds_net_pick_feat.mean('tid').item())
        print('\tboth pick gap:', all_ds_orc.mean('tid').item() - all_ds_net_pick_both.mean('tid').item())
        print('\tBoA pick gap:', all_ds_orc.mean('tid').item() - all_ds_boa_scores.mean('tid').item())
        print('\tBoA choices:', all_ds_boa_scores.mean('tid').rep_ftype.item(), all_ds_boa_scores.mean('tid').loc_ftype.item(), all_ds_boa_scores.mean('tid').layer.item())
        print('===================')

    out = dict(net_pick=all_ds_net_pick_both, net_pick_lvl=all_ds_net_pick_lvl, net_pick_feat=all_ds_net_pick_feat, orc=all_ds_orc)
    return out


def setup_optimizer_adamw(net):
    # training details
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": 5,
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(grouped_parameters)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-7, max_lr=1e-4, cycle_momentum=False, mode='triangular', step_size_up=2048 * 2
    )
    return optimizer, lr_scheduler


def setup_optimizer_sgd(net, step_size=256, init_lr=1e-4):
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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9) # goes down by ~1k every 135 steps
    return optimizer, lr_scheduler


def setup_dataset(option='all', perf_margin=0.001, epoch_length=2048):
    if option == 'hmx':
        train_ds = ssdm.hmx.PairDS(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.hmx.InferDS(split='val')]
    elif option == 'slm':
        train_ds = ssdm.slm.PairDS(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.slm.InferDS(split='val')]
    elif option == 'jsd':
        train_ds = ssdm.jsd.PairDS(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.jsd.InferDS(split='val')]
    elif option == 'rwcpop':
        train_ds = ssdm.rwcpop.PairDS(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.rwcpop.InferDS(split='val')]
    elif option == 'all':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDS(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDS(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
    elif option == 'all-but-hmx':
        train_ds = ConcatDataset(
            [ssdm.slm.PairDS(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDS(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-slm':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.jsd.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDS(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-jsd':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDS(split='train', perf_margin=perf_margin),
             ssdm.rwcpop.PairDS(split='train', perf_margin=perf_margin)])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]

    elif option == 'all-but-rwcpop':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDS(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDS(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDS(split='train', perf_margin=perf_margin)])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val')]
    else:
        assert False

    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=PermutationSampler(train_ds, epoch_length),
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
    def __init__(self, data_source, epoch_length=2048):
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
    parser.add_argument('margin', help='sampling score margin')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date
    opt = 'adamw'
    etp = 0.01
    margin = float(kwargs.margin) # 0.001

    model_id, ds= list(itertools.product(
        ['MultiResSoftmaxB'],
        # ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-jsd', 'all-but-rwcpop', 'all-but-slm', 'all-but-hmx'],
        ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-slm'],
    ))[int(kwargs.config_idx)]

    main(MODEL_ID=model_id, EPOCH=total_epoch, DATE=date, DS=ds, OPT=opt, ETP=etp, MARGIN=margin, net=None)
    print('done without failure!')

