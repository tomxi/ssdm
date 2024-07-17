import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, Sampler

from tqdm import tqdm
import json, argparse, itertools
import numpy as np

import ssdm
import ssdm.scanner as scn


def main(MODEL_ID='MultiResSoftmaxB', EPOCH=7, DATE='YYMMDD', DS='slm', OPT='sgd', ETP=0.01, MARGIN=0.001, net=None):
    """
    """
    short_xid_str = f'{MODEL_ID}-{DS}-{OPT}-l_recall{MARGIN}'
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
    
    train_loader, val_datasets = setup_dataset(DS, float(MARGIN))
    
    if OPT == 'adamw':
        optimizer, lr_scheduler = setup_optimizer_adamw(net)
    elif OPT == 'sgd':
        optimizer, lr_scheduler = setup_optimizer_sgd(net)
    else:
        assert False

    ### Train loop:
    # pretrain check-up and setup baseline:
    net_pick_score_val, net_pick_score_val_orclvl = eval_net_picking(val_datasets, net, device=device, verbose=True)
    best_net_pick_score = net_pick_score_val.mean().item()
    best_net_pick_score_orclvl = net_pick_score_val_orclvl.mean().item()
    print('init netpick_score:', best_net_pick_score, 'with oracle lvl choice:', best_net_pick_score_orclvl)

    # simple logging
    train_losses = []
    val_perfs = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_contrastive_epoch(train_loader, net, optimizer, lr_scheduler=lr_scheduler, entro_pen=float(ETP), device=device, verbose=False)
        net_pick_score_val, net_pick_score_val_orclvl = eval_net_picking(val_datasets, net, device=device, verbose=True)
        val_perf = net_pick_score_val.mean().item()
        val_perf_orclvl = net_pick_score_val_orclvl.mean().item()
        
        print(f'\n {short_xid_str} Post Epoch {epoch + 1}:'), 
        print(f'\t Train: Ranking Hinge {training_loss[0]:.4f}, nlvl Loss {training_loss[1]:.4f}, entropy loss: {training_loss[2]:.4f}')
        print(f'\t Valid: Net Pick Performance {val_perf:.4f} oracle lvl choice {val_perf_orclvl:.4f}')
        
        train_losses.append(training_loss)
        val_perfs.append((val_perf, val_perf_orclvl))

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


def train_contrastive_epoch(ds_loader, net, optimizer, lr_scheduler=None, entro_pen=0.01, device='cpu', verbose=False):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_ranking = 0
    running_loss_nlvl = 0
    running_ent_loss = 0
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    rank_loss_fn = torch.nn.MarginRankingLoss(margin=1)
    nlvl_loss_fn = NLvlLoss(scale_by_target=True)

    for i, s in iterator:

        x = torch.cat([s['x1'], s['x2']], 0).to(device)
        vmeasures = torch.cat([s['x1_vmeasure'], s['x2_vmeasure']], 0).to(device)
        
        util_pred, nlvl_softmax = net(x)
        
        contrast_label = s['perf_gap'].to(device).sign()
        rank_loss = rank_loss_fn(util_pred[0, None], util_pred[1, None], contrast_label)
        nlvl_loss, entropy = nlvl_loss_fn(nlvl_softmax, vmeasures)
        ent_loss = -entropy * entro_pen
        
        loss = rank_loss + nlvl_loss + ent_loss
        loss.backward()

        # logging
        running_loss_ranking += rank_loss.item()
        running_loss_nlvl += nlvl_loss.item()
        running_ent_loss += ent_loss.item()

        # Manual batching
        batch_size = 32
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    return (running_loss_ranking / len(ds_loader)) , (running_loss_nlvl / len(ds_loader)), (running_ent_loss / len(ds_loader))


def eval_net_picking(val_datasets, net, device='cpu', verbose=True):
    all_ds_net_pick = []
    all_ds_net_pick_orc_lvl = []
    for val_ds in val_datasets:
        net_pick, net_pick_orc_lvl = ssdm.net_pick_performance(val_ds, net, device=device)
        new_tids = [val_ds.name + str(i) for i in net_pick['tid'].values]
        all_ds_net_pick.append(net_pick.assign_coords(tid=new_tids))
        all_ds_net_pick_orc_lvl.append(net_pick_orc_lvl.assign_coords(tid=new_tids))
        if verbose:
            print(f'net pick v-measure on {val_ds}:')
            print('\t' , net_pick.mean('tid').item(), net_pick_orc_lvl.mean('tid').item())
            print('__________')
        # break
    all_ds_net_pick = ssdm.xr.concat(all_ds_net_pick, 'tid')
    all_ds_net_pick_orc_lvl = ssdm.xr.concat(all_ds_net_pick_orc_lvl, 'tid')
    if verbose:
        print(f'net pick v-measure on all val ds combined:')
        print('\t' , all_ds_net_pick.mean('tid').item(), all_ds_net_pick_orc_lvl.mean('tid').item())
        print('===================')

    return all_ds_net_pick, all_ds_net_pick_orc_lvl


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


def setup_optimizer_sgd(net, init_lr=1e-4):
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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1200, gamma=0.92)
    return optimizer, lr_scheduler


def setup_dataset(option='all', perf_margin=0.01):
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

    # random_sampler = RandomSampler(train_ds, replacement=False, num_samples=10000)
    random_sampler = PermutationSampler(train_ds, 20000)
    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=random_sampler,
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
    def __init__(self, data_source, epoch_length=10000):
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

    main(MODEL_ID=model_id, EPOCH=total_epoch, DATE=date, DS=ds, OPT=opt, ETP=0.01, MARGIN=0.001, net=None)
    print('done without failure!')



