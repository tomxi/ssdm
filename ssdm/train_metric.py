import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from tqdm import tqdm
import json, argparse, itertools

import ssdm
import ssdm.scanner as scn



def train(MODEL_ID='MultiResSoftmaxB', EPOCH=7, DATE='YYMMDD', DS='slm', OPT='sgd', MARGIN=0.05, net=None):
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

    slm_train_ds = ssdm.slm.PairDS(split='train', perf_margin=float(MARGIN))
    slm_val_ds = ssdm.slm.InferDS(split='val')
    hmx_train_ds = ssdm.hmx.PairDS(split='train', perf_margin=float(MARGIN))
    hmx_val_ds = ssdm.hmx.InferDS(split='val')
    jsd_train_ds = ssdm.jsd.PairDS(split='train', perf_margin=float(MARGIN))
    jsd_val_ds = ssdm.jsd.InferDS(split='val')
    rwcpop_train_ds = ssdm.rwcpop.PairDS(split='train', perf_margin=float(MARGIN))
    rwcpop_val_ds = ssdm.rwcpop.InferDS(split='val')

    # setup dataloaders
    if DS == 'hmx':
        train_ds = hmx_train_ds
        val_datasets = [hmx_val_ds]
    elif DS == 'slm':
        train_ds = slm_train_ds
        val_datasets = [slm_val_ds]
    elif DS == 'jsd':
        train_ds = jsd_train_ds
        val_datasets = [jsd_val_ds]
    elif DS == 'rwcpop':
        train_ds = rwcpop_train_ds
        val_datasets = [rwcpop_val_ds]
    elif DS == 'all':
        train_ds = ConcatDataset([hmx_train_ds, slm_train_ds, jsd_train_ds, rwcpop_train_ds])
        val_datasets = [hmx_val_ds, slm_val_ds, jsd_val_ds, rwcpop_val_ds]
    elif DS == 'all-but-hmx':
        train_ds = ConcatDataset([slm_train_ds, jsd_train_ds, rwcpop_train_ds])
        val_datasets = [slm_val_ds, jsd_val_ds, rwcpop_val_ds]
    elif DS == 'all-but-slm':
        train_ds = ConcatDataset([hmx_train_ds, jsd_train_ds, rwcpop_train_ds])
        val_datasets = [hmx_val_ds, jsd_val_ds, rwcpop_val_ds]
    elif DS == 'all-but-jsd':
        train_ds = ConcatDataset([hmx_train_ds, slm_train_ds, rwcpop_train_ds])
        val_datasets = [hmx_val_ds, slm_val_ds, rwcpop_val_ds]
    elif DS == 'all-but-rwcpop':
        train_ds = ConcatDataset([hmx_train_ds, slm_train_ds, jsd_train_ds])
        val_datasets = [hmx_val_ds, slm_val_ds, jsd_val_ds]
    else:
        assert False
    random_sampler = RandomSampler(train_ds, replacement=False, num_samples=2000)

    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=random_sampler,
        num_workers=4, pin_memory=True,
    )

    if OPT == 'adamw':
        optimizer, lr_scheduler = setup_optimizer_adamw(net)
    elif OPT == 'sgd':
        optimizer, lr_scheduler = setup_optimizer_sgd(net)
    else:
        assert False

    ### Train loop:
    # pretrain check-up
    net_pick_score_val = eval_net_picking(val_datasets, net, device=device, verbose=True)
    best_net_pick_score = net_pick_score_val.mean().item()
    print('init netpick_score:', best_net_pick_score)

    # simple logging
    train_losses = []
    val_perfs = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_contrastive(train_loader, net, optimizer, lr_scheduler=lr_scheduler, device=device, verbose=False)
        net_pick_score_val = eval_net_picking(val_datasets, net, device=device, verbose=True)
        val_perf = net_pick_score_val.mean().item()
        
        print(f'\n {short_xid_str} Post Epoch {epoch}:'), 
        print(f'\t Train: Ranking Hinge {training_loss:.4f}')
        print(f'\t Valid: Net Pick Performance {val_perf:.4f}')
        
        train_losses.append(training_loss)
        val_perfs.append(val_perf)

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


def train_contrastive(ds_loader, net, optimizer, lr_scheduler=None, device='cpu', verbose=False):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_ranking = 0
    # running_loss_nlvl = 0
    # running_ent_loss = 0
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    for i, s in iterator:
        x1 = s['x1'].to(device)
        x2 = s['x2'].to(device)

        x = torch.cat([x1, x2], 0)
        util = net(x)
        
        rank_loss_fn = torch.nn.MarginRankingLoss(margin=1)
        # nlvl_loss_fn = scn.NLvlLoss()
        contrast_label = s['perf_gap'].to(device).sign()
        rank_loss = rank_loss_fn(util[0, None], util[1, None], contrast_label)
        # nlvl_loss = nlvl_loss_fn(nlayer[0, None], s['x1_layer_score'].to(device)) + nlvl_loss_fn(nlayer[1, None], s['x2_layer_score'].to(device))
        
        # logp1 = torch.log(nlayer[0])
        # logp2 = torch.log(nlayer[1])
        # neg_entropy = torch.sum(nlayer[0] * logp1) + torch.sum(nlayer[1] * logp2) 
        # ent_loss = neg_entropy * entro_pen
        
        # loss = rank_loss + nlvl_loss + ent_loss
        # loss.backward()
        rank_loss.backward()

        # logging
        running_loss_ranking += rank_loss.item()
        # running_loss_nlvl += nlvl_loss.item()
        # running_ent_loss += ent_loss.item()

        # Manual batching
        batch_size = 8
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    return (running_loss_ranking / len(ds_loader)) #, (running_loss_nlvl / len(ds_loader)), (running_ent_loss / len(ds_loader))


def eval_net_picking(val_datasets, net, device='cpu', verbose=True):
    all_ds_net_pick = []
    for val_ds in val_datasets:
        net_pick = ssdm.net_pick_performance_util_only(val_ds, net, device=device)
        all_ds_net_pick.append(net_pick)
        if verbose:
            print(f'net pick l recall on {val_ds}:')
            print('\t' , net_pick.mean('tid').item())
            print('__________')
    all_ds_net_pick = ssdm.xr.concat(all_ds_net_pick, 'tid')
    if verbose:
        print(f'net pick l recall score on all datasets combined:')
        print('\t' , all_ds_net_pick.mean('tid').item())
        print('===================')

    return all_ds_net_pick


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    # parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('optimizer', help='Which sgd or adamw')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date
    opt = kwargs.optimizer

    overwrite_config_list = list(itertools.product(
        ['MultiResSoftmaxB', 'EfficientNetB0'],
        ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-jsd', 'all-but-rwcpop', 'all-but-slm', 'all-but-hmx'],
        [0.05],
    ))
    model_id, ds, score_margin = overwrite_config_list[int(kwargs.config_idx)]

    train(model_id, total_epoch, date, ds, opt, score_margin, None)
    print('done without failure!')

