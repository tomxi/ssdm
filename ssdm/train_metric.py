import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm
import json, argparse, itertools

import ssdm
import ssdm.scanner as scn

def train(MODEL_ID='MultiRes', EPOCH=7, DATE='YYMMDD', METRIC='l', MARGIN=0.05, ETP=0.01, net=None):
    """
    """
    short_xid_str = f'{MODEL_ID}-etp{ETP}-{METRIC}{MARGIN}'
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

    # setup dataloaders
    train_ds = ssdm.hmx.PairDS(split='train', perf_metric=METRIC, perf_margin=float(MARGIN))
    val_ds = ssdm.base.HmxDS(split='val', infer=True, device=device, nlvl_metric=METRIC)
    random_sampler = RandomSampler(train_ds, replacement=False, num_samples=5000)

    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=random_sampler,
        num_workers=4, pin_memory=True,
    )

    # training details
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": 10,
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(grouped_parameters)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-6, max_lr=2e-4, cycle_momentum=False, mode='triangular', step_size_up=2000
    )


    ### Train loop:
    # pretrain check-up
    net_pick_score_val, net_pick_score_val_orclvl = ssdm.net_pick_performance(val_ds, net, METRIC=='l')
    best_net_pick_score = net_pick_score_val.sel(m_type='f').mean().item()
    best_net_pick_score_orclvl = net_pick_score_val_orclvl.sel(m_type='f').mean().item()
    print('init netpick_score:', best_net_pick_score, 'with oracle lvl choice:', best_net_pick_score_orclvl)

    # simple logging
    train_losses = []
    val_perfs = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_contrastive(train_loader, net, optimizer, lr_scheduler=lr_scheduler, device=device, entro_pen=ETP, verbose=False)
        net_pick_score_val, net_pick_score_val_orclvl = ssdm.net_pick_performance(val_ds, net, METRIC=='l')
        val_perf = net_pick_score_val.sel(m_type='f').mean().item()
        val_perf_orclvl = net_pick_score_val_orclvl.sel(m_type='f').mean().item()
        
        print(f'\n {short_xid_str} Post Epoch {epoch}:'), 
        print(f'\t Train: Ranking Hinge {training_loss[0]:.4f}, nlvl Loss {training_loss[1]:.4f}, entropy loss: {training_loss[2]:.4f}')
        print(f'\t Valid: Net Pick Performance {val_perf:.4f} oracle lvl choice {val_perf_orclvl:.4f}')
        
        train_losses.append(training_loss)
        val_perfs.append((val_perf, val_perf_orclvl))

        if val_perf > best_net_pick_score:
            # update best_loss and save model
            best_net_pick_score = val_perf
            best_state = net.state_dict()
            torch.save(best_state, f'{short_xid_str}_best_netpick')

        if epoch % 1 == 0:
            # save every 1 epoch regardless
            torch.save(net.state_dict(), f'{short_xid_str}_e{epoch}_statedict')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                          'val_perf': val_perfs
                          }
        with open(f'{short_xid_str}.json', 'w') as file:
            json.dump(trainning_info, file)
    
    return net


def train_contrastive(ds_loader, net, optimizer, lr_scheduler=None, device='cpu', entro_pen=0, verbose=False):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_ranking = 0
    running_loss_nlvl = 0
    running_ent_loss = 0
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    for i, s in iterator:
        x1 = s['x1'].to(device)
        x2 = s['x2'].to(device)
        util1, nlayer1 = net(x1)
        util2, nlayer2 = net(x2)
        
        rank_loss_fn = torch.nn.MarginRankingLoss(margin=1)
        nlvl_loss_fn = scn.NLvlLoss()
        contrast_label = s['perf_gap'].to(device).sign()
        rank_loss = rank_loss_fn(util1, util2, contrast_label)
        nlvl_loss = nlvl_loss_fn(nlayer1, s['x1_layer_score'].to(device)) + nlvl_loss_fn(nlayer2, s['x2_layer_score'].to(device))
        
        logp1 = torch.log(nlayer1)
        logp2 = torch.log(nlayer2)
        neg_entropy = torch.sum(nlayer1 * logp1) + torch.sum(nlayer2 * logp2) 
        ent_loss = neg_entropy * entro_pen
        
        loss = rank_loss + nlvl_loss + ent_loss
        loss.backward()

        # logging
        running_loss_ranking += rank_loss.item()
        running_loss_nlvl += nlvl_loss.item()
        running_ent_loss += ent_loss.item()

        # Manual batching
        batch_size = 8
        if i % batch_size == (batch_size - 1):
            # take back prop step
            # nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    return (running_loss_ranking / len(ds_loader)), (running_loss_nlvl / len(ds_loader)), (running_ent_loss / len(ds_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    # parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('metric', help='Which dataset? l or pfc')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date
    metric = kwargs.metric

    overwrite_config_list = list(itertools.product(
        ['MultiRes', 'MultiResSoftmax', 'MultiResC'],
        [0.07, 0.02, 0.005],
        [0.01],
    ))
    model_id, score_margin, etp = overwrite_config_list[int(kwargs.config_idx)]

    train(model_id, total_epoch, date, metric, score_margin, etp, None)
    print('done without failure!')
