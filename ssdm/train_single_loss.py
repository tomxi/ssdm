import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse, itertools
import numpy as np

import ssdm
import ssdm.scanner as scn


def main(MODEL_ID='MultiResSoftmaxUtil', EPOCH=7, DATE='YYMMDD', DS='dev', OPT='adamw', MARGIN=0.05, net=None):
    """
    """
    short_xid_str = f'{MODEL_ID}-{DS}-{OPT}-lmeasure_gap{MARGIN}'
    experiment_id_str = f'{DATE}_{short_xid_str}'
    print(experiment_id_str)

    # setup writer
    writer = SummaryWriter(f'/scratch/qx244/tblogs/{experiment_id_str}')
    
    # setup device
    # torch.multiprocessing.set_start_method('spawn')
    device = torch.device(torch.device('cuda') if torch.cuda.is_available() else "cpu")
    print(device)

    if net is None:
        # Initialize network based on model_id:
        net = scn.AVAL_MODELS[MODEL_ID]()
    net.to(device)
    
    train_loader, val_datasets, train_infer = setup_dataset(DS, epoch_length=2048 * 16)
    
    if OPT == 'adamw':
        optimizer, lr_scheduler = setup_optimizer_adamw(
            net, step_size_up=2048 * 12, weight_decay=0.1, base_lr=3e-6, max_lr=3e-3
        )
    elif OPT == 'sgd':
        optimizer, lr_scheduler = setup_optimizer_sgd(net, step_size=2048 * 6, init_lr=2e-3)
    else:
        assert False

    ### Train loop:
    # pretrain check-up and setup baseline:
    net_score_val = eval_net_picking(val_datasets, net, device=device)
    net_score_train = eval_net_picking(train_infer, net, device=device)
    writer.add_scalars('epoch/perf_on_val', net_score_val, 0)
    writer.add_scalars('epoch/perf_on_train', net_score_train, 0)


    for epoch in tqdm(range(int(EPOCH))):
        training_loss = train_contrastive_epoch(
            train_loader, net, optimizer, 
            lr_scheduler=lr_scheduler,
            device=device, verbose=False, writer=writer, epoch=epoch
        )
        writer.add_scalar('epoch/loss', training_loss['hinge'].mean(), epoch + 1)
        
        net_score_val = eval_net_picking(val_datasets, net, device=device)
        net_score_train = eval_net_picking(train_infer, net, device=device)
        writer.add_scalars('epoch/perf_on_val', net_score_val, epoch + 1)
        writer.add_scalars('epoch/perf_on_train', net_score_train, epoch + 1)
        writer.flush()
        torch.save(net.state_dict(), f'{short_xid_str}_e{epoch}_statedict')  
    return net


def train_contrastive_epoch(ds_loader, net, optimizer, lr_scheduler=None, device='cuda', verbose=False, writer=None, epoch=0):
    optimizer.zero_grad()
    net.to(device)
    net.train()
    running_loss_ranking = []
    iterator = enumerate(ds_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(ds_loader))

    rank_loss_fn = torch.nn.MarginRankingLoss(margin=1)
    num_batches_per_epoch = len(ds_loader) // 16
    gloab_batch_count = num_batches_per_epoch * epoch

    for i, s in iterator:
        x = torch.cat([s['x1'], s['x2']], 0).to(device)
        util = net(x)
        
        contrast_label = s['perf_gap'].to(device).sign()
        loss = rank_loss_fn(util[0, None], util[1, None], contrast_label)
        loss.backward()
        running_loss_ranking.append(loss.item())

        # Manual batching
        batch_size = 16
        if i % batch_size == (batch_size - 1):
            # take back prop step
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            # logging
            writer.add_scalar('batch/loss', np.mean(running_loss_ranking[-batch_size:]), gloab_batch_count + i // batch_size)
            writer.add_scalar('batch/LR', lr_scheduler.get_last_lr()[0], gloab_batch_count + i // batch_size)

    losses = dict(hinge = np.array(running_loss_ranking))
    return losses


def eval_net_picking(val_datasets, net, device='cuda', full=False):
    all_ds_net_feat_scores = []
    all_ds_orc_scores = []
    all_ds_scores = []

    for val_ds in val_datasets:
        ds_score = val_ds.scores.max('layer')
        net_output = scn.net_infer_util_only(val_ds, net, device=device)
        net_feat_picks = net_output.argmax(dim=['rep_ftype', 'loc_ftype'])
        orc_feat_picks = ds_score.argmax(dim=['rep_ftype', 'loc_ftype'])
        all_ds_net_feat_scores.append(ds_score.isel(net_feat_picks))
        all_ds_orc_scores.append(ds_score.isel(orc_feat_picks))
        all_ds_scores.append(ds_score)

    all_ds_scores = ssdm.xr.concat(all_ds_scores, 'tid')
    all_ds_net_feat_scores = ssdm.xr.concat(all_ds_net_feat_scores, 'tid')
    all_ds_orc_scores = ssdm.xr.concat(all_ds_orc_scores, 'tid')
    all_ds_boa_picks = all_ds_scores.mean('tid').argmax(dim=['rep_ftype', 'loc_ftype'])

    if full:
        out = dict(net_pick=all_ds_net_feat_scores, 
                orc_pick=all_ds_orc_scores, 
                boa_pick=all_ds_scores.isel(all_ds_boa_picks)
                )
    else:
        out = dict(net_pick=all_ds_net_feat_scores.mean().item(), 
                orc_pick=all_ds_orc_scores.mean().item(), 
                boa_pick=all_ds_scores.isel(all_ds_boa_picks).mean().item()
                )
    return out


def setup_optimizer_adamw(net, step_size_up=2048*2, weight_decay=0.1, base_lr=3e-6, max_lr=3e-3):
    # training details
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(grouped_parameters)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=False, mode='triangular2', step_size_up=step_size_up
    )
    return optimizer, lr_scheduler


def setup_optimizer_sgd(net, step_size=2048, init_lr=2e-4):
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


def setup_dataset(option='all', perf_margin=0.05, epoch_length=8 * 2048):
    if option == 'hmx':
        train_ds = ssdm.hmx.PairDSLmeasure(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.hmx.InferDS(split='val')]
        train_infer = [ssdm.hmx.InferDS(split='train')]
    elif option == 'slm':
        train_ds = ssdm.slm.PairDSLmeasure(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.slm.InferDS(split='val')]
        train_infer = [ssdm.slm.InferDS(split='train')]
    elif option == 'jsd':
        train_ds = ssdm.jsd.PairDSLmeasure(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.jsd.InferDS(split='val')]
        train_infer = [ssdm.jsd.InferDS(split='train')]
    elif option == 'rwcpop':
        train_ds = ssdm.rwcpop.PairDSLmeasure(split='train', perf_margin=perf_margin)
        val_datasets = [ssdm.rwcpop.InferDS(split='val')]
        train_infer = [ssdm.rwcpop.InferDS(split='train')]
    elif option == 'all':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDSLmeasure(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDSLmeasure(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
        train_infer = [ssdm.hmx.InferDS(split='train'), 
                       ssdm.slm.InferDS(split='train'),
                       ssdm.jsd.InferDS(split='train'), 
                       ssdm.rwcpop.InferDS(split='train')]

    elif option == 'all-but-hmx':
        train_ds = ConcatDataset(
            [ssdm.slm.PairDSLmeasure(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDSLmeasure(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
        train_infer = [ssdm.slm.InferDS(split='train'),
                       ssdm.jsd.InferDS(split='train'), 
                       ssdm.rwcpop.InferDS(split='train')]

    elif option == 'all-but-slm':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.jsd.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.rwcpop.PairDSLmeasure(split='train', perf_margin=perf_margin)]
        )
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
        train_infer = [ssdm.hmx.InferDS(split='train'), 
                       ssdm.jsd.InferDS(split='train'), 
                       ssdm.rwcpop.InferDS(split='train')]

    elif option == 'all-but-jsd':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDSLmeasure(split='train', perf_margin=perf_margin),
             ssdm.rwcpop.PairDSLmeasure(split='train', perf_margin=perf_margin)])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.rwcpop.InferDS(split='val')]
        train_infer = [ssdm.hmx.InferDS(split='train'), 
                       ssdm.slm.InferDS(split='train'), 
                       ssdm.rwcpop.InferDS(split='train')]

    elif option == 'all-but-rwcpop':
        train_ds = ConcatDataset(
            [ssdm.hmx.PairDSLmeasure(split='train', perf_margin=perf_margin), 
             ssdm.slm.PairDSLmeasure(split='train', perf_margin=perf_margin),
             ssdm.jsd.PairDSLmeasure(split='train', perf_margin=perf_margin)])
        val_datasets = [ssdm.hmx.InferDS(split='val'), 
                        ssdm.slm.InferDS(split='val'), 
                        ssdm.jsd.InferDS(split='val')]
        train_infer = [ssdm.hmx.InferDS(split='train'), 
                       ssdm.slm.InferDS(split='train'),
                       ssdm.jsd.InferDS(split='train')]
    
    elif option == 'dev':
        train_ds = ssdm.rwcpop.PairDSLmeasure(split='dev', perf_margin=perf_margin)
        val_datasets = [ssdm.rwcpop.InferDS(split='dev')]
        train_infer = [ssdm.rwcpop.InferDS(split='dev')]
    else:
        assert False

    train_loader = DataLoader(
        train_ds, batch_size=None, sampler=PermutationSampler(train_ds, epoch_length),
        num_workers=4, pin_memory=True,
    )
    
    return train_loader, val_datasets, train_infer


class PermutationSampler(Sampler):
    def __init__(self, data_source, epoch_length=2048):
        self.data_source = data_source
        self.epoch_length = epoch_length
        
        self.n = len(data_source)
        self.indices = np.random.permutation(self.n)
        self.i = -1


    def __iter__(self):
        for _ in range(0, self.epoch_length):
            self.i += 1
            if self.i % self.n == 0:
                self.indices = np.random.permutation(self.n)
            yield self.indices[self.i % self.n]


    def __len__(self):
        return self.epoch_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('opt', help='adamw or sgd')
    parser.add_argument('margin', help='sampling score margin')
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date
    opt = kwargs.opt

    margin = float(kwargs.margin) # 0.05

    model_id, ds = list(itertools.product(
        ['MultiResSoftmaxUtil'],
        # ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-jsd', 'all-but-rwcpop', 'all-but-slm', 'all-but-hmx'],
        ['all', 'jsd', 'rwcpop', 'slm', 'hmx', 'all-but-slm'],
        # ['sgd', 'adamw'],
    ))[int(kwargs.config_idx)]

    main(MODEL_ID=model_id, EPOCH=total_epoch, DATE=date, DS=ds, OPT=opt, MARGIN=margin, net=None)
    print('done without failure!')

