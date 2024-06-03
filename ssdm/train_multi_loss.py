import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import json, argparse, itertools

import ssdm
import ssdm.scanner as scn
# import ssdm.salami as slm
# from ssdm import harmonix as hmx

# No Dropout

def train(MODEL_ID='MultiRes', EPOCH=7, DATE=None, LOSS_TYPE='multi', DS='new-hmx', WD='1', LR='cyclic', LAPNORM='random_walk', AUGMENT=False, net=None):
    """
    """
    aug_str = 'aug' if AUGMENT else ''
    short_xid_str = f'{MODEL_ID}_{DS}wd{WD}lr{LR}{aug_str}'
    experiment_id_str = f'{DATE}{short_xid_str}{LOSS_TYPE}{LAPNORM}{EPOCH}'
    print(experiment_id_str)
    
    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if net is None:
        # Initialize network based on model_id:
        net = scn.AVAL_MODELS[MODEL_ID]()
    net.to(device)

    # setup dataloaders
    if AUGMENT:
        augmentor = scn.perm_layer
    else:
        augmentor = None

    if DS == 'new-hmx':
        train_dataset = ssdm.base.HmxDS(split='train', infer=False, device=device, nlvl_metric='l', transform=augmentor)
        val_dataset = ssdm.base.HmxDS(split='val', infer=False, device=device, nlvl_metric='l')
    elif DS == 'new-hmx-flat':
        train_dataset = ssdm.base.HmxDS(split='train', infer=False, device=device, nlvl_metric='pfc', transform=augmentor)
        val_dataset = ssdm.base.HmxDS(split='val', infer=False, device=device, nlvl_metric='pfc')
    else:
        print('bad DS')

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    # training details
    utility_loss = torch.nn.BCELoss()
    # num_layer_loss = torch.nn.MSELoss()
    num_layer_loss = scn.NLvlLoss()

    
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if 'bias' not in n],
            "weight_decay": float(WD),
        },
        {
            "params": [p for n, p in net.named_parameters() if 'bias' in n],
            "weight_decay": 0.0,
        },
    ]

    if LR == 'cyclic':
        optimizer = optim.AdamW(grouped_parameters)

        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-6, max_lr=1e-4, cycle_momentum=False, mode='triangular2', step_size_up=2000
        )
    else:
        optimizer = optim.AdamW(grouped_parameters, lr=float(LR))
        lr_scheduler = None

    ### Train loop:
    # pretrain check-up
    net_eval_val, nlvl_outputs = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device, verbose=True)
    best_u_loss = net_eval_val.u_loss.mean()
    best_lvl_loss = net_eval_val.loc[net_eval_val.label == 1].lvl_loss.mean()

    # simple logging
    val_losses = []
    train_losses = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = scn.train_multi_loss(train_loader, net, utility_loss, num_layer_loss, optimizer, lr_scheduler=lr_scheduler, device=device, loss_type=LOSS_TYPE)
        
        net_eval_val, nlvl_outputs_val = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device)
        net_eval_val.to_pickle(f'{short_xid_str}_val_perf_e{epoch}.pkl')
        nlvl_outputs_val.to_pickle(f'{short_xid_str}_val_nlvl_e{epoch}.pkl')
        
        net_eval_train, nlvl_outputs_train = scn.net_eval_multi_loss(train_dataset, net, utility_loss, num_layer_loss, device)
        net_eval_train.to_pickle(f'{short_xid_str}_train_perf_e{epoch}.pkl')
        nlvl_outputs_train.to_pickle(f'{short_xid_str}_train_nlvl_e{epoch}.pkl')
        
        u_loss = net_eval_val.u_loss.mean()
        lvl_loss = net_eval_val.loc[net_eval_val.label == 1].lvl_loss.mean()
        val_loss = (u_loss, lvl_loss)
        
        print(f'\n Post Epoch {epoch}:'), 
        print(f'\t Train: util BCE {training_loss[0]:.4f}, nlvl Loss {training_loss[1]:.4f}')
        print(f'\t Valid: util BCE {val_loss[0]:.4f}, nlvl Loss {val_loss[1]:.4f}, on both pos and neg {net_eval_val.lvl_loss.mean():.4f}')
        
        val_losses.append(val_loss)
        train_losses.append(training_loss)
        
        if u_loss < best_u_loss:
            # update best_loss and save model
            best_u_loss = u_loss
            best_state = net.state_dict()
            torch.save(best_state, f'{short_xid_str}_e{epoch}u{u_loss:.3f}l{lvl_loss:.2f}_best_util')

        if lvl_loss < best_lvl_loss:
            # update best_loss and save model
            best_lvl_loss = lvl_loss
            best_state = net.state_dict()
            torch.save(best_state, f'{short_xid_str}_e{epoch}u{u_loss:.3f}l{lvl_loss:.2f}_best_nlvl')

        if epoch % 10 == 0:
            # save every 5 epoch regardless
            best_state = net.state_dict()
            torch.save(best_state, f'{short_xid_str}_e{epoch}u{u_loss:.3f}l{lvl_loss:.2f}')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                          'val_loss': val_losses,
                          }
        with open(f'{short_xid_str}.json', 'w') as file:
            json.dump(trainning_info, file)
    
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    # parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    # parser.add_argument('loss_type', help='what loss? util, nlvl or multi')
    # parser.add_argument('dataset', help='Which dataset? slm or hmx')
    # parser.add_argument('wd', help='multiplier for weight decay parameter for optimizer')
    # parser.add_argument('ls', help='learning signal, tau or score?')
    # parser.add_argument('lap_norm', help='how to normalize laplacian, symmetrical or random_walk?')
    
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')
    
    

    kwargs = parser.parse_args()
    total_epoch = int(kwargs.total_epoch)
    date = kwargs.date

    model_id = 'MultiRes'
    dataset = 'new-hmx'
    wd = 10
    lr = 1e-4
    loss_type = 'multi'
    lap_norm = 'random_walk'
    aug = False


    overwrite_config_list = list(itertools.product(
        ['MultiResB', 'MultiResC', 'MultiResD'],
        ['new-hmx', 'new-hmx-flat'],
        [10],
        ['cyclic'],
    ))
    model_id, dataset, wd, lr = overwrite_config_list[int(kwargs.config_idx)]

    train(model_id, total_epoch, date, loss_type, dataset, wd, lr, lap_norm, aug, None)
    print('done without failure!')
