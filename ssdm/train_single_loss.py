import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import json, argparse, itertools

import ssdm
import ssdm.scanner as scn
import ssdm.salami as slm
from ssdm import harmonix as hmx

# BACKUP, not using this anymore
# DROP_FEATURES=[]

def train(MODEL_ID, EPOCH, DATE, LOSS_TYPE='multi', DS='slm', WDM='1e-5', LS='score', LAPNORM='random_walk'):
    """
    MODEL_ID
    DS can be slm and hmx for now
    LOSS_TYPE: mutil, util, nlvl
    WDM: weight decay multiplyer
    
    """
    experiment_id_str = f'{DATE}{MODEL_ID}_{DS}{WDM}{LOSS_TYPE}{LS}{LAPNORM}{EPOCH}'
    print(experiment_id_str)

    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize network based on model_id:
    net = scn.AVAL_MODELS[MODEL_ID]().to(device)

    utility_loss = torch.nn.BCELoss()
    optimizer = optim.AdamW(net.parameters(), weight_decay=float(WDM))

    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-9, max_lr=1e-4, cycle_momentum=False, mode='triangular', step_size_up=8000
    )

    # setup dataloaders   
    # augmentor = lambda x: scn.time_mask(x, T=100, num_masks=4, replace_with_zero=False, tau=TAU_TYPE)
    augmentor = None
    if LS == 'score':
    #     sample_selector = ssdm.select_samples_using_outstanding_l_score
        sample_selector = ssdm.sel_samp_l
    elif LS == 'tau':
        sample_selector = ssdm.select_samples_using_tau_percentile
    else:
        print('bad learning signal: score or tau')
    

    if DS == 'slm':
        train_dataset = slm.NewDS(split='train', infer=False, mode='both', transform=augmentor, lap_norm=LAPNORM, sample_select_fn=sample_selector, beat_sync=True)
        val_dataset = slm.NewDS(split='val', infer=False, mode='both', lap_norm=LAPNORM, sample_select_fn=sample_selector, beat_sync=True)
    elif DS == 'hmx':
        train_dataset = hmx.NewDS(split='train', infer=False, mode='both', transform=augmentor, lap_norm=LAPNORM, sample_select_fn=sample_selector, beat_sync=True)
        val_dataset = hmx.NewDS(split='val', infer=False, mode='both', lap_norm=LAPNORM, sample_select_fn=sample_selector, beat_sync=True)
    else:
        print('bad DS')

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    # pretrain check-up
    net_eval_val = scn.net_eval(val_dataset, net, utility_loss, device, verbose=True)
    best_loss = net_eval_val.loss.mean()

    # simple logging
    val_losses = []
    train_losses = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = scn.train_epoch(train_loader, net, utility_loss, optimizer, lr_scheduler=lr_scheduler, device=device)
        net_eval_val = scn.net_eval(val_dataset, net, utility_loss, device)

        loss = net_eval_val.loss.mean()
        # accu = (net_eval_val.pred == net_eval_val.label).mean()
        print(epoch, training_loss, loss)
        val_losses.append(loss)
        # val_accus.append(accu)
        train_losses.append(training_loss)
        
        if loss < best_loss:
            # update best_loss and save model
            best_loss = loss
            best_state = net.state_dict()
            torch.save(best_state, f'{experiment_id_str}')

        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                          'val_loss': val_losses,
                          }
        with open(f'{experiment_id_str}epoch.json', 'w') as file:
            json.dump(trainning_info, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    # parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    # parser.add_argument('total_epoch', help='total number of epochs to train')
    # parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    # parser.add_argument('loss_type', help='what loss? util, nlvl or multi')
    # parser.add_argument('dataset', help='Which dataset? slm or hmx')
    # parser.add_argument('wd', help='multiplier for weight decay parameter for optimizer')
    # parser.add_argument('ls', help='learning signal, tau or score?')
    # parser.add_argument('lap_norm', help='how to normalize laplacian, symmetrical or random_walk?')
    
    parser.add_argument('config_idx', help='which config to use. it will get printed, but see .py file for the list itself')
    
    config_list = list(itertools.product(
        ['EvecNet'],
        ['tau'],
    ))

    model_id, dataset, ls = config_list[int(parser.parse_args().config_idx)]
    total_epoch = 30
    date = 202404011
    loss_type = 'single'
    wd = 2e-3
    # ls = 'score'
    lap_norm = 'random_walk'

    train(model_id, total_epoch, date, loss_type, dataset, wd, ls, lap_norm)
    print('done without failure!')
