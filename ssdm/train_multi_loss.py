import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import json, argparse

import ssdm
import ssdm.scanner as scn
import ssdm.salami as slm
from ssdm import harmonix as hmx

# BACKUP, not using this anymore
# DROP_FEATURES=[]

def train(MODEL_ID, EPOCH, DATE, LOSS_TYPE='multi', DS='hmx', LS='score'):
    """DS can be slm and hmx for now, LS can be tau or score"""
    print(MODEL_ID, EPOCH, DATE, LOSS_TYPE, DS, LS)

    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize network based on model_id:
    net = scn.AVAL_MODELS[MODEL_ID]().to(device)

    utility_loss = torch.nn.BCELoss()
    num_layer_loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': [param for name, param in net.named_parameters() if 'head' not in name], 
         'weight_decay': 1e-6},
        {'params': [param for name, param in net.named_parameters() if 'head' in name], 
         'weight_decay': 1e-4}  # Only weight decay for the specified layer
    ])

    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-9, max_lr=1e-4, cycle_momentum=False, mode='triangular', step_size_up=2000
    )

    # setup dataloaders   
    # augmentor = lambda x: scn.time_mask(x, T=100, num_masks=4, replace_with_zero=False, tau=TAU_TYPE)
    augmentor = None
    sample_selector = ssdm.select_samples_using_outstanding_l_score
    if DS == 'slm':
        train_dataset = slm.NewDS(split='train', infer=False, mode='both', transform=augmentor, sample_select_fn=sample_selector)
        val_dataset = slm.NewDS(split='val', infer=False, mode='both', sample_select_fn=sample_selector)
    elif DS == 'hmx':
        train_dataset = hmx.NewDS(split='train', infer=False, mode='both', transform=augmentor, sample_select_fn=sample_selector)
        val_dataset = hmx.NewDS(split='val', infer=False, mode='both', sample_select_fn=sample_selector)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    # pretrain check-up
    net_eval_val = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device, verbose=True)
    best_u_loss = net_eval_val.u_loss.mean()
    best_lvl_loss = net_eval_val.lvl_loss.mean()

    # simple logging
    val_losses = []
    train_losses = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = scn.train_multi_loss(train_loader, net, utility_loss, num_layer_loss, optimizer, lr_scheduler=lr_scheduler, device=device, loss_type=LOSS_TYPE)
        net_eval_val = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device)
        u_loss = net_eval_val.u_loss.mean()
        lvl_loss = net_eval_val.lvl_loss.mean()
        val_loss = (u_loss, lvl_loss)
        
        print(epoch, training_loss, val_loss)
        val_losses.append(val_loss)
        train_losses.append(training_loss)
        
        if u_loss < best_u_loss:
            # update best_loss and save model
            best_u_loss = u_loss
            best_state = net.state_dict()
            torch.save(best_state, f'{DATE}{MODEL_ID}_{DS}{LS}{LOSS_TYPE}_best_util_epoch{epoch}in{EPOCH}')

        if lvl_loss < best_lvl_loss:
            # update best_loss and save model
            best_lvl_loss = lvl_loss
            best_state = net.state_dict()
            torch.save(best_state, f'{DATE}{MODEL_ID}_{DS}{LS}{LOSS_TYPE}_best_nlvl_epoch{epoch}in{EPOCH}')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                          'val_loss': val_losses}
        with open(f'{DATE}{MODEL_ID}_{DS}{LS}{LOSS_TYPE}_{EPOCH}epoch.json', 'w') as file:
            json.dump(trainning_info, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('loss_type', help='what loss? util, nlvl or multi')
    parser.add_argument('dataset', help='Which dataset? slm or hmx')

    kwargs = parser.parse_args()
    train(kwargs.model_id, kwargs.total_epoch, kwargs.date, kwargs.loss_type, kwargs.dataset, 'score')
    print('done without failure!')
