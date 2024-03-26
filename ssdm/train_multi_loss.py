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

def train(MODEL_ID, EPOCH, DATE, TAU_TYPE='both', DS='hmx', LS='score'):
    """DS can be slm and hmx for now, LS can be tau or score"""
    print(MODEL_ID, EPOCH, DATE, TAU_TYPE, DS, LS)

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
        optimizer, base_lr=1e-9, max_lr=1e-4, cycle_momentum=False, mode='triangular', step_size_up=1000
    )

    # setup dataloaders   
    augmentor = lambda x: scn.time_mask(x, T=100, num_masks=4, replace_with_zero=False, tau=TAU_TYPE)
    sample_selector = ssdm.select_samples_using_outstanding_l_score
    if DS == 'slm':
        train_dataset = slm.DS('train', infer=False, mode=TAU_TYPE, transform=augmentor, sample_select_fn=sample_selector)
        val_dataset = slm.DS('val', infer=False, mode=TAU_TYPE, sample_select_fn=sample_selector)
    elif DS == 'hmx':
        train_dataset = hmx.DS(split='train', infer=False, mode=TAU_TYPE, transform=augmentor, sample_select_fn=sample_selector)
        val_dataset = hmx.DS(split='val', infer=False, mode=TAU_TYPE, sample_select_fn=sample_selector)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

    # pretrain check-up
    net_eval_val = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device, verbose=True)
    best_loss = net_eval_val.loss.mean()

    # simple logging
    val_losses = []
    # val_accus = []
    train_losses = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = scn.train_multi_loss(train_loader, net, utility_loss, num_layer_loss, optimizer, lr_scheduler=lr_scheduler, device=device)
        net_eval_val = scn.net_eval_multi_loss(val_dataset, net, utility_loss, num_layer_loss, device)

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
            torch.save(best_state, f'{DATE}{MODEL_ID}_{DS}{LS}{TAU_TYPE}_epoch{epoch}')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                        'val_loss': val_losses,
                        # 'val_accu': val_accus
                        }
        with open(f'{DATE}{MODEL_ID}_{DS}{LS}{TAU_TYPE}_{EPOCH}epoch.json', 'w') as file:
            json.dump(trainning_info, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('tau_type', help='which tau? rep or loc')
    parser.add_argument('dataset', help='Which dataset? slm or hmx')
    # parser.add_argument('learning_signal', help='tau or score')

    kwargs = parser.parse_args()
    # print(kwargs)
    for l_sig in ['score']:
        train(kwargs.model_id, kwargs.total_epoch, kwargs.date, kwargs.tau_type, kwargs.dataset, l_sig)
    print('done without failure!')
