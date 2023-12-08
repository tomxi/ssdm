import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import json, argparse

import ssdm.scanner as scn

# BACKUP, not using this anymore
DROP_FEATURES=[]

def train(MODEL_ID, EPOCH, DATE, TAU_TYPE):
    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize network based on model_id:
    net = scn.AVAL_MODELS[MODEL_ID]().to(device)
    net.to(device)

    # L2 Regularization on specified layers:
    layer_with_weight_decay = net.rep_predictor

    # training tools
    criterion = torch.nn.BCELoss()
    optimizer = optim.AdamW([
        {'params': [param for name, param in net.named_parameters() if 'rep_predictor' not in name], 
         'weight_decay': 1e-6},
        {'params': layer_with_weight_decay.parameters(), 
         'weight_decay': 1e-5}  # Only weight decay for the specified layer
    ])

    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                            base_lr=1e-9,
                                            max_lr=1e-4,
                                            cycle_momentum=False,
                                            mode='triangular',
                                            step_size_up=1000)

    # setup dataloaders
    train_dataset = scn.SlmDS('train', mode=TAU_TYPE, drop_features=DROP_FEATURES)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_dataset = scn.SlmDS('val', mode=TAU_TYPE, drop_features=DROP_FEATURES)

    # pretrain check-up
    net_eval_val = scn.net_eval(val_dataset, net, criterion, device, verbose=True)
    best_loss = net_eval_val.loss.mean()

    # simple logging
    val_losses = []
    val_accus = []
    train_losses = []

    for epoch in tqdm(range(int(EPOCH))):
        training_loss = scn.train_epoch(train_loader, net, criterion, optimizer, lr_scheduler=lr_scheduler, device=device)
        net_eval_val = scn.net_eval(val_dataset, net, criterion, device)

        loss = net_eval_val.loss.mean()
        accu = (net_eval_val.pred == net_eval_val.label).mean()
        print(epoch, training_loss, loss, accu)
        val_losses.append(loss)
        val_accus.append(accu)
        train_losses.append(training_loss)
        
        if loss < best_loss:
            # update best_loss and save model
            best_loss = loss
            best_state = net.state_dict()
            torch.save(best_state, f'{MODEL_ID}{DATE}_epoch{epoch}')
        
        # save simple log as json
        trainning_info = {'train_loss': train_losses,
                        'val_loss': val_losses,
                        'val_accu': val_accus}
        with open(f'{DATE}{MODEL_ID}_l2-5_{EPOCH}epoch.json', 'w') as file:
            json.dump(trainning_info, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Tau hats')
    parser.add_argument('model_id', help='see scanner.AVAL_MODELS')
    parser.add_argument('total_epoch', help='total number of epochs to train')
    parser.add_argument('date', help='just a marker really, can be any text but mmdd is the intension')
    parser.add_argument('tau_type', help='which tau? rep or loc')

    kwargs = parser.parse_args()
    # print(kwargs)
    train(kwargs.model_id, kwargs.total_epoch, kwargs.date, kwargs.tau_type)
    print('done without failure!')
