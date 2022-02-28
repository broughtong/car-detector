import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
#from sklearn.model_selection import train_test_split
from model_unet import SmallerUnet, ZDUNet
from unet_dataset import UNetCarDataset


parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=16, help='input batch size')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--lr', type=float, default=0.0001, help="dataset path")
parser.add_argument('--optim', default='adam', help="optimizer for backprop")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay for optimizer")
parser.add_argument('--weight', type=float, default=1, help="weight of loss for noise label")
parser.add_argument('--gpu', type=int, default=0, help="specify gpu")
parser.add_argument('--start_epoch', type=int, default=0, help="specify the first epoch number to be persisted")
parser.add_argument('--dataset', type=str, help="specify dataset")
parser.add_argument('--lanoise', action='store_true', help="train on lanoised data")
opt = parser.parse_args()


def save_model(model, destination):
    torch.save(model.state_dict(), destination, _use_new_zipfile_serialization=False)


def load_model():
    model = ZDUNet(num_classes=3)
    model.load_state_dict(torch.load(opt.model, map_location='cpu'))
    return model


def get_device(gpu=0):  # Manually specify gpu
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(gpu)
    else:
        device = 'cpu'

    return device


# init output directory, logs
try:
    os.makedirs(opt.outf)
except OSError:
    pass
log_file = '%s/log.txt' % opt.outf
f = open(log_file, 'a+')
f.write("bs: {}, n_epochs: {}, lr: {}, optim: {}, momentum: {}, weight_dec: {},weight: {}, lanoise: {}\n".format(opt.bs, opt.nepoch, opt.lr, opt.optim,opt.momentum, opt.weight_decay, opt.weight, opt.lanoise))
f.close()

device = get_device(opt.gpu)

# new datasets
data_path = "/datafast/janota/lanoising-ts4-"
if opt.lanoise:
	data_path += "scans"
else:
	data_path += "lanoising"

trn_dataset = UNetCarDataset(path=data_path, trn=True)
val_dataset = UNetCarDataset(path=data_path, trn=False)

trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=opt.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)


# init model
if opt.model != '':
    model = load_model().to(device)
else:
    model = ZDUNet(num_classes=3).to(device)

# init optimizer
if opt.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

#class_weight = torch.tensor((opt.weight, 1.), dtype=torch.float32).to(device)
#class_weight = torch.tensor((0.0004, 0.15, 1.), dtype=torch.float32).to(device)
class_weight = torch.tensor((0.0000001, 0.5, 1.), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)

shape = 512 * 512
#shape = 128 * 128

"""def weight_loss(prediction, label):
    #print(prediction.shape, label.shape)
    #torch.gather(data, 0, t[:, None])

    label = label.view((shape * label.shape[0], 1))
    prediction = prediction.view((shape * prediction.shape[0], -1))
    
    weights = class_weight[label]
    #losses = torch.log(torch.gather(prediction, 0, label[:, None])) * weights
    print(label)
    print(prediction)
    print(weights)
    print(torch.gather(prediction, 0, label))
    losses = (- torch.gather(prediction, 0, label) + torch.log(torch.sum(torch.exp(prediction), axis=1))) * weights
    print(losses)
    return torch.sum(losses) / torch.sum(weights)
"""


# **** TRAINING LOOP ****
print("Training started")
for epoch in range(opt.start_epoch, opt.nepoch):
    print("Epoch: {} started".format(epoch))
    model.train()
    loss_list = []
    loss_val = []
    TP_t = FP_t = TN_t = FN_t = 0
    CB = CN = CC = NB = NN = NC = BB = BN = BC = 0
    start = time.time()
    start_iter = time.time()
    total_batch_num = len(trn_loader)
    # TRAINING
    for it, batch in enumerate(trn_loader):
        if it % 100 == 0:
            now = time.time()
            print(f".  Iteration {it} / {total_batch_num} - elapsed {int(now - start_iter)/60} min, total {int(now - start)/60} min")
            start_iter = now
        input_data = batch['data'].to(device).requires_grad_()
        label = batch['labels'].to(device)
        
        #print("predicting batch")
        predictions = model(input_data)
        loss = criterion(predictions, label)
        #loss = weight_loss(predictions, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.item())
        
        pred = torch.argmax(predictions, dim=1).to(device)
        # conf_vector = pred/label
        # TP_t += torch.sum(conf_vector == 1).item()  # pred=1 and label=1
        # FP_t += torch.sum(conf_vector == float('inf')).item()  # pred=1 and label=0
        # TN_t += torch.sum(torch.isnan(conf_vector)).item()  # pred=0 and label=0
        # FN_t += torch.sum(conf_vector == 0).item()  # pred=0 and label=1

        CB += torch.sum((pred == 2) & (label == 0)).item()
        CN += torch.sum((pred == 2) & (label == 1)).item()
        CC += torch.sum((pred == 2) & (label == 2)).item()
        NB += torch.sum((pred == 1) & (label == 0)).item()
        NN += torch.sum((pred == 1) & (label == 1)).item()
        NC += torch.sum((pred == 1) & (label == 2)).item()
        BB += torch.sum((pred == 0) & (label == 0)).item()
        BN += torch.sum((pred == 0) & (label == 1)).item()
        BC += torch.sum((pred == 0) & (label == 2)).item()
        

        #print(f"batch took {(time.time() - start)} s")     
        #print(TP_t, FP_t, TN_t, FN_t)
    #print(TP_t, FP_t, TN_t, FN_t)#, TP_t/(TP_t+FP_t), TP_t/(TP_t+FN_t))
    
    trn_cnf = f"training: CB = {CB}, CN = {CN}, CC = {CC}, NB = {NB}, NN = {NN}, NC = {NC}, BB = {BB}, BN = {BN}, BC = {BC}, TOT = {CB + CN + CC + NB + NN + NC + BB + BN + BC}"
    print(trn_cnf)
    
    # VALIDATION
    TP = FP = TN = FN = 0
    CB = CN = CC = NB = NN = NC = BB = BN = BC = 0
    with torch.no_grad():
        model.eval()
        # confusion_matrix = np.zeros((2, 2))
        for it_val, batch in enumerate(val_loader):
            data = batch['data'].to(device)
            label = batch['labels'].to(device)      # NxHxW

            output = model(data)    # Nx2xHxW
            # loss = torch.nn.functional.cross_entropy(output, label, weight=torch.tensor((opt.weight,1), dtype=torch.float32))
            loss = torch.nn.functional.cross_entropy(output, label, weight=class_weight)
            #loss = weight_loss(output, label)
            loss_val.append(loss.item())

            pred = torch.argmax(output, dim=1).to(device)
            # conf_vector = pred / label
            # TP += torch.sum(conf_vector == 1).item()     # pred=1 and label=1
            # FP += torch.sum(conf_vector == float('inf')).item()      # pred=1 and label=0
            # TN += torch.sum(torch.isnan(conf_vector)).item()     # pred=0 and label=0
            # FN += torch.sum(conf_vector == 0).item()     # pred=0 and label=1'''
            
            CB += torch.sum((pred == 2) & (label == 0)).item()
            CN += torch.sum((pred == 2) & (label == 1)).item()
            CC += torch.sum((pred == 2) & (label == 2)).item()
            NB += torch.sum((pred == 1) & (label == 0)).item()
            NN += torch.sum((pred == 1) & (label == 1)).item()
            NC += torch.sum((pred == 1) & (label == 2)).item()
            BB += torch.sum((pred == 0) & (label == 0)).item()
            BN += torch.sum((pred == 0) & (label == 1)).item()
            BC += torch.sum((pred == 0) & (label == 2)).item()

    #print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    val_cnf = f"validation: CB = {CB}, CN = {CN}, CC = {CC}, NB = {NB}, NN = {NN}, NC = {NC}, BB = {BB}, BN = {BN}, BC = {BC}, TOT = {CB + CN + CC + NB + NN + NC + BB + BN + BC}"
    print(val_cnf)

    print(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.12f} \t')
    print(f"epoch took {(time.time() - start) / 60} mins")     
    # print(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.3f} \t Val Loss: {sum(loss_val) / (it_val + 1):.3f}')
    '''f = open(log_file, 'a+')
    f.write("Epoch {} started\n".format(epoch))
    f.write(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.3f} \t Val Loss: {sum(loss_val) / (it_val + 1):.3f}\n')
    f.write("TP: {}, FP: {}, TN: {}, FN: {}, P: {}, R: {}\n".format(TP_t, FP_t, TN_t, FN_t, TP_t/(TP_t+FP_t), TP_t/(TP_t+FN_t)))
    f.write("TP: {}, FP: {}, TN: {}, FN: {}, P: {}, R: {}\n".format(TP, FP, TN, FN, TP/(TP+FP), TP/(TP+FN)))
    f.close()'''
    with open(log_file, "a+") as f:
        f.write(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (it + 1):.5f} \t Val Loss: {sum(loss_val) / (it_val + 1):.5f}\n')
        f.write(trn_cnf)
        f.write(val_cnf)
    save_model(model, opt.outf+'/'+'epoch'+str(epoch)+'.pth')
