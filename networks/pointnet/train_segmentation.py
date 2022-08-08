from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np

import dataset_car_detector
from model_pointnet import PointNetDenseCls, feature_transform_regularizer

def get_device(gpu=0):  # Manually specify gpu
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device

def get_free_gpu():
    return 0
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:])
    return int(index)  # Returns index of the gpu with the most memory available

def accuracy(prediction, labels_batch, dim=-1):
    pred_index = prediction.argmax(dim)
    return (pred_index == labels_batch).float().mean()

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--lr', type=float, default=0.001, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--optim', default='adam', help="optimizer for backprop")
parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
parser.add_argument('--weight_decay', type=float, default=0, help="weight_decay for optimizer")
parser.add_argument('--weight', type=float, default=1, help="weight of loss for noise label")
parser.add_argument('--numc', type=int, default=2, help="number of classes")
parser.add_argument('--normalize', action='store_true', help="normalize input")
parser.add_argument('--gpu', type=int, default=-1, help="specify gpu")
parser.add_argument('--lanoise', action='store_true', help="train on lanoised data")
parser.add_argument('--num_dimensions', type=int, default=2, help="dimension of the input point cloud")
opt = parser.parse_args()

# init dataset
data_path = "../bags/"
if opt.lanoise:
    data_path += "lanoising"
else:
    data_path += "scans"
print("Data path: ", data_path)

trn_dataset = dataset_car_detector.CarDetectorDataset(num_classes=opt.numc, path=data_path, npoints=1024,
                                                      normalize=opt.normalize, trn=True, num_dimensions=opt.num_dimensions)
val_dataset = dataset_car_detector.CarDetectorDataset(num_classes=opt.numc, path=data_path, npoints=1024,
                                                      normalize=opt.normalize, trn=False, num_dimensions=opt.num_dimensions)
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=opt.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.bs, shuffle=True)

# init device
if opt.gpu != -1:
    device = get_device(opt.gpu)
else:
    device = get_device(get_free_gpu())

# init out directory and logs
try:
    os.makedirs(opt.outf)
except OSError:
    pass
log_file = '%s/log.txt' % opt.outf
f = open(log_file, 'a+')
f.write("bs: {}, n_epochs: {}, lr: {}, feature_trans: {}, optim: {}, momentum: {}, weight_dec: {},weight: {}, "
        "num_classes: {}, normalize: {}, lanoise {}\n".format(opt.bs, opt.nepoch, opt.lr, opt.feature_transform,
                                                              opt.optim, opt.momentum,
                                                              opt.weight_decay, opt.weight, opt.numc, opt.normalize,
                                                              opt.lanoise))
f.close()

# init model
classifier = PointNetDenseCls(k=opt.numc, feature_transform=opt.feature_transform, dev=device, num_dimensions=opt.num_dimensions).to(device)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# init optimization
if opt.optim == 'sgd':
    optimizer = torch.optim.SGD(classifier.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
elif opt.optim == 'amsgrad':
    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, amsgrad=True)
else:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

num_batch = len(trn_dataset) / opt.bs
lr = opt.lr
if opt.numc == 2:
    class_weights = torch.tensor((opt.weight, 1), dtype=torch.float).to(device)
else:
    class_weights = torch.tensor((opt.weight, 1, 1), dtype=torch.float).to(device)

# **** TRAINING PROCEDURE ****
for epoch in range(opt.nepoch):
    print("Epoch {} started".format(epoch))
    loss_list = []
    loss_val = []
    classifier.train()

    print()
    for i, data in enumerate(trn_loader, 0):
        print("Batch {}".format(i), end="\r")
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(device, dtype=torch.float32), target.to(device, torch.int64)

        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, opt.numc)
        target = target.view(-1, 1)[:, 0]

        loss = F.nll_loss(pred, target, weight=class_weights)

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat, device) * 0.001
        loss.backward()
        loss_list.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        classifier.eval()
        output_list = []
        label_list = []
        for j, data in enumerate(val_loader):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.to(device, dtype=torch.float32), target.to(device, torch.int64)

            pred, _, _ = classifier(points)
            pred = pred.view(-1, opt.numc)
            target = target.view(-1, 1)[:, 0]

            loss = F.nll_loss(pred, target, weight=class_weights)
            loss_val.append(loss.item())

            output_list.append(pred)
            label_list.append(target)

        acc = accuracy(torch.cat(output_list), torch.cat(label_list))

    print(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (i + 1):.3f} \t Val Loss: {sum(loss_val) / (j + 1):.3f} Acc: {acc:.3f}')
    f = open(log_file, 'a+')
    f.write("Epoch {} started\n".format(epoch))
    f.write(f'Epoch: {epoch:03d} \t Trn Loss: {sum(loss_list) / (i + 1):.3f} \t Val Loss: {sum(loss_val) / (j + 1):.3f} Acc: {acc:.3f}\n')
    f.close()
    torch.save(classifier.state_dict(), '%s/epoch_%d.pth' % (opt.outf, epoch))
