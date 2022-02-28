import os
import os.path
import numpy as np
import sys
import h5py
import random


def pc_normalize(pc):
    # center = np.mean(pc[:, :2], axis=0)
    # pc[:, 0:2] = pc[:, 0:2] - center
    pc[:, 0:2] = (pc[:, 0:2] + 40) / 80

    return pc


def generate_pc(point_set, label, npoints):
    # discard zero positions
   
    # id = 0
    # while id != len(label):
    #     if point_set[id][0] == 0.0 and point_set[id][1] == 0.0 and label[id] == 0:
    #         point_set = np.delete(point_set, id, 0)
    #         label = np.delete(label, id)
    #         continue
    #     id += 1

    # discard zero positions like a big boy
    mask = (point_set[:, 0] == 0) & (point_set[:, 1] == 0) & (label == 0)
    point_set = point_set[~mask]
    label = label[~mask]

    # add/discard positions based on number of points in the frame
    act_npoints = len(point_set)
    center = np.mean(point_set[:, :2], axis=0)
    # print(act_npoints, npoints)
    if act_npoints < npoints:  # randomly duplicate
        # print("Discarding")
        for _ in range(npoints - act_npoints):
            act_id = random.randint(0, act_npoints - 1)
            point_set = np.vstack((point_set, [point_set[act_id]]))
            label = np.hstack((label, label[act_id]))
    else:  # duplicate/discard frame
        # resample and choose accurate quantity of point set
        choice = np.random.choice(len(label), npoints, replace=True)
        point_set = point_set[choice, :]
        label = label[choice]
        # print("Duplicating")
        # distances = (point_set[:, 0] - center[0]) ** 2 + (point_set[:, 1] - center[1]) ** 2
        # indexes = np.argsort(distances)
        # label = label[indexes]
        # point_set = point_set[indexes]

    return point_set, label, center


class CarDetectorDataset():
    def __init__(self, path, num_classes=2, npoints=1024, normalize=True, trn=True):
        self.npoints = npoints
        self.normalize = normalize
        self.num_classes = num_classes
        self.path = path

        if trn:
            trn = "trn"
        else:
            trn = "val"
        with open(trn + "_list.txt") as f:
            self.data = f.readlines()

    def __getitem__(self, index):
        file = self.data[index]
        file = file.split("_")
        idx = int(file[1])
        file = file[0]

        # with h5py.File(self.path + "/" + file, 'r') as f:
        #     points = f["data"][idx]
        #     labels = f["labels"][idx]
       
        file = self.path + "/" + file
        points = np.copy(np.load(file + ".points.npy", mmap_mode="r")[idx])
        labels = np.copy(np.load(file + ".labels.npy", mmap_mode="r")[idx])

        labels[labels != -1] = 1
        labels[labels == -1] = 0

        # print("Before {}".format(len(points)))
        point_set, label, center = generate_pc(points, labels, self.npoints)
        # print("After {}".format(len(point_set)))
        point_set[:, 0:2] = point_set[:, 0:2] - center
        if self.normalize:
            point_set = pc_normalize(point_set)
        # print("Final {}".format(len(point_set)))

        return point_set, label

    def __len__(self):
        return len(self.data)
