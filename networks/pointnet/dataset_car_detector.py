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

    # discard zero positions like a big boy
    mask = (point_set[:, 0] == 0) & (point_set[:, 1] == 0) & (label == 0)
    point_set = point_set[~mask]
    label = label[~mask]

    # add/discard positions based on number of points in the frame
    act_npoints = len(point_set)
    center = np.mean(point_set[:, :2], axis=0)
    if act_npoints < npoints:  # randomly duplicate frames
        choice = np.random.choice(len(label), npoints-act_npoints, replace=True)
        point_set = np.concatenate((point_set, point_set[choice]))
        label = np.concatenate((label, label[choice]))
    else:  # discard frames randomly
        choice = np.random.choice(len(label), npoints, replace=False)
        point_set = point_set[choice, :]
        label = label[choice]

    return point_set, label, center


class CarDetectorDataset():
    def __init__(self, path, num_classes=2, npoints=1024, normalize=True, trn=True):
        self.npoints = npoints
        self.normalize = normalize
        self.num_classes = num_classes
        self.path = path

        if trn:
            self.path = path + "/training"
        else:
            self.path = path + "/testing"

        self.bags = os.listdir(self.path)
        self.cumul_num_frames = np.zeros(len(self.bags), dtype=int)
        self.count_frames()

        print(self.bags)

    def count_frames(self):
        for i in range(len(self.bags)):
            #print(len(os.listdir(self.path + "/" + self.bags[i])), self.bags[i])
            #self.cumul_num_frames[i] = sum(self.cumul_num_frames[:i]) + len(os.listdir(self.path + "/" + self.bags[i]))
            if i > 0:
                self.cumul_num_frames[i] = self.cumul_num_frames[i-1] + len(os.listdir(self.path + "/" + self.bags[i]))
            else:
                self.cumul_num_frames[i] = len(os.listdir(self.path + "/" + self.bags[i]))

    def __getitem__(self, index):

        orig_index = index
        bag_id = np.min(np.nonzero(index < self.cumul_num_frames))
        index = index - (self.cumul_num_frames[bag_id-1] if bag_id > 0 else 0)

        with np.load(self.path + "/" + self.bags[bag_id] + "/" + str(index) + ".npz", mmap_mode="r") as f:
            points = f['pc']
            labels = f['labels']

        points = points[:, :2]
        labels[labels != -1] = 1
        labels[labels == -1] = 0

        point_set, label, center = generate_pc(points, labels, self.npoints)
        point_set[:, 0:2] = point_set[:, 0:2] - center
        if self.normalize:
            point_set = pc_normalize(point_set)

        return point_set, label

    def __len__(self):
        return self.cumul_num_frames[-1]
