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


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    labels = f['labels'][:]
    return data, labels


def generate_pc(point_set, label, npoints):
    # discard zero positions
    id = 0
    while id != len(label):
        if point_set[id][0] == 0.0 and point_set[id][1] == 0.0 and point_set[id][2] == 0.0 and label[id] == -1:
            point_set = np.delete(point_set, id, 0)
            label = np.delete(label, id)
            continue
        id += 1

    # add/discard positions based on number of points in the frame
    act_npoints = len(point_set)
    center = np.mean(point_set[:, :2], axis=0)
    if act_npoints < npoints:  # randomly discard
        for _ in range(npoints - act_npoints):
            act_id = random.randint(0, act_npoints - 1)
            point_set = np.vstack((point_set, [point_set[act_id]]))
            label = np.hstack((label, label[act_id]))
    else:  # duplicate/discard frame
        distances = (point_set[:, 0] - center[0]) ** 2 + (point_set[:, 1] - center[1]) ** 2
        indexes = np.argsort(distances)
        label = label[indexes]
        point_set = point_set[indexes]

    return point_set, label, center


class WheelDataset():
    def __init__(self, path, num_classes=2, npoints=1024, classification=False, normalize=True):
        self.npoints = npoints
        self.classification = classification
        self.normalize = normalize
        self.num_classes = num_classes

        self.data, self.labels = load_h5(path)
        print("Loaded data: ", self.data.shape, self.labels.shape)

    def __getitem__(self, index):
        point_set = self.data[index].astype(np.float32)
        label = self.labels[index].astype(np.int32)
        label[label != -1] = 1
        label[label == -1] = 0

        point_set, label, center = generate_pc(point_set, label, self.npoints)
        point_set[:, 0:2] = point_set[:, 0:2] - center
        if self.normalize:
            point_set = pc_normalize(point_set)

        # resample and choose accurate quantity of point set
        choice = np.random.choice(len(label), self.npoints, replace=True)
        point_set = point_set[choice, :]
        label = label[choice]

        return point_set, label

    def __len__(self):
        return len(self.labels)
