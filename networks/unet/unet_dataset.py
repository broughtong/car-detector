import os
import os.path
import numpy as np
import h5py
import torch


class UNetCarDataset(torch.utils.data.Dataset):
    def __init__(self, path, trn, H=512, W=512, resolution=0.075):
        super().__init__()
        self.path = path
 
        if trn:
            trn = "trn"
        else:
            trn = "val"
        with open(trn + "_list.txt") as f:
            self.data = f.readlines()
    
        self.H = H
        self.W = W
        self.resolution = resolution

    def __getitem__(self, index):
        file = self.data[index]
        file = file.split("_")
        idx = int(file[1])
        file = file[0]

        file = self.path + "/" + file
        point_set = np.copy(np.load(file + ".points.npy", mmap_mode="r")[idx])
        label = np.copy(np.load(file + ".labels.npy", mmap_mode="r")[idx])
        
        indices = ~((point_set[:, 0] == 0.0) & (point_set[:, 1] == 0.0) & (label == -1))
        point_set = point_set[indices, :]
        label = label[indices]

        # convert to image
        im, labels = self.convert_to_image(point_set, label)
        return {'labels': torch.from_numpy(labels).type(torch.int64), 'data': torch.from_numpy(im).type(torch.float32)}

    def convert_to_image(self, point_set, label):
        center = np.mean(point_set[:, :2], axis=0)
        grid = np.zeros((1, self.H, self.W))
        grid_labels = np.zeros((self.H, self.W))
        
        ids = np.ones_like(label, dtype=bool)
        pts = point_set[ids, :2]
        lbls = label[ids]
        rows = np.expand_dims(np.ceil(self.H // 2 - 1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
        cols = np.expand_dims(np.ceil(self.W // 2 - 1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
        ids = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
        rows = rows[ids].astype(int)
        cols = cols[ids].astype(int)
        lbls = lbls[ids]
        grid[0, rows, cols] = 1
        grid_labels[rows[lbls == -1], cols[lbls == -1]] = 1
        grid_labels[rows[lbls != -1], cols[lbls != -1]] = 2
        
        return grid, grid_labels

    def __len__(self):
        return len(self.data)
