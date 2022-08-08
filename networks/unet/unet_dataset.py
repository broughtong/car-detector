import os
import os.path
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt


class UNetCarDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_classes=2, trn=True, H=512, W=512, resolution=0.075, num_channels=2):
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.resolution = resolution
        self.num_channels = num_channels

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
            #self.cumul_num_frames[i] = sum(self.cumul_num_frames[:i]) + len(os.listdir(self.path + "/" + self.bags[i]))
            if i > 0:
                self.cumul_num_frames[i] = self.cumul_num_frames[i-1] + len(os.listdir(self.path + "/" + self.bags[i]))
            else:
                self.cumul_num_frames[i] = len(os.listdir(self.path + "/" + self.bags[i]))

    def __getitem__(self, index):
        bag_id = np.min(np.nonzero(index < self.cumul_num_frames))
        index = index - (self.cumul_num_frames[bag_id - 1] if bag_id > 0 else 0)

        with np.load(self.path + "/" + self.bags[bag_id] + "/" + str(index) + ".npz", mmap_mode="r") as f:
            point_set = f['pc']
            label = f['labels']
        
        indices = ~((point_set[:, 0] == 0.0) & (point_set[:, 1] == 0.0) & (label == -1))
        point_set = point_set[indices, :]
        label = label[indices]

        # convert to image
        im, labels = self.convert_to_image(point_set, label)
        return {'labels': torch.from_numpy(labels).type(torch.int64), 'data': torch.from_numpy(im).type(torch.float32)}

    def convert_to_image(self, point_set, label):
        center = np.mean(point_set[:, :2], axis=0)
        grid = np.zeros((self.num_channels, self.H, self.W))
        grid_labels = np.zeros((self.H, self.W))

        ids = np.ones_like(label, dtype=bool)
        pts = point_set[ids, :]
        lbls = label[ids]
        rows = np.expand_dims(np.ceil(self.H // 2 - 1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
        cols = np.expand_dims(np.ceil(self.W // 2 - 1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
        ids = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
        rows = rows[ids].astype(int)
        cols = cols[ids].astype(int)
        lbls = lbls[ids]

        # 1st channel with xy positions
        grid[0, rows, cols] = 1
        # plt.imshow(grid[0])
        # plt.show()

        # optionally 2nd channel with mean intensity values
        if self.num_channels > 1:
            pts = pts[ids]
            grid[1, rows, cols] += np.expand_dims(pts[:, 3], axis=1)
            tmp_grid = np.zeros((self.H, self.W))
            tmp_grid[rows, cols] += 1
            grid[1, rows, cols] = grid[1, rows, cols] / tmp_grid[rows, cols]
            # plt.imshow(grid[1])
            # plt.show()

        # optionally 3rd channel with max of "z"-coordinates in the cell
        if self.num_channels == 3:
            unique_ids = np.unique(np.concatenate((np.expand_dims(rows, axis=1), np.expand_dims(cols, axis=1)), axis=1), axis=0)
            for i in range(unique_ids.shape[0]):
                grid[2, unique_ids[i][0], unique_ids[i][1]] = np.max(pts[np.nonzero(np.logical_and(rows == unique_ids[i][0], cols == unique_ids[i][1][0]))[0], 2])
            # plt.imshow(grid[2])
            # plt.show()

        if self.num_classes == 2:
            grid_labels[rows[lbls != -1], cols[lbls != -1]] = 1
        else:
            grid_labels[rows[lbls == -1], cols[lbls == -1]] = 1
            grid_labels[rows[lbls != -1], cols[lbls != -1]] = 2

        # plt.imshow(grid_labels)
        # plt.show()
        
        return grid, grid_labels

    def __len__(self):
        return self.cumul_num_frames[-1]
