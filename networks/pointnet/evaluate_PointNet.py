#!/usr/bin/env python
import os
import os.path
import sys
import h5py
import numpy as np
import math
import random
import torch

from model_pointnet import PointNetDenseCls
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class EvaluatorPointNet:
    def __init__(self, model_path, gpu=0):
        self.device = self.get_device(gpu=gpu)
        self.model = PointNetDenseCls(k=2, feature_transform=True).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.n_points = 1024
        self.data = np.empty((0, self.n_points, 3))
        self.centers = np.empty((0, 2))
        self.corr_n_points = np.empty(0)

    def get_device(self, gpu=0):
        if torch.cuda.is_available():
            device = torch.device(gpu)
        else:
            device = 'cpu'
        return device

    def pc_normalize(self):
        self.data[:, 0:2] = self.data[:, 0:2] - self.centers[:]
        self.data[:, 0:2] = (self.data[:, 0:2] + 40) / 80

    def pc_denormalize(self):
        self.data[:, 0:2] = (self.data[:, 0:2] * 80) - 40
        self.data[:, 0:2] = self.data[:, 0:2] + self.centers[:]

    def transform_input(self, data):
        scans = ["sick_back_right", "sick_back_left", "sick_back_middle"]

        for i in range(len(data)):
            tmp_data = np.empty((0, 3))
            for scan_id, lidar in enumerate(scans):
                sick_data = data[i][lidar][:, :3]
                sick_data[:, 2] = 0.3 * scan_id + 0.4
                tmp_data = np.vstack((tmp_data, sick_data))

            act_npoints = len(tmp_data)
            self.corr_n_points = np.hstack((self.corr_n_points, act_npoints))
            center = np.mean(tmp_data[:, :2], axis=0)
            self.centers = np.vstack((self.centers, center))
            if act_npoints < self.n_points:  # randomly discard
                for _ in range(self.n_points - act_npoints):
                    act_id = random.randint(0, act_npoints - 1)
                    tmp_data = np.vstack((tmp_data, [tmp_data[act_id]]))
            else:  # duplicate/discard frame
                distances = (tmp_data[:, 0] - center[0]) ** 2 + (tmp_data[:, 1] - center[1]) ** 2
                indexes = np.argsort(distances)
                tmp_data = tmp_data[indexes]
                tmp_data = tmp_data[:self.n_points]

            self.data = np.vstack((self.data, [tmp_data]))

    def evaluate(self, data):
        self.transform_input(data)
        self.pc_normalize()

        with torch.no_grad():
            self.model.eval()
            pred, _, _ = self.model(torch.from_numpy(self.data))

        self.pc_denormalize()

        wheels = self.extract_wheels(output)

    def extract_wheels(self, output):
        wheels = []
        for i in range(len(output)):

            act_wheels = []
            pred = output[i]
            pred = pred.view(-1, 2)
            pred = torch.argmax(pred, dim=1)
            pred = pred[: self.corr_n_points[i]]

            points = self.data[i, :self.corr_n_points[i]]
            points = points[pred, :]

            if points_backup.shape[0] < 5:
                wheels.append(act_wheels)
                continue

            dbscan = DBSCAN(eps=0.35, min_samples=3)
            clustering = dbscan.fit_predict(points[:, :2])

            num_clusters = np.unique(clustering)
            num_clusters = len(num_clusters[num_clusters != -1])
            cluster_centers = np.empty((0, 3))
            parent_clusters = np.empty(0)
            cluster_centers_middle = np.empty((0, 3))

            
        return 0

