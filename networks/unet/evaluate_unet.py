#!/usr/bin/env python
import os
import os.path
import sys
import h5py
import numpy as np
import math
import random
import torch
import pickle

from model_unet import SmallerUnet
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


WIDTH = 1.5
LENGTH = 2.5
DIAGONAL = 3.1
DIVISION_FACTOR = 5
THRESHOLD = 0.3


def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1-x_2)*(x_1-x_2)+(y_1-y_2)*(y_1-y_2))


class EvaluatorPointNet:
    def __init__(self, model_path, gpu=0):
        self.device = self.get_device(gpu=gpu)
        self.model = self.load_model(model_path)
        print("UNet model loaded, device={}".format(self.device))

        self.H = 512
        self.W = 512
        self.resolution = 0.075

        self.data = np.empty((0, 3), dtype=np.float32)
        self.indices = np.empty((0, 2), dtype=np.int8)
        self.grid = torch.zeros([1, 3, self.H, self.W], dtype=torch.float32)
        self.cars_out = []

    def load_model(self, model_path):
        model = SmallerUnet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def get_device(self, gpu=0):
        if torch.cuda.is_available():
            device = torch.device(gpu)
        else:
            device = 'cpu'
        return device

    def transform_input(self, data):
        scans = ["sick_back_right", "sick_back_left", "sick_back_middle"]
        self.data = np.empty((0, 3))
        self.indices = np.empty((0, 2), dtype=np.int8)
        self.grid = torch.zeros([1, 3, self.H, self.W], dtype=torch.float32)
        self.cars_out = []

        for scan_id, lidar in enumerate(scans):
            sick_data = np.array(data[lidar])
            if sick_data.shape[0] < 1:
                continue
            sick_data = sick_data[:, :3]
            sick_data[:, 2] = 0.3 * scan_id + 0.4
            self.data = np.vstack((self.data, sick_data))

        center = np.mean(self.data[:, :2], axis=0)
        for scan_id in range(3):
            pts = self.data[self.data[:, 2] == 0.3 * scan_id + 0.4]
            rows = np.expand_dims(np.ceil(self.H//2-1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
            cols = np.expand_dims(np.ceil(self.W//2-1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
            ids = np.nonzero(np.logical_and.reduce((rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
            rows = rows[ids].astype(int)
            cols = cols[ids].astype(int)
            self.grid[0][scan_id][rows[:, 0], cols[:, 0]] = 1
            self.indices = np.vstack((self.indices, np.hstack((rows, cols))))

        '''plt.figure()
        plt.imshow(self.grid[0].permute(1, 2, 0))
        plt.show()
        input()'''

    def evaluate(self, data):
        self.transform_input(data)

        with torch.no_grad():
            self.model.eval()
            output = self.model(self.grid.to(self.device))
            output = torch.argmax(output, dim=1)
            output = output[0].cpu()

        pred = output[self.indices[:, 0], self.indices[:, 1]]
        pred = torch.squeeze(torch.nonzero(pred)).to('cpu')
        self.extract_wheels(pred)

    def extract_wheels(self, pred, min_wheels=2, use_bumper=True):
        points = self.data[pred]

        dbscan = DBSCAN(eps=0.35, min_samples=3)
        idxs = points[:, 2] == 1
        if np.nonzero(np.logical_not(idxs))[0].shape[0] < 5:
            return

        clustering = dbscan.fit_predict(points[np.logical_not(idxs), :2])
        if np.nonzero(idxs)[0].shape[0] > 0:
            clustering_middle = dbscan.fit_predict(points[idxs, :])
        else:
            clustering_middle = []

        num_clusters = np.unique(clustering)
        num_clusters = len(num_clusters[num_clusters != -1])
        num_clusters_middle = np.unique(clustering_middle)
        num_clusters_middle = len(num_clusters_middle[num_clusters_middle != -1])
        cluster_centers = np.empty((0, 3))
        parent_clusters = np.empty(0)
        cluster_centers_middle = np.empty((0, 3))

        # calculate centers of clusters from middle scanner
        for i in xrange(num_clusters_middle):
            cluster = points[idxs]
            cluster = cluster[np.nonzero(clustering_middle == i)]
            # center = np.mean(cluster, axis=0)
            center = cluster.mean(axis=0)
            cluster_centers_middle = np.vstack((cluster_centers_middle, [center[0], center[1], i]))

        # calculate centers of clusters from right/left scanners
        for i in xrange(num_clusters):
            cluster = points[np.logical_not(idxs)]
            cluster = cluster[np.nonzero(clustering == i)]
            diffs = (cluster - cluster[:, np.newaxis]) ** 2
            dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
            # max_dist = np.max(dists)
            max_dist = dists.max()

            if (1 < max_dist < DIAGONAL) and (len(cluster) > 25):
                sub_cluster_size = int(math.ceil(len(cluster) / DIVISION_FACTOR))
                decision_dists = dists[0]  # array of distances from the first point
                indexes = np.argsort(decision_dists)
                cluster = cluster[indexes]  # sort array of points according to distances
                for j in xrange(DIVISION_FACTOR - 1):
                    tmp = j * sub_cluster_size
                    # center = np.mean(cluster[tmp:tmp + sub_cluster_size, :], axis=0)
                    center = cluster[tmp:tmp + sub_cluster_size, :].mean(axis=0)
                    cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                    parent_clusters = np.append(parent_clusters, i)
                tmp = (DIVISION_FACTOR - 1) * sub_cluster_size
                # center = np.mean(cluster[tmp:, :], axis=0)
                center = cluster[tmp:, :].mean(axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                parent_clusters = np.append(parent_clusters, i)
            else:
                # center = np.mean(cluster, axis=0)
                center = cluster[:, :2].mean(axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                parent_clusters = np.append(parent_clusters, i)

        fragment_ids = np.negative(np.ones(len(cluster_centers), dtype=int))
        fragments = []
        for j in xrange(len(cluster_centers)):
            splitted = False
            if len(np.nonzero(parent_clusters == parent_clusters[j])) > 1:
                splitted = True
            for k in xrange(j + 1, len(cluster_centers)):
                if splitted and len(np.nonzero(parent_clusters == parent_clusters[k])) > 1 and parent_clusters[j] != \
                        parent_clusters[k]:
                    continue
                d = euclidean_distance(cluster_centers[j][0], cluster_centers[k][0], cluster_centers[j][1],
                                       cluster_centers[k][1])
                if abs(d - WIDTH) < THRESHOLD:
                    w1 = cluster_centers[j]
                    w2 = cluster_centers[k]

                    # CALC APPROX POSITIONS OF REMAINING WHEELS ON BOTH SIDES
                    n = [(w1[1] - w2[1]) / d, (w2[0] - w1[0]) / d]  # n=(-u2, u1)/norm - normalized normal vector
                    w3_1 = [w1[0] + LENGTH * n[0], w1[1] + LENGTH * n[1]]
                    w4_1 = [w2[0] + LENGTH * n[0], w2[1] + LENGTH * n[1]]
                    w3_2 = [w1[0] - LENGTH * n[0], w1[1] - LENGTH * n[1]]
                    w4_2 = [w2[0] - LENGTH * n[0], w2[1] - LENGTH * n[1]]

                    # FIND POTENTIAL WHEELS
                    d_values = [float('inf')] * 4  # d3_1_best, d3_2_best, d4_1_best, d4_2_best
                    d_values = np.array(d_values)
                    w_ids = [-1] * 4  # w3_1, w3_2, w4_1, w4_2
                    w_ids = np.array(w_ids)

                    for l in xrange(len(cluster_centers)):
                        if l == j or l == k: continue
                        dw3_1 = euclidean_distance(cluster_centers[l][0], w3_1[0], cluster_centers[l][1], w3_1[1])
                        dw4_1 = euclidean_distance(cluster_centers[l][0], w4_1[0], cluster_centers[l][1], w4_1[1])
                        dw3_2 = euclidean_distance(cluster_centers[l][0], w3_2[0], cluster_centers[l][1], w3_2[1])
                        dw4_2 = euclidean_distance(cluster_centers[l][0], w4_2[0], cluster_centers[l][1], w4_2[1])
                        # one side check
                        if (dw3_1 < THRESHOLD) and (dw3_1 < d_values[0]):
                            d_values[0] = dw3_1
                            w_ids[0] = l
                        elif (dw4_1 < THRESHOLD) and (dw4_1 < d_values[2]):
                            d_values[2] = dw4_1
                            w_ids[2] = l
                        # opposite side check
                        elif (dw3_2 < THRESHOLD) and (dw3_2 < d_values[1]):
                            d_values[1] = dw3_2
                            w_ids[1] = l
                        elif (dw4_2 < THRESHOLD) and (dw4_2 < d_values[3]):
                            d_values[3] = dw4_2
                            w_ids[3] = l

                    # decide which side gives better results if any
                    tmp = np.array(w_ids[:] != -1, dtype=int)
                    d_values[d_values[:] == float('inf')] = 0
                    numw = np.array([tmp[0] + tmp[2] + 2, tmp[1] + tmp[3] + 2])
                    idx = -1

                    if not (numw[0] > 3 or numw[1] > 3):  # found less than four wheels
                        if numw[0] < min_wheels and numw[1] < min_wheels:
                            continue
                        if use_bumper:
                            for l in range(len(cluster_centers_middle)):
                                d1 = euclidean_distance(cluster_centers[j][0], cluster_centers_middle[l][0],
                                                        cluster_centers[j][1], cluster_centers_middle[l][1])
                                d2 = euclidean_distance(cluster_centers[k][0], cluster_centers_middle[l][0],
                                                        cluster_centers[k][1], cluster_centers_middle[l][1])
                                if 0.35 < d1 < 1.2 and 0.35 < d2 < 1.2:
                                    if numw[0] == numw[1]:
                                        idx = np.argmin(
                                            np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                                    else:
                                        idx = np.argmax(np.array(numw))
                                    break
                        else:
                            if numw[0] == numw[1]:
                                idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                            else:
                                idx = np.argmax(np.array(numw))
                    else:
                        # found all four wheels, decide better side - have more wheels/better distances
                        if numw[1] == numw[0]:
                            idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
                        else:
                            idx = np.argmax(np.array(numw))

                    # determine the best global fit
                    if idx != -1:
                        the_best = True
                        act_ids = np.array([j, k, w_ids[idx], w_ids[idx + 2]])
                        act_ids = act_ids[act_ids != -1]
                        act_d = d_values[idx] + d_values[idx + 2] + abs(d - WIDTH)
                        ids = np.unique(fragment_ids[act_ids])
                        ids = ids[ids != -1]
                        # check if the clusters aren't part of other car
                        for l in xrange(len(ids)):
                            if numw[idx] < fragments[ids[l]][0]:
                                the_best = False
                                break
                            elif numw[idx] == fragments[ids[l]][0]:
                                if act_d < fragments[ids[l]][1]:
                                    continue
                                else:
                                    the_best = False
                                    break
                        if not the_best:
                            continue
                        # check if clusters from the same parent cluster aren't part of other car
                        related_parents = np.unique(parent_clusters[act_ids])
                        parents_ids = np.unique(fragment_ids[np.nonzero(np.in1d(parent_clusters, related_parents))])
                        parents_ids = parents_ids[parents_ids != -1]
                        for l in xrange(len(parents_ids)):
                            if numw[idx] < fragments[parents_ids[l]][0]:
                                the_best = False
                                break
                            elif numw[idx] == fragments[parents_ids[l]][0]:
                                if act_d < fragments[parents_ids[l]][1]:
                                    continue
                                else:
                                    the_best = False
                                    break
                        # if all check went gut, discard eventually other cars and add this
                        if the_best:
                            indices = np.nonzero(np.in1d(fragment_ids, ids))
                            fragment_ids[indices] = -1
                            indices = np.nonzero(np.in1d(fragment_ids, parents_ids))
                            fragment_ids[indices] = -1
                            # store car fragment: num of wheels, ideal dist deviation, position of a car
                            if idx == 0:
                                loc = [(w1[0] + w2[0] + w3_1[0] + w4_1[0]) / 4,
                                       (w1[1] + w2[1] + w3_1[1] + w4_1[1]) / 4, np.array(n) * (-2 * idx + 1)]
                            else:
                                loc = [(w1[0] + w2[0] + w3_2[0] + w4_2[0]) / 4,
                                       (w1[1] + w2[1] + w3_2[1] + w4_2[1]) / 4, np.array(n) * (-2 * idx + 1)]
                            fragments.append([numw[idx], act_d, act_ids, loc])
                            fragment_ids[act_ids] = len(fragments) - 1

        plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1], color='black')
        plt.scatter(points[:, 0], points[:, 1], color='red')

        car_idx = np.unique(fragment_ids)
        car_idx = car_idx[car_idx != -1]
        for j in range(len(car_idx)):
            self.cars_out.append([fragments[car_idx[j]][3][0], fragments[car_idx[j]][3][1], fragments[car_idx[j]][3][2]])
            plt.scatter(fragments[car_idx[j]][3][0], fragments[car_idx[j]][3][1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        input()


if __name__ == "__main__":
    model_path = "weight_norm/epoch21.pth"
    data_path = "../gps-square.bag.pickle"

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    evaluator = EvaluatorPointNet(model_path)
    data = data["scans"]
    for i in range(len(data)):
        print("Processing {}".format(i))
        evaluator.evaluate(data[i])
