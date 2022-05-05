#!/usr/bin/python
# import utils
import pickle
import math
import sys
import os
import torch
import multiprocessing
from os import devnull
from sklearn.cluster import DBSCAN
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import random
from model_pointnet import PointNetDenseCls
import copy

datasetPath = "../../data/results/lanoise"
outputPath = "../../data/results/pn-eval"
os.makedirs(outputPath, exist_ok=True)

WIDTH = 1.5
LENGTH = 2.5
DIAGONAL = 3.1
DIVISION_FACTOR = 5
THRESHOLD = 0.3

lidar_dict = {'sick_back_left':0, 'sick_back_right':1, 'sick_front':2, 'sick_back_middle':3}


def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))


def pc_normalize(pc, center):
    pc[:, 0:2] = pc[:, 0:2] - center
    pc[:, 0:2] = (pc[:, 0:2] + 40) / 80
    return pc
    

def pc_denormalize(pc):
    pc[:, 0:2] = (pc[:, 0:2] * 80) - 40
    return pc
    

def get_device(gpu=0):
        if torch.cuda.is_available():
            device = torch.device(gpu)
        else:
            device = 'cpu'
        return device


def prepare_pc(point_set, npoints, normalize=True):
    act_npoints = len(point_set)
    center = np.mean(point_set[:, :2], axis=0)

    if act_npoints < npoints:
        for _ in range(npoints - act_npoints):
            act_id = random.randint(0, act_npoints - 1)
            point_set = np.vstack((point_set, [point_set[act_id]]))
    else:
        choice = np.random.choice(len(point_set), npoints, replace=True)
        point_set = point_set[choice, :]

    # point_set[:, 0:2] = point_set[:, 0:2] - center
    # if normalize:
    #    point_set = pc_normalize(point_set)

    return point_set, center


@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


class Inference(multiprocessing.Process):
    def __init__(self, path, filename, model_path, num_classes=2, gpu=0):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path
        self.num_classes = num_classes
        self.device = get_device(gpu=gpu)

    def load_model(self, model_path):
        model = PointNetDenseCls(k=2, feature_transform=False)
        print(model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def run(self):
        self.model = self.load_model(model_path).to(self.device)

        print("Process spawned for file %s" % (self.filename), flush=True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        self.inference()

        with open(os.path.join(outputPath, self.filename), "wb") as f:
             pickle.dump(self.data, f, protocol=2)

    def inference(self):
        keys = list(self.data["scans"][0].keys())
        self.data["pointnet"] = []

        for i in range(len(self.data["scans"])):
            frame = np.empty((0, 3))
            for j in range(len(keys)):
                scan = np.array(self.data["scans"][i][keys[j]])
                scan = scan[:, :3]
                scan[:, 2] = lidar_dict[keys[j]] 
                frame = np.vstack((frame, scan))
            frame, center = prepare_pc(frame, 1024)
            frame_old = copy.deepcopy(frame)
            frame = pc_normalize(frame, center)
            frame = frame[:, :2]
            frame = np.expand_dims(frame, 0)
            frame = np.transpose(frame, (0, 2, 1)).astype(dtype=np.float32)

            with torch.no_grad():
                self.model.eval()
                output, _, _ = self.model(torch.from_numpy(frame).to(self.device))
                output = output.view(-1, self.num_classes)
                output = torch.argmax(output, dim=1).cpu()
                frame = np.transpose(np.squeeze(frame), (1, 0))
                frame_old = frame_old[output == 1]

            self.extract_wheels(frame_old, min_wheels=3, use_bumper=False, min_samples=2, splitting=True)
            # print(self.cars_out)
            self.data["pointnet"].append(self.cars_out)
            # print(self.data["unet"])

            '''frame = pc_denormalize(frame)
            plt.scatter(frame[:, 0], frame[:, 1], color='black')
            plt.scatter(frame_old[frame_old[:, 2] == lidar_dict[keys[0]], 0], frame_old[frame_old[:, 2] == lidar_dict[keys[0]], 1], color='green')
            plt.scatter(frame_old[frame_old[:, 2] == lidar_dict[keys[1]], 0], frame_old[frame_old[:, 2] == lidar_dict[keys[1]], 1], color='blue')
            plt.scatter(frame_old[frame_old[:, 2] == lidar_dict[keys[2]], 0], frame_old[frame_old[:, 2] == lidar_dict[keys[2]], 1], color='orange')
            plt.scatter(frame_old[frame_old[:, 2] == lidar_dict[keys[3]], 0], frame_old[frame_old[:, 2] == lidar_dict[keys[3]], 1], color='red')
            # plt.scatter(frame_old[:, 0], frame_old[:, 1], color='black')
            for j in range(len(self.cars_out)):
                plt.scatter(self.cars_out[j][0], self.cars_out[j][1], color='yellow', s=20)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()
            plt.show()
            input()'''

        # modify extract wheels - using lidars pertinence

    def extract_wheels(self, points, min_wheels=3, use_bumper=False, min_samples=2, splitting=True):
        self.cars_out = []
        # print("working on it boi")
        dbscan = DBSCAN(eps=0.35, min_samples=min_samples)
        if points.shape[0] < min_samples or points.ndim < 2:
            # print("Lack of detections")
            return
        idxs = points[:, 2] == lidar_dict['sick_back_middle']
        if np.nonzero(np.logical_not(idxs))[0].shape[0] < 2:
            # print("Lack of detections")
            return
        
        points = points[:, :2]
        # clustering = dbscan.fit_predict(points[np.logical_not(idxs), :2])
        clustering = dbscan.fit_predict(points[np.logical_not(idxs)])
        print(len(points[np.logical_not(idxs), :2]))
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
        for i in range(num_clusters_middle):
            cluster = points[idxs]
            cluster = cluster[np.nonzero(clustering_middle == i)]
            # center = np.mean(cluster, axis=0)
            center = cluster.mean(axis=0)
            cluster_centers_middle = np.vstack((cluster_centers_middle, [center[0], center[1], i]))

        # calculate centers of clusters from right/left scanners
        for i in range(num_clusters):
            cluster = points[np.logical_not(idxs)]
            cluster = cluster[np.nonzero(clustering == i)]
            diffs = (cluster - cluster[:, np.newaxis]) ** 2
            dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
            # max_dist = np.max(dists)
            max_dist = dists.max()

            if splitting and (1 < max_dist < DIAGONAL) and (len(cluster) > 25):
                sub_cluster_size = int(math.ceil(len(cluster) / DIVISION_FACTOR))
                decision_dists = dists[0]  # array of distances from the first point
                indexes = np.argsort(decision_dists)
                cluster = cluster[indexes]  # sort array of points according to distances
                for j in range(DIVISION_FACTOR - 1):
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
 
        # print(len(cluster_centers))
        fragment_ids = np.negative(np.ones(len(cluster_centers), dtype=int))
        fragments = []
        for j in range(len(cluster_centers)):
            splitted = False
            if len(np.nonzero(parent_clusters == parent_clusters[j])) > 1:
                splitted = True
            for k in range(j + 1, len(cluster_centers)):
                if splitted and len(np.nonzero(parent_clusters == parent_clusters[k])) > 1 and parent_clusters[j] != \
                        parent_clusters[k]:
                    continue
                d = euclidean_distance(cluster_centers[j][0], cluster_centers[k][0], cluster_centers[j][1],
                                       cluster_centers[k][1])
                if abs(d - WIDTH) < THRESHOLD:
                    # print("maybe")
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

                    for l in range(len(cluster_centers)):
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
                            # print("not enoguh wheels")
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
                        for l in range(len(ids)):
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
                        for l in range(len(parents_ids)):
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
                            n = np.array(n) * (-2 * idx + 1)
                            if idx == 0:
                                loc = [(w1[0] + w2[0] + w3_1[0] + w4_1[0]) / 4,
                                       (w1[1] + w2[1] + w3_1[1] + w4_1[1]) / 4, np.arctan2(n[1], n[0])]
                            else:
                                loc = [(w1[0] + w2[0] + w3_2[0] + w4_2[0]) / 4,
                                       (w1[1] + w2[1] + w3_2[1] + w4_2[1]) / 4, np.arctan2(n[1], n[0])]
                            fragments.append([numw[idx], act_d, act_ids, loc])
                            fragment_ids[act_ids] = len(fragments) - 1

        car_idx = np.unique(fragment_ids)
        car_idx = car_idx[car_idx != -1]
        for j in range(len(car_idx)):
            self.cars_out.append(
                [fragments[car_idx[j]][3][0], fragments[car_idx[j]][3][1], fragments[car_idx[j]][3][2]])


if __name__ == "__main__":

    model_path = "models/epoch_0.pth"

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":# and "2020-11-11-16-03-33" in filename:
                jobs.append(Inference(files[0], filename, model_path))
    print("Spawned %i processes" % (len(jobs)), flush=True)
    cpuCores = 1
    limit = cpuCores
    batch = cpuCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].run()
        else:
            for j in range(limit):
                try:
                    jobs[j].join()
                except:
                    pass
            limit += batch
            jobs[i].start()
