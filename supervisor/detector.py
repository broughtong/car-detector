#!/usr/bin/python
import rospy
import pickle
import math
import cv2
import sys
import os
import multiprocessing
import numpy as np
from sklearn.cluster import DBSCAN

datasetPath = "../data/extracted/"
outputPath = "../data/results/simple-s/"
visualisationPath = "../visualisation/simple-s/"
WIDTH = 1.5
LENGTH = 2.5
DIAGONAL = 3.1
DIVISION_FACTOR = 5
THRESHOLD = 0.3

def euclidean_distance(x_1, x_2, y_1, y_2):
    return math.sqrt((x_1-x_2)*(x_1-x_2)+(y_1-y_2)*(y_1-y_2))

class Annotator(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.filename = filename
        self.fileCounter = 0
        self.detections = []
        self.relaxed = []

    def run(self):
        
        print("Process spawned for file %s %s" % (self.path, self.filename))

        with open(self.path + self.filename, "rb") as f:
            self.data = pickle.load(f)

        for idx in range(len(self.data["scans"])):
            scan = self.data["scans"][idx]
            trans = self.data["trans"][idx]
            ts = self.data["ts"][idx]

            cars, cars_rel = self.processScan(scan)
            self.detections.append(cars)
            self.relaxed.append(cars_rel)
            self.fileCounter += 1
        
        self.data["annotations"] = self.detections
        self.data["annotations_rel"] = self.relaxed
        os.makedirs(outputPath, exist_ok=True)
        fn = os.path.join(outputPath, self.filename)
        with open(fn, "wb") as f:
            print("Writing to ", fn)
            pickle.dump(self.data, f, protocol=2)

        print("Process finished for file %s" % (self.filename))

    def processScan(self, scan):

        #only uses 3 scans!
        # points = np.concatenate([scan["sick_back_left"], scan["sick_back_right"], scan["sick_front"]])
        # cars = self.detectCarGeometry(points)

        if len(scan["sick_back_middle"]) == 0:
            return []
        if len(scan["sick_back_left"]) == 0 or len(scan["sick_back_right"]) == 0:
            return []

        points = np.concatenate([scan["sick_back_left"], scan["sick_back_right"]])
        middle_points = np.array(scan["sick_back_middle"])
        cars = self.detect_car_geometric(points, middle_points, min_wheels=4, splitting=True, use_bumper=False)
        cars_relaxed = self.detect_car_geometric(points, middle_points, min_wheels=3, splitting=True, use_bumper=False)

        wheels = []
        colours = []
        for i in cars:
            wheels.append([i[0], i[1]])
            colours.append([0, 255, 255])
    
        self.pointsToImgsDrawWheels(points, "car-estimates", wheels, colours)

        return cars, cars_relaxed

    def detect_car_geometric(self, points, middle_points, min_wheels=4, splitting=False, use_bumper=True):

        dbscan = DBSCAN(eps=0.35, min_samples=5)
        if min_wheels == 3:
            dbscan = DBSCAN(eps=0.35, min_samples=2)

        clustering = dbscan.fit_predict(points[:, :2])
        clustering_middle = dbscan.fit_predict(middle_points[:, :2])

        num_clusters = np.unique(clustering)
        num_clusters = len(num_clusters[num_clusters != -1])
        num_clusters_middle = np.unique(clustering_middle)
        num_clusters_middle = len(num_clusters_middle[num_clusters_middle != -1])

        cluster_centers = np.empty((0, 3))
        parent_clusters = np.empty(0)
        cluster_centers_middle = np.empty((0, 3))

        # calculate centers of clusters from middle scanner
        for i in range(num_clusters_middle):
            cluster = middle_points[np.nonzero(clustering_middle == i)]
            center = np.mean(cluster, axis=0)
            cluster_centers_middle = np.vstack((cluster_centers_middle, [center[0], center[1], i]))

        # calculate centers of clusters from right/left scanners
        for i in range(num_clusters):

            cluster = points[np.nonzero(clustering == i)]
            diffs = (cluster - cluster[:, np.newaxis]) ** 2
            dists = np.sqrt(diffs[:, :, 0] + diffs[:, :, 1])  # dist matrix = sqrt((x_1 - y_1)**2 + (x_2- y_2)**2)
            max_dist = np.max(dists)

            if splitting and (1 < max_dist < DIAGONAL) and (len(cluster) > 25):
                sub_cluster_size = int(math.ceil(len(cluster) / DIVISION_FACTOR))
                decision_dists = dists[0]  # array of distances from the first point
                indexes = np.argsort(decision_dists)
                cluster = cluster[indexes]  # sort array of points according to distances
                for j in range(DIVISION_FACTOR - 1):
                    tmp = j * sub_cluster_size
                    center = np.mean(cluster[tmp:tmp + sub_cluster_size, :], axis=0)
                    cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                    parent_clusters = np.append(parent_clusters, i)
                tmp = (DIVISION_FACTOR - 1) * sub_cluster_size
                center = np.mean(cluster[tmp:, :], axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                parent_clusters = np.append(parent_clusters, i)
            else:
                center = np.mean(cluster, axis=0)
                cluster_centers = np.vstack((cluster_centers, [center[0], center[1], i]))
                parent_clusters = np.append(parent_clusters, i)

        cols = []
        for i in range(len(cluster_centers)):
            cols.append([0, 0, 255])

        self.pointsToImgsDrawWheels(np.concatenate([points, middle_points]), "clusters-j", cluster_centers[:, :2], cols)

        # **************** GEOMETRY FITTING PART ****************
        fragment_ids = np.negative(np.ones(len(cluster_centers), dtype=int))
        fragments = []

        for j in range(len(cluster_centers)):
            split = False
            if len(np.nonzero(parent_clusters == parent_clusters[j])) > 1:
                split = True
            for k in range(j + 1, len(cluster_centers)):
                if split and len(np.nonzero(parent_clusters == parent_clusters[k])) > 1 and parent_clusters[j] != \
                        parent_clusters[k]:
                    continue
                d = euclidean_distance(cluster_centers[j][0], cluster_centers[k][0], cluster_centers[j][1],
                                       cluster_centers[k][1])
                if abs(d - WIDTH) < THRESHOLD:
                    w1 = cluster_centers[j]
                    w2 = cluster_centers[k]

                    # CALC APPROX POSITIONS OF REMAINING WHEELS ON BOTH SIDES
                    n = [(w1[1] - w2[1]) / d, (w2[0] - w1[0]) / d]  # n=(-u2, u1)/norm <=> normalized normal vector
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

                    if not (numw[0] > 3 or numw[1] > 3):  # found less than four wheels check for middle scan points
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
                                        idx = np.argmin(np.array([d_values[0] + d_values[2], d_values[1] + d_values[3]]))
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
                            if idx == 0:
                                n = np.array(n) * (-2 * idx + 1)
                                loc = [(w1[0] + w2[0] + w3_1[0] + w4_1[0]) / 4, (w1[1] + w2[1] + w3_1[1] + w4_1[1]) / 4, np.arctan2(n[1], n[0])]
                                # loc = [w1, w2, w3_1, w4_1]
                            else:
                                loc = [(w1[0] + w2[0] + w3_2[0] + w4_2[0]) / 4, (w1[1] + w2[1] + w3_2[1] + w4_2[1]) / 4, np.arctan2(n[1], n[0])]
                                # loc = [w1, w2, w3_2, w4_2]
                            fragments.append([numw[idx], act_d, act_ids, loc])
                            fragment_ids[act_ids] = len(fragments) - 1

        car_idx = np.unique(fragment_ids)
        car_idx = car_idx[car_idx != -1]
        cars = []
        for i in range(len(car_idx)):
            #prevent misdetections from the robot itself
            x = fragments[car_idx[i]][3][0]
            y = fragments[car_idx[i]][3][1]
            z = fragments[car_idx[i]][3][2]
            if abs(x - -2.8) < 0.4 and abs(y - 0) < 0.4 and abs(z - -3.1) < 0.4:
                continue
            cars.append([fragments[car_idx[i]][3][0], fragments[car_idx[i]][3][1], fragments[car_idx[i]][3][2]])

        # cars are in format: center_x, center_y, angle [radians]

        wheels = [] #cluster_centers[:, :2]
        cols = cols

        for i in cars:
            wheels.append([i[0], i[1]])
            cols.append([255, 0, 0])


        if len(wheels) > 0:
            self.pointsToImgsDrawWheels(np.concatenate([points, middle_points]), "clusters-j-c", np.concatenate([cluster_centers[:, :2], wheels]), cols)
        return cars

    def pointsToImgsDrawWheels(self, points, location, wheels, colours):

        res = 1024
        scale = 25
        accum = np.zeros((res, res, 3))
        accum.fill(255)

        for point in points:
            x, y = point[:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                accum[x+int(res/2), y+int(res/2)] = [0, 0, 0]
            except:
                pass

        for wheel in range(len(wheels)):
            x, y = wheels[wheel][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)

            try:
                accum[x+int(res/2), y+int(res/2)] = colours[wheel]
            except:
                pass

            try:
                accum[x+int(res/2)+1, y+int(res/2)+1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)+1, y+int(res/2)-1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)-1, y+int(res/2)+1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)-1, y+int(res/2)-1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)-1, y+int(res/2)] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2), y+int(res/2)-1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2), y+int(res/2)+1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)+1, y+int(res/2)] = colours[wheel]
            except:
                pass
        fn = os.path.join(visualisationPath, location)
        os.makedirs(fn, exist_ok=True)
        fn = os.path.join(fn, self.filename + "-" + str(self.fileCounter) + ".png")
        cv2.imwrite(fn, accum)

if __name__ == "__main__":

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            jobs.append(Annotator(files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    maxCores = 7
    limit = maxCores
    batch = maxCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()
