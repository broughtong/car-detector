#!/usr/bin/python
import time
import multiprocessing
import rospy
import pickle
import os
import h5py
import numpy as np
import math
import random
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

rotationIndex = 6   # how many times rotate one frame
testingIndex = 6    # ratio of saving to testing folder
dimIndex = 4    # how many dimensions to keep (in order: 1. x-coord, 2. y-coord, 3. lidar pertinence, 4. intensity)

datasetPath = "../data/results/lanoising/"
scanFields = ["scans", "lanoising"]
gtPath = "../data/gt"


class DataGenerator(multiprocessing.Process):
    def __init__(self, path, folder, filename, target_path, scanField, version="extrapolated"):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.folder = folder
        self.filename = filename
        self.target_path = target_path
        self.scanField = scanField
        with open(os.path.join(path, folder, filename), "rb") as f:
            self.data = pickle.load(f)
        self.version = version
        self.save_id = 0

        self.line_coefficients = []
        self.num_cars = 0

        self.n_points = 1526

        self.H = 512
        self.W = 512
        self.resolution = 0.075

        self.length = 4.85
        self.width = 2.5

        self.tmp_cars = []

    def run(self):
        print("Processing to target {}".format(self.target_path))
        n = len(self.data[self.scanField])

        scans = list(self.data[self.scanField][0].keys())
        # FIXME: add all possible topics to the dictionary
        scans_z_coords = {"sick_back_right": 0, "sick_back_left": 0, "sick_back_middle": 1, "sick_front": 0.5}

        for i in range(n):
            if i % 1000 == 0 and i != 0:
                print("Processing frame {}/{}". format(i, n-1))
            self.init_lines(i)
            data = np.empty((0, dimIndex))
            labels = np.empty(0)

            for scan_id, lidar in enumerate(scans):
                for j in range(len(self.data[self.scanField][i][lidar])):
                    point = self.data[self.scanField][i][lidar][j]
                    labels = np.hstack((labels, self.is_car(point, scan_id)))
                    point[2] = scans_z_coords[lidar]
                    data = np.vstack((data, point[:dimIndex]))

            # rotate the frame multiple times and save it
            for j in range(rotationIndex):
                angle = math.radians(random.randrange(1, 360))

                pc = copy.deepcopy(data)
                pc_out = copy.deepcopy(data)
                pc_out[:, 0] = pc[:, 0] * math.cos(angle) - pc[:, 1] * math.sin(angle)
                pc_out[:, 1] = pc[:, 0] * math.sin(angle) + pc[:, 1] * math.cos(angle)

                self.save_np(pc_out, labels)
                self.save_id += 1

                # visualize one of rotated frames with corresponding BB for cars
                """plt.figure()
                plt.scatter(pc_out[labels != -1, 0], pc_out[labels != -1, 1], color='red')
                plt.scatter(pc_out[labels == -1, 0], pc_out[labels == -1, 1], color='black')
                for j in range(len(self.tmp_cars)):

                    px = self.tmp_cars[j][0]
                    py = self.tmp_cars[j][1]
                    px1 = px*math.cos(angle)-py*math.sin(angle)
                    py1 = px*math.sin(angle)+py*math.cos(angle)
                    angle1 = self.tmp_cars[j][2] + angle

                    plt.scatter(px1, py1, color='yellow', s=50)
                    corners = self.get_corners_of_rectangle_reentrant(px1, py1, angle1)
                    corners[[2, 3]] = corners[[3, 2]]
                    poly = Polygon(corners, facecolor='none', edgecolor='red')
                    plt.gca().add_patch(poly)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.draw()
                plt.show()"""

    def init_lines(self, i):
        # cars = self.data["annotations"][i]
        cars = self.data[self.version][i]
        self.tmp_cars = cars
        self.line_coefficients = []
        for j in range(len(cars)):
            corners = self.get_corners_of_rectangle(i, j)
            self.calculate_line_coefficients(corners)
        self.num_cars = len(cars)

    def calculate_line_coefficients(self, corners):
        """function to calculate line coefficients from an array of wheels"""

        p1 = corners[0]
        p2 = corners[1]
        p3 = corners[2]
        p4 = corners[3]

        a1 = -(p2[1] - p1[1])
        b1 = -(p1[0] - p2[0])
        n1 = (a1 ** 2 + b1 ** 2) ** 0.5
        self.line_coefficients.append([a1 / n1, b1 / n1, (p2[1] * p1[0] - p2[0] * p1[1]) / n1])  # a1, b1, c1

        a2 = -(p3[1] - p4[1])
        b2 = -(p4[0] - p3[0])
        n2 = (a2 ** 2 + b2 ** 2) ** 0.5
        self.line_coefficients.append([a2 / n2, b2 / n2, (- p4[1] * p3[0] + p4[0] * p3[1]) / n2])  # a2, b2, c2

        a3 = -(p1[1] - p3[1])
        b3 = -(p3[0] - p1[0])
        n3 = (a3 ** 2 + b3 ** 2) ** 0.5
        self.line_coefficients.append([a3 / n3, b3 / n3, (- p3[1] * p1[0] + p3[0] * p1[1]) / n3])  # a3, b3, c3

        a4 = -(p4[1] - p2[1])
        b4 = -(p2[0] - p4[0])
        n4 = (a4 ** 2 + b4 ** 2) ** 0.5
        self.line_coefficients.append([a4 / n4, b4 / n4, (p4[1] * p2[0] - p4[0] * p2[1]) / n4])  # a4, b4, c4

    def calculate_dist_from_line(self, id, point):
        """function that returns distance of a point from a specific line"""
        return self.line_coefficients[id][0] * point[0] + self.line_coefficients[id][1] * point[1] + self.line_coefficients[id][2]

    def get_corners_of_rectangle(self, i, j):
        """function that calculates corners of a rectangle parametrized with center coordinates and angle"""

        h = self.length
        w = self.width

        center = np.array([self.data[self.version][i][j][0], self.data[self.version][i][j][1]])
        angle = self.data[self.version][i][j][2]
        v1 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([-v1[1], v1[0]])

        return np.array([center+h/2*v1+w/2*v2, center+h/2*v1-w/2*v2, center-h/2*v1+w/2*v2, center-h/2*v1-w/2*v2])

    def get_corners_of_rectangle_reentrant(self, c_x, c_y, angle):
        """function that calculates corners of a rectangle parametrized with center coordinates and angle"""

        h = self.length
        w = self.width

        center = np.array([c_x, c_y])
        v1 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([-v1[1], v1[0]])

        return np.array([center+h/2*v1+w/2*v2, center+h/2*v1-w/2*v2, center-h/2*v1+w/2*v2, center-h/2*v1-w/2*v2])

    def is_car(self, point, scan_id):
        """function that determines if point belongs to a car"""

        best = float('inf')
        best_id = -1
        tolerance = 0 if scan_id < 2 else 0.25

        for k in range(0, len(self.line_coefficients), 4):
            d1 = self.calculate_dist_from_line(k, point)
            d2 = self.calculate_dist_from_line(k + 1, point)
            d3 = self.calculate_dist_from_line(k + 2, point)
            d4 = self.calculate_dist_from_line(k + 3, point)

            if (d1 < 0) and (d2 < 0) and (d3 < 0) and (d4 < 0):
                return k // 4
            else:
                tmp_d1 = abs(d1) < self.length + tolerance
                tmp_d2 = abs(d2) < self.length + tolerance
                tmp_d3 = abs(d3) < self.width
                tmp_d4 = abs(d4) < self.width

                if tmp_d1 and tmp_d2 and tmp_d3 and tmp_d4:
                    tmp_arr = np.array([d1, d2, d3, d4])
                    tmp_arr = tmp_arr[tmp_arr > 0]
                    act_d = max(tmp_arr)
                    if act_d < best:
                        best = act_d
                        best_id = k // 4

        return best_id

    def save_np(self, pc_out, labels):
        #print(os.path.join(self.target_path, str(self.save_id) + ".npz"))
        np.savez(os.path.join(self.target_path, str(self.save_id) + ".npz"), pc=pc_out, labels=labels, allow_pickle=False)

if __name__ == "__main__":

    gtBags = []
    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag.pickle"
                gtBags.append(fn)

    evalBags = []
    bags = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            path = datasetPath
            folder = files[0][len(path):]
            if filename in gtBags:
                evalBags.append([path, folder, filename])
            else:
                bags.append([path, folder, filename])

    print(bags)

    jobs = []
    for scanField in scanFields:
        outputPath = "./bags/" + scanField

        for i in range(len(bags)):
            if i % testingIndex == 0:
                target_dir = os.path.join(outputPath, 'testing', bags[i][2][:-11])
            else:
                target_dir = os.path.join(outputPath, 'training', bags[i][2][:-11])
            os.makedirs(target_dir, exist_ok=True)
            jobs.append(DataGenerator(bags[i][0], bags[i][1], bags[i][2], target_dir, scanField))
            print("Adding job for %s for %s" % (scanField, bags[i][2]))
        for i in range(len(evalBags)):
            target_dir = os.path.join(outputPath, 'evaluation', evalBags[i][2][:-11])
            os.makedirs(target_dir, exist_ok=True)
            jobs.append(DataGenerator(evalBags[i][0], evalBags[i][1], evalBags[i][2], target_dir, scanField))
            print("Adding job for %s for %s" % (scanField, evalBags[i][2]))
     
    print("Spawned %i processes" % (len(jobs)), flush = True)
    maxCores = 32
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
