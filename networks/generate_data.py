#!/usr/bin/python
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

labels_out = np.empty((0, 1526))
data_out = np.empty((0, 1526, 2))

rotationIndex = 2   # how many times rotate one frame
testingIndex = 5    # ratio of saving to testing folder
dimIndex = 4    # how many dimensions to keep (in order: 1. x-coord, 2. y-coord, 3. lidar pertinence, 4. intensity)

datasetPath = "../data/results/lanoising-ts5"
scanField = "scans"
outputPath = "./bags/" + datasetPath.split("/")[-1] + "-" + scanField
os.makedirs(outputPath, exist_ok=True)
os.makedirs(os.path.join(outputPath, 'training'), exist_ok=True)
os.makedirs(os.path.join(outputPath, 'testing'), exist_ok=True)

class DataGenerator:
    def __init__(self, filename, target_path, version="extrapolated"):

        self.target_path = target_path
        with open(filename, "rb") as f:
            self.data = pickle.load(f)
        self.version = version
        self.save_id = 0

        self.line_coefficients = []
        self.num_cars = 0

        self.n_points = 1526

        self.H = 512
        self.W = 512
        self.resolution = 0.075

        self.length = 4.5
        self.width = 2.3

        self.tmp_cars = []

    def generate(self):
        print("Processing to target {}".format(self.target_path))
        n = len(self.data[scanField])
        scans = ["sick_back_right", "sick_back_left", "sick_back_middle", "sick_front"]

        for i in range(n):
            if i % 100 == 0:
                print("Processing frame {}/{}". format(i, n-1))
            self.init_lines(i)
            data = np.empty((0, dimIndex))
            labels = np.empty(0)

            for scan_id, lidar in enumerate(scans):
                for j in range(len(self.data[scanField][i][lidar])):
                    point = self.data[scanField][i][lidar][j]
                    labels = np.hstack((labels, self.is_car(point, scan_id)))
                    point[2] = 0.3 * scan_id + 0.4
                    data = np.vstack((data, point[:dimIndex+1]))

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
                '''plt.figure()
                plt.scatter(pc_out[labels != -1, 0], pc_out[labels != -1, 1], color='red')
                plt.scatter(pc_out[labels == -1, 0], pc_out[labels == -1, 1], color='black')
                for j in range(len(self.tmp_cars)):

                    px = self.tmp_cars[j][0]
                    py = self.tmp_cars[j][1]
                    px1 = px*math.cos(angle)-py*math.sin(angle)
                    py1 = px*math.sin(angle)+py*math.cos(angle)
                    angle1 = self.tmp_cars[j][2] + angle

                    plt.scatter(px1, py1, color='yellow', s=100)
                    corners = self.get_corners_of_rectangle_reentrant(px1, py1, angle1)
                    corners[[2, 3]] = corners[[3, 2]]
                    poly = Polygon(corners, facecolor='none', edgecolor='red')
                    plt.gca().add_patch(poly)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.draw()
                plt.show()'''

        # self.save_h5()
        # self.stack_to_global()

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

    def generate_img(self, data, labels):

        center = np.mean(data[:, :2], axis=0)
        grid = np.zeros((3, self.H, self.W))
        grid_labels = np.zeros((self.H, self.W))

        ids = data[:, 2] == 0
        pts = data[ids, :2]
        lbls = labels[ids]
        rows = np.expand_dims(np.ceil(self.H // 2 - 1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
        cols = np.expand_dims(np.ceil(self.W // 2 - 1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
        ids = np.nonzero(np.logical_and.reduce(
            (rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
        rows = rows[ids].astype(int)
        cols = cols[ids].astype(int)
        lbls = lbls[ids]
        grid[0][rows[:], cols[:]] = 1
        grid_labels[rows[lbls != -1], cols[lbls != -1]] = 1

        ids = data[:, 2] == 1
        pts = data[ids, :2]
        lbls = labels[ids]
        rows = np.expand_dims(np.ceil(self.H // 2 - 1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
        cols = np.expand_dims(np.ceil(self.W // 2 - 1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
        ids = np.nonzero(np.logical_and.reduce(
            (rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
        rows = rows[ids].astype(int)
        cols = cols[ids].astype(int)
        lbls = lbls[ids]
        grid[1][rows[:], cols[:]] = 1
        grid_labels[rows[lbls != -1], cols[lbls != -1]] = 1

        ids = data[:, 2] == 2
        pts = data[ids, :2]
        lbls = labels[ids]
        rows = np.expand_dims(np.ceil(self.H // 2 - 1 - ((pts[:, 1] - center[1]) / self.resolution)), axis=1)
        cols = np.expand_dims(np.ceil(self.W // 2 - 1 + ((pts[:, 0] - center[0]) / self.resolution)), axis=1)
        ids = np.nonzero(np.logical_and.reduce(
            (rows[:, 0] >= 0, rows[:, 0] <= self.H - 1, cols[:, 0] >= 0, cols[:, 0] <= self.W - 1)))
        rows = rows[ids].astype(int)
        cols = cols[ids].astype(int)
        lbls = lbls[ids]
        grid[2][rows[:], cols[:]] = 1
        grid_labels[rows[lbls != -1], cols[lbls != -1]] = 1

        '''plt.figure()
        plt.imshow(grid.transpose(1, 2, 0))
        plt.imshow(grid_labels)
        plt.show()
        input()'''

        self.labels_imgs = np.vstack((self.labels_imgs, [grid_labels]))
        self.data_imgs = np.vstack((self.data_imgs, [grid]))

    def save_h5(self):
        print("Saving data of shape: ", self.data_out.shape, self.labels_out.shape)
        f = h5py.File(self.target_path + ".h5", 'w')
        lab = f.create_dataset("labels", data=self.labels_out)
        dat = f.create_dataset("data", data=self.data_out)
        f.close()

    def save_np(self, pc_out, labels):
        # np.save(os.path.join(self.target_path, str(self.save_id) + "pc.npy"), pc_out, allow_pickle=False)
        # np.save(os.path.join(self.target_path, str(self.save_id) + "l.npy"), labels, allow_pickle=False)
        np.savez(os.path.join(self.target_path, str(self.save_id) + ".npz"), pc=pc_out, labels=labels, allow_pickle=False)

    def stack_to_global(self):
        global data_out, labels_out
        data_out = np.vstack((data_out, self.data_out))
        labels_out = np.vstack((labels_out, self.labels_out))


if __name__ == "__main__":

    #goodbags = []
    #with open("goodbags", "r") as f:
    #    goodbags = f.read()
    #goodbags = goodbags.split("\n")
    #goodbags = filter(None, goodbags)
    #goodbags = [x.split(" ")[0] + ".bag.pickle" for x in goodbags]
    #print(goodbags)

    bags = []
    for folders in os.walk(datasetPath):
        if folders[0] == datasetPath:
            bags = folders[2]
            break
    # print(bags)

    for files in os.walk(outputPath):
        for filename in files[2]:
            if filename[:-3] in bags:
                idx = bags.index(filename[:-3])
                del bags[idx]

    # print("Some bags maybe skipped")
    print(bags)

    for files in os.walk(outputPath):
        for filename in files[2]:
            if filename[:-3] in bags:
                idx = bags.index(filename[:-3])
                del bags[idx]

    print("Some bags maybe skipped")
    print(bags)

    #usebags = []
    #for i in bags:
    #    if i in goodbags:
    #        usebags.append(i)

    #print(usebags)
    #for i in range(len(usebags)):
    #    print(i)
    #    print(usebags[i])

    for i in range(len(bags)):
        if i % testingIndex == 0:
            target_dir = os.path.join(outputPath, 'testing', bags[i][:-11])     # [:-11] is used to discard extension .bag.pickle
        else:
            target_dir = os.path.join(outputPath, 'training', bags[i][:-11])
        os.makedirs(target_dir, exist_ok=True)
        gen = DataGenerator(os.path.join(datasetPath, bags[i]), target_dir)
        gen.generate()
