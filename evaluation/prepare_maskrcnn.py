#!/usr/bin/python
import math
import rospy
import pickle
import math
import cv2
import sys
import os
import rosbag
import multiprocessing
import time
from numpy.linalg import eig
from os import devnull
from sklearn.cluster import DBSCAN
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

datasetPath = "../data/results/maskrcnn_raw"
combinePath = "../data/results/temporal-s"
outputPath = "../data/results/maskrcnn"
#visualisationPath = "../visualisation/"

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Converter(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path

    def run(self):

        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(datasetPath, self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)
        combinefn = os.path.join(combinePath, self.filename.split(".pickle")[0]+".pickle")
        with open(combinefn, "rb") as f:
            self.combinedata = pickle.load(f)

        self.convert()
        
        os.makedirs(os.path.join(outputPath, self.path), exist_ok=True)
        with open(os.path.join(outputPath, self.path, self.filename), "wb") as f:
            pickle.dump(self.annotations, f, protocol=2)

    def convert(self):

        annotations = []
        for idx in range(len(self.data[0]["boxes"])):
            box = self.data[0]["boxes"][idx]
            label = self.data[0]["labels"][idx]
            score = self.data[0]["scores"][idx]
            mask = self.data[0]["masks"][idx][0]

            x = (float(box[0]) + float(box[2])) / 2
            y = (float(box[1]) + float(box[3])) / 2

            mask = mask.detach().cpu().numpy()
            mask = mask[math.floor(box[1]):math.ceil(box[3]), math.floor(box[0]):math.ceil(box[2])]

            coords = []
            for row in range(len(mask)):
                for val in range(len(mask[row])):
                    mask[row][val] = (mask[row][val])*255
                    if mask[row][val] > 50:
                        coords.append([row, val])

            cov = np.cov(np.transpose(coords))
            w, v = eig(cov)
            bigIdx = 0
            if w[1] > w[0]:
                bigIdx = 1
            ev = v[bigIdx]
            rot = math.atan2(ev[1], ev[0])

            #cv2.imwrite("et.png", mask)
            #import sys
            #sys.exit(0)

            annotation = [x, y, rot]
            annotations.append(annotation)
        self.annotations = annotations

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Converter(files[0].split("/")[-1], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 6
    limit = cpuCores
    batch = cpuCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()

