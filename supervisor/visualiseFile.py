#!/usr/bin/python
import utils
import rospy
import pickle
import math
import cv2
import sys
import os
import rosbag
import multiprocessing
import time
from os import devnull
from sklearn.cluster import DBSCAN
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

datasetPath = "../data/results/temporal-s"
visualisationPath = "../visualisation/temporal-s"

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Visualiser(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path

    def run(self):

        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        self.visualise()

    def visualise(self):

        scans = self.data["scans"]
        raws = self.data["annotations"]
        temporals = self.data["extrapolated"]

        self.fileCounter = 0
        for i in range(len(scans)):

            points = np.concatenate([scans[i]["sick_back_left"], scans[i]["sick_back_right"], scans[i]["sick_back_middle"], scans[i]["sick_front"]])

            raw = raws[i]
            temporal = temporals[i]

            os.makedirs(visualisationPath, exist_ok=True)
            fn = os.path.join(visualisationPath, "%s-%s-%s.png" % ("interpolated", self.filename, self.fileCounter))
            utils.drawImgFromPoints(fn, points, [], [], temporal, raw, renderAnnotations=True)
            self.fileCounter += 1

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Visualiser(files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 7
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

