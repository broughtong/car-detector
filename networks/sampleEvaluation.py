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

datasetPath = "../data/results/simple-s"
outputPath = "../data/results/temporal-prc"
os.makedirs(outputPath, exist_ok=True)

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Inference(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path

    def run(self):

        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        self.inference()
        
        with open(os.path.join(outputPath, self.filename), "wb") as f:
            pickle.dump(self.data, f, protocol=2)

    def inference(self):

        #HERE, loop over each frame: self.data["scans"][frameIndex]

        #combine all the self.data["scans"][frameIndex]["sick_front, etc"]

        #pass the combined fram to the network, get the car array

        #write it to for example #self.data["unet"][frameIdx]

        #the run function will then save this to disk

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Inference(files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 4
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

