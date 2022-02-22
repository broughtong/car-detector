#!/usr/bin/python
import rospy
import pickle
import math
import sys
import os
import multiprocessing
import time
import numpy as np

datasetPath = "../data/results/temporal-s"
outPath = "../data/results/temporal-s-stripped"

os.makedirs(outPath, exist_ok=True)

class Stripper(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path

    def run(self):

        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        blackList = ["annotations_rel", "annotationsOdom", "lerpedOdom", "lerped", "extrapOdom"]

        newData = {}
        for key in self.data.keys():
            if key in blackList:
                continue
            newData[key] = self.data[key]

        with open(os.path.join(outPath, self.filename), "wb") as f:
            pickle.dump(newData, f)

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Stripper(files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 5
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

