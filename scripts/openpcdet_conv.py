#!/usr/bin/python
import shutil
import copy
import utils
import pickle
import math
import sys
import os
import multiprocessing
import time
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np

datasetPath = "../data/temporal/"
outputPath = "../data/npy"
visualisationPath = "../visualisation/temporal-"

points[:, 3] = 0 
np.save(`my_data.npy`, points) 

if __name__ == "__main__":

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" in filename:
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Temporal(path, folder, filename, 0.8, 50, 5, 50, 10))
                #distance thresh, interp window, interp dets req, extrap window, extrap dets req

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

