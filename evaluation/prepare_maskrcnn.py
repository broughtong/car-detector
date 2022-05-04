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

datasetPath = "../data/results/maskrcnn_scans_huge"
combinePath = "../data/results/lanoising"
outputPath = "../data/results/maskrcnn_scans_reprocessed-debug"
combinedOutPath = "../data/results/maskrcnn_scans_rectified-debug"
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
        #combinefn = os.path.join(combinePath, self.filename.split(".pickle")[0]+".pickle")
        #with open(combinefn, "rb") as f:
        #    self.combinedata = pickle.load(f)

        self.convert()
        self.data = None
        
        os.makedirs(os.path.join(outputPath, self.path), exist_ok=True)
        with open(os.path.join(outputPath, self.path, self.filename + ".annotations"), "wb") as f:
            pickle.dump(self.annotations, f, protocol=2)

    def convert(self):

        annotations = []
        for idx in range(len(self.data[0]["boxes"])):
            box = self.data[0]["boxes"][idx]
            label = self.data[0]["labels"][idx]
            score = self.data[0]["scores"][idx]
            mask = self.data[0]["masks"][idx]

            if score < 0.8:
                continue

            x = (float(box[0]) + float(box[2])) / 2
            y = (float(box[1]) + float(box[3])) / 2
            print(x, y)
            res = 1024
            scale = 25
            x = (x - (res/2)) / scale
            y = (y - (res/2)) / scale
            print(x, y)

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
    
    #extract result from network output into files
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                if ".annotations." not in filename:
                    if "-47-41" in filename:
                        if "ckle-99.pn" in filename:
                            jobs.append(Converter(files[0].split("/")[-1], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 50
    limit = cpuCores
    batch = cpuCores
    for i in range(len(jobs)):
        print("%i frame of %i" % (i, len(jobs)))
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                try:
                    jobs[j].join()
                except ValueError:
                    pass
                jobs[j].close()
            limit += batch
            jobs[i].start()

    print("Jobs finished, checking them too", flush=True)
    for job in jobs:
        try:
            job.join()
            #print("Job fine", flush=True)
        except:
            #print("Job not fine", flush=True)
            pass

    print("All jobs checked", flush=True)
    import sys
    sys.exit(0)

    #now we merge those detections into the original file structures
    #basically bring it into a common format
    combinableFilenames = []
    for files in os.walk(outputPath):
        for filename in files[2]:
            if filename[-12:] == ".annotations":
                combinableFilenames.append(os.path.join(files[0], filename.split(".")[0]))
    combinableFilenames = list(set(combinableFilenames))

    print("Combining %i files" % (len(combinableFilenames)))

    for base in combinableFilenames:

        print("Combining bag", base)
        modelPath = base.split("/")[-2]
        print(modelPath)

        #open combinable file
        combineFile = ""
        for files in os.walk(combinePath):
            for filename in files[2]:
                if filename.split(".")[0] == base.split("/")[-1]:
                    subPath = files[0][len(combinePath)+1:]
                    combineFile = os.path.join(subPath, filename)
                    break

        if combineFile == "":
            print("Error combining ", filename)
            break

        data = []
        with open(os.path.join(combinePath, combineFile), "rb") as f:
            data = pickle.load(f)

        readyFiles = []
        for files in os.walk(outputPath):
            for filename in files[2]:
                print(filename)
                if filename[-12:] == ".annotations":
                    if filename.split(".")[0] == base.split("/")[-1]:
                        readyFiles.append(filename)

        print(len(readyFiles), len(data["ts"]))
        #if len(readyFiles) != len(data["ts"]):
        #    print("Warning, frame mismatch", base)

        data["maskrcnn"] = []
        for i in range(len(data["scans"])):
            data["maskrcnn"].append([])

        #open each file, add it to the correct frame
        for filename in readyFiles:
            idx = int(filename.split(".pickle-")[1].split(".")[0])
            with open(os.path.join(outputPath, modelPath, filename), "rb") as f:
                annotations = pickle.load(f)
                data["maskrcnn"][idx] = annotations
        
        #for i in range(len(readyFiles)):
        #    if data["maskrcnn"][i] == None:
        #        print("Warning, empty frame found")

        os.makedirs(os.path.join(combinedOutPath, modelPath), exist_ok=True)
        print("Saving to %s" % (os.path.join(combinedOutPath, modelPath, base.split("/")[-1] + ".bag.pickle")))
        with open(os.path.join(combinedOutPath, modelPath, base.split("/")[-1] + ".bag.pickle"), "wb") as f:
            pickle.dump(data, f, protocol=2)
