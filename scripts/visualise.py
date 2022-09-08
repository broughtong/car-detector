import cv2
import shutil
import multiprocessing
import pickle
import os
import math
import numpy as np
import copy
import multiprocessing
import utils

datasetPath = "../data/temporal"
outPath = "../visualisation/temporal"
scanField = "pointclouds"
annotationField = "extrapolated"

class Visualise(multiprocessing.Process):
    def __init__(self, folder, filename):
        multiprocessing.Process.__init__(self)

        self.folder = folder
        self.filename = filename[:-12]
        self.fileCounter = 0
        self.data = {}

    def run(self):
        
        print("Process spawned for file %s" % (self.filename), flush = True)

        basefn = os.path.join(datasetPath, self.folder, self.filename)
        with open(basefn + ".data.pickle", "rb") as f:
            self.data.update(pickle.load(f))
        with open(basefn + ".3d.pickle", "rb") as f:
            self.data.update(pickle.load(f))

        for frameIdx in range(len(self.data[scanField])):
            self.drawFrame(frameIdx)

    def drawFrame(self, idx):

        scans = self.data[scanField][idx]
        #scans = combineScans([scans["sick_back_left"], scans["sick_back_right"], scans["sick_back_middle"], scans["sick_front"]])
        #scans = combineScans(self.data["scans"][idx])

        fn = os.path.join(outPath, self.folder, self.filename + "-" + str(idx) + ".png")
        os.makedirs(os.path.join(outPath, self.folder), exist_ok=True)
        utils.drawImgFromPoints(fn, scans, [], [], self.data[annotationField][idx], [], 1, False)

if __name__ == "__main__":

    try:
        shutil.rmtree(outPath)
    except:
        pass
    os.makedirs(outPath, exist_ok=True)

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" in filename:
                path = datasetPath
                folder = files[0][len(path)+1:]
                jobs.append(Visualise(folder, filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    maxCores = 28
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

