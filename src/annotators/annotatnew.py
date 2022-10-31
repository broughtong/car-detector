#!/usr/bin/python
import multiprocessing
import copy
import utils
import random
import pickle
import math
import cv2
import sys
import os
import time
import numpy as np
import concurrent
import concurrent.futures
import tqdm
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from scipy.spatial.transform import Rotation as R

from annotation_backends import maskrcnn, pointcloud
methods = [maskrcnn, pointcloud]
methods = [pointcloud]

datasetPath = "../data/lanoising"
annotationSource = "extrapolated"
scanFields = ["scans", "lanoising"]
outputPath = "../annotations/"
movementThreshold = 0.5
gtPath = "../data/gt"
gtBags = []

#augmentations
flipV = True
rotations = [0, 1, 2, 3, 4, 5, 6]
rotations = [0]

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Annotator():
    def __init__(self, path, folder, filename):

        self.path = path
        self.folder = folder
        self.filename = filename[:-12]
        self.fileCounter = 0
        self.data = {}

    def run(self):
        
        if self.filename in gtBags:
            print("Process spawned for GT file %s" % (self.filename), flush = True)
        else:
            print("Process spawned for file %s" % (self.filename), flush = True)

        basefn = os.path.join(self.path, self.folder, self.filename)
        with open(os.path.join(self.path, self.folder, self.filename + ".data.pickle"), "rb") as f:
            self.data.update(pickle.load(f))
        #with open(os.path.join(self.path, self.folder, self.filename + ".3d.pickle"), "rb") as f:
        #    self.data.update(pickle.load(f))
        with open(os.path.join(self.path, self.folder, self.filename) + ".scans.pickle", "rb") as f:
            self.data.update(pickle.load(f))

        oldx, oldy, = 999, 999
        for frame in range(len(self.data["scans"])):

            annotations = self.data[annotationSource][frame]

            if self.filename not in gtBags and 1==2:
                #throw away frames without much movement
                trans = self.data["trans"][frame]
                x = trans[0][-1]
                y = trans[1][-1]

                diffx = x - oldx
                diffy = y - oldy
                dist = ((diffx**2)+(diffy**2))**0.5

                if dist < movementThreshold:
                    print("too samll dist", flush=True)
                    continue
                oldx = x
                oldy = y
                if len(annotations) == 0:
                    #for training purposes, dont teach on empty images
                    print("no annos skip", flush=True)
                    continue

            for scanField in scanFields:

                scan = self.data[scanField][frame]
                combined = utils.combineScans(scan)
                #cloud = self.data["pointclouds"][frame]

                #augmentations
                rotationsToDo = rotations
                if self.filename in gtBags:
                    rotationsToDo = [0]

                for rotation in rotationsToDo:

                    r = np.identity(4)
                    orientationR = R.from_euler('z', 0)
                    if rotation != 0:
                        orientationR = R.from_euler('z', rotation)
                        r = R.from_euler('z', rotation)
                        r = r.as_matrix()
                        r = np.pad(r, [(0, 1), (0, 1)])
                        r[-1][-1] = 1

                    newScan = np.copy(combined)
                    for point in range(len(newScan)):
                        newScan[point] = np.matmul(r, newScan[point])
                    newAnnotations = np.copy(annotations)
                    for i in range(len(annotations)):
                        v = [*annotations[i][:2], 1, 1]
                        v = np.matmul(r, v)[:2]
                        o = orientationR.as_euler('zxy', degrees=False)
                        o = o[0]
                        o += annotations[i][2]
                        newAnnotations[i] = [*v, o]

                    if self.filename in gtBags and 1==2:
                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=3)
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + ".png")
                        carPoints, nonCarPoints = self.getInAnnotation(newScan, newAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, newAnnotations)

                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + ".")
                        self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + ".")
                        self.saveAnnotations(fn, newScan, newAnnotations)

                    #raw
                    if self.filename not in gtBags and len(newAnnotations):

                        for method in methods:
                            filename = self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation)
                            method.annotate(filename, scan, newAnnotations, scanField)

                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=5)
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        carPoints, nonCarPoints = utils.getInAnnotation(newScan, newAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, newAnnotations)

                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".")
                        #self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".ply")
                        #self.saveAnnotations(fn, newScan, newAnnotations, cloud)

                    if flipV and len(newAnnotations) and self.filename not in gtBags :

                        fScan = np.copy(newScan)
                        for point in range(len(newScan)):
                            fScan[point][0] = -fScan[point][0]
                        fAnnotations = np.copy(newAnnotations)
                        for i in range(len(annotations)):
                            fAnnotations[i][0] = -fAnnotations[i][0]
                            fAnnotations[i][2] = -fAnnotations[i][2]

                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        utils.drawImgFromPoints(fn, fScan, [], [], [], [], dilation=3)
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        carPoints, nonCarPoints = self.getInAnnotation(fScan, fAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, fAnnotations) 

                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        utils.drawImgFromPoints(fn, fScan, [], [], fAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                        self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                        #self.saveAnnotations(fn, newScan, newAnnotations)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, None):
        if type(item) == int:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, 1))
    listenProcess.start()

    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag.pickle"
                gtBags.append(fn)

    if len(gtBags) == 0:
        print("Not evaluating against ground truth")

    for idx, method in enumerate(methods):
        methods[idx] = method.Annotator(outputPath, scanFields)

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" in filename:
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Annotator(path, folder, filename))

    workers = 1
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)

        for future in futures:
            try:
                pass
                queue.put(str(future.result()))
            except Exception as e:
                queue.put("P Exception: " + str(e))
