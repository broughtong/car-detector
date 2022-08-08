#!/usr/bin/python
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import copy
import cv2
import numpy as np
import rospy
import pickle
import os
import sys
    
detectionThreshold = 1.0
closestOnly = False

evalName = "eval-gt2"
datasetPaths = {"../data/results/lanoising": "annotations"}
visualisationPath = "../visualisation/eval-" + evalName
tfPath = "../data/static_tfs"
gtPath = "/home/george/carloc/annotator/car_anotation/out"

resultsPath = os.path.join("./results/", evalName)

for key in datasetPaths.keys():
    if key[-1] == "/":
        print("Warning, please remove trailing slash")
        sys.exit(0)

def combineScans(scans):
    newScans = copy.deepcopy(scans[list(scans.keys())[0]])
    for key in list(scans.keys())[1:]:
        newScans = np.concatenate([newScans,scans[key]])
    return newScans

def findfile(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def evaluateFile(filename, method, filePart):
    global lastVal

    gtfn = filename.split(".")[0]
    print(gtfn)
    gtfn = findfile(gtfn + "-lidar.pkl", gtPath) 
    if gtfn is None:
        print("no gt found %s %s" % (gtfn, filename))
        return

    print("Evaluating %s from %s (%s)" % (filename, method, filePart))

    fn = findfile(filename, method)
    if not os.path.isfile(fn):
        print("Unable to open data file: %s" % (fn))
        return
    
    tffn = os.path.join(tfPath, filename)
    if not os.path.isfile(tffn):
        print("Unable to open tf file: %s" % (tffn))
        return

    data = []
    gtdata = []
    tfdata = []

    with open(fn, "rb") as f:
        data = pickle.load(f)
    with open(tffn, "rb") as f:
        tfdata = pickle.load(f)
    with open(gtfn, "rb") as f:
        gtdata = pickle.load(f, encoding='latin1')

    highestFreq = 0
    highestFreqFrames = 0
    for sensorIdx in range(len(gtdata)):
        if len(gtdata[sensorIdx]) > highestFreqFrames:
            highestFreq = sensorIdx
            highestFreqFrames = len(gtdata[sensorIdx])

    lastLoc = None

    for frameIdx in range(0, len(gtdata[highestFreq])):
        frame = gtdata[highestFreq][frameIdx]
        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        annotations = {}

        if gttime not in data["ts"]:
            continue
        dataFrameIdx = data["ts"].index(gttime)
        dataFrame = data["lanoising"][dataFrameIdx]
        dataFrame = combineScans(dataFrame)
        print("He")

        if lastLoc is None:
            lastLoc = data["trans"][dataFrameIdx]
        else:
            current = data["trans"][dataFrameIdx]
            dx = current[0][-1] - lastLoc[0][-1]
            dy = current[0][-1] - lastLoc[0][-1]
            dist = ((dx**2) + (dy**2)) ** 0.5
            if dist < 3:
                continue
            lastLoc = data["trans"][dataFrameIdx]


        for sensorIdx in range(len(gtdata)):
            if sensorIdx == highestFreq:
                annotations[sensorIdx] = gtdata[sensorIdx][frameIdx][1:]
                continue

            closestFrame = None
            bestdist = None
            for otherFrame in range(len(gtdata[sensorIdx])):
                oTime = rospy.Time(gtdata[sensorIdx][otherFrame][0].secs, gtdata[sensorIdx][otherFrame][0].nsecs)
                if bestdist is None:
                    bestdist = abs(oTime - gttime)
                    closestFrame = otherFrame
                    continue
                dist = abs(oTime - gttime)
                if dist < bestdist:
                    bestdist = dist
                    closestFrame = otherFrame
            annotations[sensorIdx] = gtdata[sensorIdx][closestFrame][1:]

        keys = {0: 'sick_back_left', 1: 'sick_back_middle', 2: 'sick_back_right', 3: 'sick_front'}

        combined = []
        for sensor in annotations.keys():
            annos = annotations[sensor]
            for i in annos:
                rotation = i[2]
                position = [i[0], i[1], 0, 1]
                mat = tfdata[keys[sensor]]["mat"]
                position = np.matmul(mat, position)
                quat = tfdata[keys[sensor]]["quaternion"]
                qx = quat[0]
                qy = quat[1]
                qz = quat[2]
                qw = quat[3]
                yaw = math.atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
                if sensor == 0:
                    yaw += -0.8
                if sensor == 1:
                    yaw += -0.0
                if sensor == 2:
                    yaw += 0.8
                if sensor == 3:
                    yaw += 0.0
                car = [*position[:2], yaw-rotation]
                combined.append(car)
        
        nmsThresh = 0.5
        new = []
        for i in range(len(combined)):
            car = combined[i]
            unique = True
            for j in range(i+1, len(combined)):
                oCar = combined[j]
                dx = car[0] - oCar[0]
                dy = car[1] - oCar[1]
                dist = ((dx**2) + (dy**2))**0.5
                if dist < nmsThresh:
                    unique = False
                    break
            if unique == True:
                new.append(car)

        if len(new):
            for rotation in [0, 1, 2, 3, 4, 5, 6]:
                r = np.identity(4)
                orientationR = R.from_euler('z', rotation)
                r = R.from_euler('z', rotation)
                r = r.as_matrix()
                r = np.pad(r, [(0, 1), (0, 1)])
                r[-1][-1] = 1

                newScan = np.copy(dataFrame)
                for point in range(len(newScan)):
                    newScan[point] = np.matmul(r, newScan[point])
                newAnnotations = np.copy(new)
                for i in range(len(newAnnotations)):
                    v = [*new[i][:2], 1, 1]
                    v = np.matmul(r, v)[:2]
                    o = orientationR.as_euler('zxy', degrees=False)
                    o = o[0]
                    o += new[i][2]
                    newAnnotations[i] = [*v, o]

                utils.drawImgFromPoints("../annotations/training-la/imgs/" + filename + "-" + str(frameIdx) + "-" + '{0:.2f}'.format(rotation) + ".png", newScan, [], [], [], [], 5)
            
                utils.drawAnnoFromPoints("../annotations/training-la/annotations/" + filename + "-" + str(frameIdx) + "-" + '{0:.2f}'.format(rotation) + ".png", [], [], [], newAnnotations, [], 3)
      
if __name__ == "__main__":

    for method in datasetPaths:
        print("Evaluation method %s" % (method))

        for files in os.walk(method):
            for filename in files[2]:
                if filename[-7:] == ".pickle":
                    if "2-2" in filename:
                        continue
                    filePart = datasetPaths[method]
                    evaluateFile(filename, method, filePart)
    print("Generating Graphs")

