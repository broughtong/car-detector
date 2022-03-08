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

datasetPath = "../data/results/detector-s"
outputPath = "../data/results/temporal-prc-"
visualisationPath = "../visualisation/temporal-prc-"

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Interpolator(multiprocessing.Process):
    def __init__(self, path, filename, interpolateFrames, detectionDistance, extrapolateFrames):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path
        self.interpolateFrames = interpolateFrames
        self.detectionDistance = detectionDistance
        self.extrapolateFrames = extrapolateFrames

    def run(self):

        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        self.interpolate()
        
        self.data["scans"] = []
        self.data["trans"] = []
        self.data["ts"] = []
        self.data["annotations"] = []
        
        folder = outputPath + "%s-%s-%s" % (str(self.interpolateFrames), str(self.detectionDistance), str(self.extrapolateFrames))
        os.makedirs(folder, exist_ok=True)
        fn = self.filename
        with open(os.path.join(folder, fn), "wb") as f:
            pickle.dump(self.data, f, protocol=2)

    def interpolate(self):

        scans = self.data["scans"]
        trans = self.data["trans"]
        ts = self.data["ts"]
        annotations = self.data["annotations"]
        annotationsOdom = []

        robotPoses = []

        #move everything into the odom frame
        for idx in range(len(scans)):
            robopose = np.array([-4.2, 0, 0, 1])
            mat = np.array(trans[idx])
            robopose = np.matmul(mat, robopose)[:2]
            robotPoses.append(robopose)
            r = R.from_matrix(mat[:3, :3])
            yaw = r.as_euler('zxy', degrees=False)
            yaw = yaw[0]
            detections = []

            for det in range(len(annotations[idx])):
                point = np.array([*annotations[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])

                orientation = annotations[idx][det][2] + yaw
                detections.append([*point, orientation])

            annotationsOdom.append(detections)
        with suppress_stdout_stderr():
            annotationsOdom = np.array(annotationsOdom)

        #if two detections close to each other in odom frame inside some temporal window
        #lerp between them
        lerped = [] #includes real detections
        lerpedOnly = [] #exclusively lerped
        for i in range(len(scans)):
            lerped.append([])
            lerpedOnly.append([])
        for mainIdx in range(len(scans)):
            for car in annotationsOdom[mainIdx]: #for each car in the current scan:

                isConsistent = False
                futurePosition = None
                nFrames = None
                toGo = min(mainIdx + self.interpolateFrames, len(scans))

                #we have some car, look for any future detections
                for smallWindowIdx in range(mainIdx+1, toGo): 
                    for possibleMatchIdx in range(len(annotationsOdom[smallWindowIdx])): #for each car in each other frame in small window
                        otherCar = annotationsOdom[smallWindowIdx][possibleMatchIdx]

                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            isConsistent = True
                            futurePosition = otherCar
                            nFrames = smallWindowIdx - mainIdx
                            break
                    if isConsistent:
                        break

                if isConsistent == False:
                    continue
                if nFrames == 1:
                    continue

                startX, endX = car[0], futurePosition[0]
                startY, endY = car[1], futurePosition[1]
                startR, endR = car[2], futurePosition[2]

                for i in range(1, nFrames):
                    interval = i / nFrames
                    x = self.lerp(startX, endX, interval)
                    y = self.lerp(startY, endY, interval)
                    r = self.lerp(startR, endR, interval, rot=True)

                    robopose = robotPoses[mainIdx+i]
                    dx = x - robopose[0] 
                    dy = y - robopose[1]
                    dist = ((dx**2) + (dy**2))**0.5
    
                    if dist > 1.8: #stop detections from under the robot
                        lerped[mainIdx+i].append([x, y, r])
                        lerpedOnly[mainIdx+i].append([x, y, r])

        #add real detections to lerped dataset
        for frameIdx in range(len(annotationsOdom)):
            for detectionIdx in range(len(annotationsOdom[frameIdx])):
                lerped[frameIdx].append(annotationsOdom[frameIdx][detectionIdx])

        #extrapolate
        extrapolated = []
        extrapolatedOnly = []
        for i in range(len(scans)):
            extrapolated.append([])
            extrapolatedOnly.append([])
        for frameIdx in range(len(lerped)):
            for detectionIdx in range(len(lerped[frameIdx])):

                #for each detection, we will check forward and back 1 frame

                #forward
                if frameIdx < len(lerped)-1:
                    nextFrame = lerped[frameIdx+1]
                    foundCar = False
                    for otherDetectionIdx in range(len(nextFrame)):
                        car = lerped[frameIdx][detectionIdx]
                        otherCar = lerped[frameIdx+1][otherDetectionIdx]
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                    if foundCar == False:
                        for i in range(frameIdx+1, frameIdx + self.extrapolateFrames):
                            if i <= len(lerped)-1: 
                                robopose = robotPoses[i]
                                dx = car[0] - robopose[0] 
                                dy = car[1] - robopose[1]
                                dist = ((dx**2) + (dy**2))**0.5
                
                                if dist > 1.8: #stop detections from under the robot
                                    extrapolated[i].append(car)
                                    extrapolatedOnly[i].append(car)
                #backward
                if frameIdx > 0:
                    prevFrame = lerped[frameIdx-1]
                    foundCar = False
                    for otherDetectionIdx in range(len(prevFrame)):
                        car = lerped[frameIdx][detectionIdx]
                        otherCar = lerped[frameIdx-1][otherDetectionIdx]
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                    if foundCar == False:
                        for i in range(frameIdx - self.extrapolateFrames, frameIdx):
                            if i >= 0:
                                robopose = robotPoses[i]
                                dx = car[0] - robopose[0] 
                                dy = car[1] - robopose[1]
                                dist = ((dx**2) + (dy**2))**0.5
                
                                if dist > 1.8: #stop detections from under the robot
                                    extrapolated[i].append(car)
                                    extrapolatedOnly[i].append(car)

        #add real detections to extrapolated
        for frameIdx in range(len(annotationsOdom)):
            for detectionIdx in range(len(lerped[frameIdx])):
                extrapolated[frameIdx].append(lerped[frameIdx][detectionIdx])

        #self.data["annotationsOdom"]  = annotationsOdom
        #self.data["lerpedOdom"] = lerped
        #self.data["lerped"] = []
        #self.data["extrapOdom"] = extrapolated
        self.data["extrapolated"] = []

        #move back to robot frame
        for idx in range(len(scans)):
            mat = np.array(trans[idx])
            mat = np.linalg.inv(mat)
            r = R.from_matrix(mat[:3, :3])
            yaw = r.as_euler('zxy', degrees=False)
            yaw = yaw[0]
            detections = []
            detectionsExtrap = []

            for det in range(len(lerped[idx])):
                point = np.array([*lerped[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = lerped[idx][det][2] + yaw
                detections.append([*point, orientation])
                
            for det in range(len(lerpedOnly[idx])):
                point = np.array([*lerpedOnly[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                lerpedOnly[idx][det] = point
            
            for det in range(len(extrapolated[idx])):
                point = np.array([*extrapolated[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = extrapolated[idx][det][2] + yaw
                detectionsExtrap.append([*point, orientation])
                
            for det in range(len(extrapolatedOnly[idx])):
                point = np.array([*extrapolatedOnly[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                extrapolatedOnly[idx][det] = point

            #self.data["lerped"].append(detections)
            self.data["extrapolated"].append(detectionsExtrap)
        with suppress_stdout_stderr():
            #self.data["lerped"] = np.array(self.data["lerped"])
            self.data["extrapolated"] = np.array(self.data["extrapolated"])


        self.fileCounter = 0
        folder = visualisationPath + "%s-%s-%s" % (str(self.interpolateFrames), str(self.detectionDistance), str(self.extrapolateFrames))
        os.makedirs(folder, exist_ok=True)
        for i in range(len(scans)):

            points = np.concatenate([scans[i]["sick_back_left"], scans[i]["sick_back_right"], scans[i]["sick_back_middle"]])
            dets = []
            colours = []
            for j in range(len(self.data["lerped"][i])):
                dets.append([self.data["lerped"][i][j][0], self.data["lerped"][i][j][1]])
                colours.append([0, 255, 255])
            for j in range(len(lerpedOnly[i])):
                dets.append([lerpedOnly[i][j][0], lerpedOnly[i][j][1]])
                colours.append([255, 0, 0])
            for j in range(len(extrapolatedOnly[i])):
                dets.append([extrapolatedOnly[i][j][0], extrapolatedOnly[i][j][1]])
                colours.append([0, 255, 0])

            annotations = self.data["extrapolated"][i]

            #self.pointsToImgsDrawWheels(points, "interpolated", dets, colours)
            fn = os.path.join(folder, "%s-%s.png" % (self.filename, self.fileCounter))
            utils.drawImgFromPoints(fn, points, dets, colours, annotations)
            self.fileCounter += 1

    def lerp(self, a, b, i, rot=False):
        if rot:
            a = a % math.pi
            b = b % math.pi
            if abs(a - b) > (math.pi/2):
                if a < b:
                    a += math.pi
                else:
                    b += math.pi
        return a + i * (b - a)

if __name__ == "__main__":
    
    #jobs = []
    #for files in os.walk(datasetPath):
    #    for filename in files[2]:
    #        if filename[-7:] == ".pickle":
    #            jobs.append(Interpolator(files[0], filename, 200, 0.4, 200))

    jobs = []
    for i in range(0, 1000, 50):
        jobs.append(Interpolator("../data/results/detector-s/", "2020-11-17-13-47-41.bag.pickle", i, 0.4, 0))

    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 1
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

