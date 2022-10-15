#!/usr/bin/python
import copy
import utils
import random
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
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
import numpy as np

datasetPath = "../data/lanoising/5-planar-gt"
annotationSource = "extrapolated"
laserPointsFields = ["scans", "lanoising"]
outputPath = "../annotations/"
lp = lg.LaserProjection()
movementThreshold = 1.0
gtPath = "../data/gt"
gtBags = []
saltPepperNoise = 0.0000

#augmentations
flipV = False
rotations = [0, 1, 2, 3, 4, 5, 6]

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Annotator(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path
        self.fileCounter = 0

    def run(self):
        
        if self.filename in gtBags:
            print("Process spawned for GT file %s" % (self.filename), flush = True)
        else:
            print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)
        scanfn = self.filename[:-12] + ".scans.pickle"
        with open(os.path.join(self.path, scanfn), "rb") as f:
            self.data.update(pickle.load(f))

        self.annotate()

    def drawCar(self, centreX, centreY, fullHeight, fullWidth, angle, img, debugimg=None, annotationIdx=None):

        angle = angle % (math.pi * 2)
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        a = [int(centreY + alpha * fullHeight - beta * fullWidth), int(centreX - beta * fullHeight - alpha * fullWidth)]
        b = [int(centreY - alpha * fullHeight - beta * fullWidth), int(centreX + beta * fullHeight - alpha * fullWidth)]
        c = [int(2 * centreY - a[0]), int(2 * centreX - a[1])]
        d = [int(2 * centreY - b[0]), int(2 * centreX - b[1])]

        contours = np.array([a, b, c, d])
        if annotationIdx == None:
            colour = lambda : [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            cv2.fillPoly(img, pts = [contours], color = colour())
        else:
            colour = annotationIdx
            cv2.fillPoly(img, pts = [contours], color = colour)

        if debugimg is not None:
            cv2.fillPoly(debugimg, pts = [contours], color =(255,0,255))

            debugimg[centreX, centreY] = [255, 0, 255]
            debugimg[a[0], a[1]] = [125, 0, 128]
            debugimg[b[0], b[1]] = [125, 255, 0]
            debugimg[c[0], c[1]] = [125, 255, 0]
            debugimg[d[0], d[1]] = [125, 255, 0]

    def testImg(self, img):

        mask = np.array(img)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask[:,:,0] == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            if xmin == xmax or ymin == ymax:
                return True
        if len(boxes) == 0:
            return True
        return False

    def drawAnnotation(self, filename, frame, annotations):
        
        res = 1024
        scale = 25
        ctr = 0

        combineMasks = True

        if combineMasks == False:
            for annotation in annotations:

                accum = np.zeros((res, res, 3))
                
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(3.5*scale), int(5.5*scale)

                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img)
                fn = os.path.join(outputPath, "annotations", self.filename + "-" + str(frame) + "-" + str(ctr) + ".png")
                if testImg(accum):
                    return True

                cv2.imwrite(fn, accum)
                ctr += 1

        else:
            accum = np.zeros((res, res, 1))
            annotationIdx = 1
            for annotation in annotations:
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(2.4*scale), int(4.85*scale)
                #self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img, annotationIdx)
                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, annotationIdx=annotationIdx)
                annotationIdx += 1
            #fn = os.path.join(foldername, self.filename + "-" + str(frame) + ".png")
            if self.testImg(accum):
                return True
            cv2.imwrite(filename, accum)

        return False

    def tripletOrientation(self, a, b, c):
        orr = ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))
        if orr > 0:
            return 1
        elif orr < 0:
            return -1
        return 0

    def colinear(self, a, b, c):
        if ((b[0] <= max(a[0], c[0])) and (b[0] >= min(a[0], c[0])) and (b[1] <= max(a[1], c[1])) and (b[1] >= min(a[1], c[1]))):
            return True
        else:
            return False

    def lineIntersect(self, a, b, c, d):
        o1 = self.tripletOrientation(a, b, c)
        o2 = self.tripletOrientation(a, b, d)
        o3 = self.tripletOrientation(c, d, a)
        o4 = self.tripletOrientation(c, d, b)

        if o1 != o2 and o3 != o4:
            return True

        if ((o1 == 0) and self.colinear(a, c, b)):
            return True
        if ((o2 == 0) and self.colinear(a, d, b)):
            return True
        if ((o3 == 0) and self.colinear(c, a, d)):
            return True
        if ((o4 == 0) and self.colinear(c, b, d)):
            return True
        return False

    def getBoundaryPoints(self, poly):

        centreX = poly[0]#*scale) + (res//2)
        centreY = poly[1]#*scale) + (res//2)
        angle = poly[2] % (math.pi*2)
        width = 2.3# * scale
        height = 4.5# * scale
        height = 2.3# * scale
        width = 4.5# * scale

        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5

        a = [centreX - beta * height - alpha * width, centreY + alpha * height - beta * width]
        b = [centreX + beta * height - alpha * width, centreY - alpha * height - beta * width]
        c = [2 * centreX - a[0], 2 * centreY - a[1]]
        d = [2 * centreX - b[0], 2 * centreY - b[1]]

        return a, b, c, d

    def isInsideAnnotation(self, pos, poly):

        poly = self.getBoundaryPoints(poly)

        counterPos = (9999, pos[1])
        intersects = 0

        for idx in range(len(poly)):
            pa = poly[idx]
            pb = poly[0]
            if idx != len(poly)-1:
                pb = poly[idx+1]

            if self.lineIntersect(pos, counterPos, pa, pb):
                intersects += 1

        return intersects % 2

    def getInAnnotation(self, scan, annotations):

        carPoints = []
        nonCarPoints = []

        for point in scan:

            inAnnotation = False

            for annotation in annotations:
                if self.isInsideAnnotation([point[0], point[1]], annotation):
                    inAnnotation = True
                    break

            if inAnnotation:
                carPoints.append(point)
            else:
                nonCarPoints.append(point)

        return carPoints, nonCarPoints

    def getAnnotation(self, scan, annotations):

        backgroundPoints = []
        points = []
        cols = []
        debugbackgroundPoints = []
        debugpoints = []
        debugcols = []

        for point in scan:
            inAnnotation = False
            for annotation in annotations:
                if self.isInsideAnnotation([point[0], point[1]], annotation):
                    inAnnotation = True
                    break
            if inAnnotation == False:
                backgroundPoints.append([point[0], point[1], point[2]])
                debugbackgroundPoints.append([point[0], point[1], point[2]])
            else:
                cols.append([255, 0, 255])
                points.append([i[0], i[1], i[2]])
                debugcols.append([255, 0, 255])
                debugpoints.append([i[0], i[1], i[2]])

        for annotation in annotations:
            poly = self.getBoundaryPoints(annotation)
            for i in poly:
                debugpoints.append(i)
                debugcols.append([255, 0, 0])
            debugpoints.append([annotation[0], annotation[1]])
            debugcols.append([0, 25, 255])

        return backgroundPoints, points, cols, debugbackgroundPoints, debugpoints, debugcols

    def saveCloud(self, filename, scan):

        #scan is just array of points, put them into file
        pass

    def saveAnnotations(self, filename, scan, annotations):

        return
        for annotation in annotations:
            poly = self.getBoundaryPoints(annotation)
            for i in poly:
                debugpoints.append(i)
                debugcols.append([255, 0, 0])
            debugpoints.append([annotation[0], annotation[1]])
            debugcols.append([0, 25, 255])

    def annotate(self):

        oldx, oldy, = 999, 999
        for frame in range(len(self.data["scans"])):
            annotations = self.data[annotationSource][frame]

            if self.filename not in gtBags:
                #throw away frames without much movement
                trans = self.data["trans"][frame]
                x = trans[0][-1]
                y = trans[1][-1]

                diffx = x - oldx
                diffy = y - oldy
                dist = ((diffx**2)+(diffy**2))**0.5

                if dist < movementThreshold:
                    continue
                oldx = x
                oldy = y
                if len(annotations) == 0:
                    #for training purposes, dont teach on empty images
                    continue

            for scanField in laserPointsFields:

                scan = self.data[scanField][frame]
                combined = utils.combineScans(scan)

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

                    if self.filename in gtBags:
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + ".png")
                        carPoints, nonCarPoints = self.getInAnnotation(newScan, newAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, newAnnotations)
                        if badAnnotation:
                            continue

                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=3)
                        

                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + ".")
                        self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + ".")
                        self.saveAnnotations(fn, newScan, newAnnotations)

                    #raw
                    if self.filename not in gtBags and len(newAnnotations) and "2022-02-15-15-08-59" not in self.filename:
                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        carPoints, nonCarPoints = self.getInAnnotation(newScan, newAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, newAnnotations)
                        if badAnnotation:
                            continue

                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=3, saltPepperProb=saltPepperNoise)
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                        utils.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".")
                        self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".")
                        self.saveAnnotations(fn, newScan, newAnnotations)
                    
                    #flip
                    if flipV and len(newAnnotations) and self.filename not in gtBags and "2022-02-15-15-08-59" not in self.filename:

                        fScan = np.copy(newScan)
                        for point in range(len(newScan)):
                            fScan[point][0] = -fScan[point][0]
                        fAnnotations = np.copy(newAnnotations)
                        for i in range(len(annotations)):
                            fAnnotations[i][0] = -fAnnotations[i][0]
                            fAnnotations[i][2] = -fAnnotations[i][2]

                        fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        carPoints, nonCarPoints = self.getInAnnotation(fScan, fAnnotations)
                        badAnnotation = self.drawAnnotation(fn, frame, fAnnotations) 
                        if badAnnotation:
                            continue

                        fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        utils.drawImgFromPoints(fn, fScan, [], [], [], [], dilation=3, saltPepperProb=saltPepperNoise)
                        
                        fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                        utils.drawImgFromPoints(fn, fScan, [], [], fAnnotations, [], dilation=None)

                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                        self.saveCloud(fn, newScan)
                        fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                        self.saveAnnotations(fn, newScan, newAnnotations)

if __name__ == "__main__":

    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag.data.pickle"
                gtBags.append(fn)
    print("GT files")
    print(gtBags)

    for scanField in laserPointsFields:
        os.makedirs(os.path.join(outputPath, scanField, "moving"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "mask", "all", "imgs"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "mask", "all", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "mask", "all", "debug"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "pointcloud", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "pointcloud", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(outputPath, scanField, "pointcloud", "debug"), exist_ok=True)
    
    jobs = []
    print(datasetPath)
    for files in os.walk(datasetPath):
        for filename in files[2]:
            print(filename)
            if ".data.pickle" in filename:
                jobs.append(Annotator(files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 20
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

