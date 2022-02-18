#!/usr/bin/python
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

datasetPath = "../data/results/temporal-s"
annotationSource = "extrapolated"
outputPath = "../annotations/pointnet/all"
lp = lg.LaserProjection()
movementThreshold = 0.5

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
        
        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        self.annotate()

    def pointsToImgsDrawWheels(self, points, location, frame, wheels, colours):

        res = 1024
        scale = 20
        accum = np.zeros((res, res, 3))
        accum.fill(255)

        for point in points:
            x, y = point[:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                accum[x+int(res/2), y+int(res/2)] = [0, 0, 0]
            except:
                pass

        for wheel in range(len(wheels)):
            x, y = wheels[wheel][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                accum[x+int(res/2), y+int(res/2)] = colours[wheel]
            except:
                pass

            try:
                accum[x+int(res/2)+1, y+int(res/2)+1] = colours[wheel] 
            except:
                pass
            try:
                accum[x+int(res/2)+1, y+int(res/2)-1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)-1, y+int(res/2)+1] = colours[wheel]
            except:
                pass
            try:
                accum[x+int(res/2)-1, y+int(res/2)-1] = colours[wheel]
            except:
                pass

        fn = os.path.join(outputPath, location, self.filename + "-" + str(frame) + ".png")
        cv2.imwrite(fn, accum)

        return accum

    def drawCar(self, centreX, centreY, fullHeight, fullWidth, angle, img, debugimg, annotationIdx=None):

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

        cv2.fillPoly(debugimg, pts = [contours], color =(255,0,255))

        debugimg[centreX, centreY] = [255, 0, 255]
        debugimg[a[0], a[1]] = [125, 0, 128]
        debugimg[b[0], b[1]] = [125, 255, 0]
        debugimg[c[0], c[1]] = [125, 255, 0]
        debugimg[d[0], d[1]] = [125, 255, 0]

    def drawAnnotation(self, img, frame, annotations):
        
        res = 1024
        scale = 20
        ctr = 0

        combineMasks = True

        if combineMasks == False:
            for annotation in annotations:

                accum = np.zeros((res, res, 3))
                
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(2.3*scale), int(4.5*scale)

                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img)
                fn = os.path.join(outputPath, "annotations", self.filename + "-" + str(frame) + "-" + str(ctr) + ".png")
                cv2.imwrite(fn, accum)
                ctr += 1

        else:
            accum = np.zeros((res, res, 1))
            annotationIdx = 1
            for annotation in annotations:
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(2.3*scale), int(4.5*scale)
                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img, annotationIdx)
                annotationIdx += 1
            fn = os.path.join(outputPath, "annotations", self.filename + "-" + str(frame) + ".png")
            cv2.imwrite(fn, accum)

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

    def getAnnotation(self, scan, frameIdx, annotations):

        backgroundPoints = []
        points = []
        cols = []
        debugbackgroundPoints = []
        debugpoints = []
        debugcols = []

        for i in scan:
            inAnnotation = False
            for annotation in annotations:
                if self.isInsideAnnotation([i[0], i[1]], annotation):
                    inAnnotation = True
                    break
            if inAnnotation == False:
                backgroundPoints.append([i[0], i[1], i[2]])
                debugbackgroundPoints.append([i[0], i[1], i[2]])
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

    def annotate(self):

        oldx, oldy, = 999, 999
        for frame in range(len(self.data["scans"])):
            
            #throw away frames without much movement
            trans = self.data["trans"][frame]
            x = trans[0][-1]
            y = trans[1][-1]

            diffx = x - oldx
            diffy = y - oldy
            dist = ((diffx**2)+(diffy**2))**0.5

            if dist < movementThreshold:
                continue

            #image dataset
            scan = self.data["scans"][frame]
            combined = np.concatenate([scan["sick_back_left"], scan["sick_back_right"], scan["sick_front"], scan["sick_back_middle"]])

            annotations = self.data[annotationSource][frame]
            if len(annotations) > 0:

                backgroundPoints, points, cols, debugbp, debugp, debugcol = self.getAnnotation(combined, frame, annotations)
                self.pointsToImgsDrawWheels(debugbp, "debug", str(frame), debugp, debugcol)

                oldx = x
                oldy = y

if __name__ == "__main__":

    os.makedirs(os.path.join(outputPath, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(outputPath, "debug"), exist_ok=True)
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Annotator(files[0], filename))
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

