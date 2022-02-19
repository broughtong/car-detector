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
outputPath = "../annotations/maskrcnn/all"
lp = lg.LaserProjection()
movementThreshold = 0.5
gtPath = "../data/gt"
gtBags = []

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
                img = self.pointsToImgsDrawWheels(combined, "imgs", str(frame), [], [])

                self.drawAnnotation(img, frame, annotations)

                fn = os.path.join(outputPath, "debug", self.filename + "-" + str(frame) + ".png")
                cv2.imwrite(fn, img)
                if self.filename not in gtBags:
                    oldx = x
                    oldy = y

if __name__ == "__main__":

    for files in os.walk(gtPath):
        for fn in files[2]:
            fn = fn.split("-")[:-1]
            fn = "-".join(fn)
            fn += ".bag.pickle"
            gtBags.append(fn)

    os.makedirs(os.path.join(outputPath, "imgs"), exist_ok=True)
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

