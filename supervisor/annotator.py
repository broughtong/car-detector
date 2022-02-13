#!/usr/bin/python
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

datasetPath = "./result/lanoise/"
datasetPath = "./result_small/interpolated/"
lp = lg.LaserProjection()

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

        with open(self.path + self.filename, "rb") as f:
            self.data = pickle.load(f)

        self.annotate()

        with open(os.path.join("result", "interpolated", self.filename), "wb") as f:
            pickle.dump(self.data, f, protocol=2)

    def pointsToImgsDrawWheels(self, points, location, frame, wheels, colours):

        res = 1024
        scale = 20
        accum = np.zeros((res, res, 3))

        for point in points:
            x, y = point[:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                accum[x+int(res/2), y+int(res/2)] = [255, 255, 255]
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

        kernel = np.ones((5, 5), 'uint8')
        dilate_img = cv2.dilate(accum, kernel, iterations=1)

        fn = "annotations/maskrcnn/" + location + "/" + self.filename + "-" + str(frame) + ".png"
        #cv2.imwrite(fn, accum)
        cv2.imwrite(fn, dilate_img)

    def drawAnnotations(self, frame, annotations):

        res = 1024
        scale = 20

        for idx in range(len(annotations)):

            i = annotations[idx]
            i = [50, 10, 0]
            print(i)

            img = np.zeros((res, res, 3), np.uint8)

            carSize = (4.38*scale, 1.84*scale)
            carPos = (int(scale*i[0])+int(res/2), int(scale*i[1])+int(res/2))

            print(carPos)
            rect = (carPos, carSize, i[2] * 180/ 3.141592)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print(box)
            cv2.drawContours(img,[box],0,(255,255,255),-1)

            fn = "annotations/maskrcnn/annotations/" + self.filename + "-" + str(frame) + "-" + str(idx) + ".png"
            cv2.imwrite(fn, img)

    def annotate(self):

        for frame in range(len(self.data["scans"])):
            
            #first create image
            points = []
            for key, value in self.data["scans"][frame].items():
                if key == "sick_back_middle" or key == "sick_front":
                    continue
                for point in value:
                    points.append(point)

            self.pointsToImgsDrawWheels(points, "imgs", frame, [[15, 4]], [[255, 0, 255]])

            annotations = self.data["extrapolated"][frame]
            self.drawAnnotations(frame, annotations)
            break

if __name__ == "__main__":

    os.makedirs("./annotations/maskrcnn/imgs", exist_ok=True)
    os.makedirs("./annotations/maskrcnn/annotations", exist_ok=True)
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                jobs.append(Annotator(files[0], filename))
                break
    print("Spawned %i processes" % (len(jobs)), flush = True)
    limit = 12
    batch = 12
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()

