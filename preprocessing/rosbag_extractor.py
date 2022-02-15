#!/usr/bin/python
import copy
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

datasetPath = "../data/rosbags/"
outputPath = "../data/extracted/"
lp = lg.LaserProjection()

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Extractor(multiprocessing.Process):
    def __init__(self, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path
        self.topicBuf = [None, None, None]
        self.lastX, self.lastY = None, None
        self.distThresh = 0.5
        self.fileCounter = 0
        self.position = None
        self.scans = []
        self.trans = []
        self.ts = []

        self.scanTopics = ["/back_left/sick_safetyscanners/scan", 
                "/back_right/sick_safetyscanners/scan", 
                "/front/sick_safetyscanners/scan",
                "/back_middle/scan"]

    def run(self):
        
        print("Process spawned for file %s" % (self.filename), flush = True)
        time.sleep(1)

        print("Reading tfs", flush = True)
        try:
            with suppress_stdout_stderr():
                 self.bagtf = BagTfTransformer(self.path + "/" + self.filename)
        except:
            print("Process finished, bag failed for file %s" % (self.filename), flush = True)
            return
        print("Reading bag", flush = True)
        for topic, msg, t in rosbag.Bag(self.path + "/" + self.filename).read_messages():

            if topic in self.scanTopics:
                for i in range(3):
                    if topic == self.scanTopics[i]:
                        self.topicBuf[i] = msg
                if topic == self.scanTopics[-1]:
                    self.processScan(msg, msg.header.stamp)
            if topic == "/odom":
                self.position = msg.pose.pose.position
                self.orientation = msg.pose.pose.orientation
        self.saveScans()
        print("Process finished for file %s" % (self.filename), flush = True)

    def odometryMoved(self):

        if self.lastX == None:
            if self.position == None:
                return
            self.lastX = self.position.x
            self.lastY = self.position.y
            return

        diffx = self.position.x - self.lastX
        diffy = self.position.y - self.lastY
        dist = ((diffx**2)+(diffy**2))**0.5
        
        if dist > self.distThresh:
            self.lastX = self.position.x
            self.lastY = self.position.y
            return True
        return False

    def combineScans(self, msgs, t):

        points = {}
        for msg in msgs:
            points[msg.header.frame_id] = []
            if t - msg.header.stamp > rospy.Duration(1, 0):
                print("Old Scan present", flush = True)
                return [], [], []
        for idx in range(len(msgs)):
            msgs[idx] = lp.projectLaser(msgs[idx])
            msg = pc2.read_points(msgs[idx])

            translation, quaternion = None, None
            
            try:
                with suppress_stdout_stderr():
                    translation, quaternion = self.bagtf.lookupTransform("base_link", msgs[idx].header.frame_id, msgs[idx].header.stamp)
            except:
                print("Error finding tf", flush = True)
                return [], [], []
            r = R.from_quat(quaternion)
            mat = r.as_matrix()
            mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
            mat[0][-1] += translation[0]
            mat[1][-1] += translation[1]
            mat[2][-1] += translation[2]
            mat[3][-1] = 1

            for point in msg:
                intensity = point[3]
                point = np.array([*point[:3], 1])
                point = np.matmul(mat, point)
                point[-1] = intensity
                points[msgs[idx].header.frame_id].append(point)

        translation, quaternion = None, None
        try:
            with suppress_stdout_stderr():
                translation, quaternion = self.bagtf.lookupTransform("odom", "base_link", msgs[0].header.stamp)
        except:
            print("Error finding tf here", flush = True)
            return [], [], []
        r = R.from_quat(quaternion)
        mat = r.as_matrix()
        mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
        mat[0][-1] += translation[0]
        mat[1][-1] += translation[1]
        mat[2][-1] += translation[2]
        mat[3][-1] = 1

        trans = mat 

        for key, value in points.items():
            points[key] = self.filterRobot(value)

        return points, trans, t

    def filterRobot(self, points):

        newPoints = []

        for i in range(len(points)):
            p = points[i]
            if p[0] < 0.25 and p[0] > -1.4:
                if p[1] < 1.5 and p[1] > -1.5:
                    continue
            if p[0] < -1.3 and p[0] > -4.8:
                if p[1] < 1.3 and p[1] > -1.3:
                    continue
            newPoints.append(p)

        return newPoints

    def saveScans(self):

        if len(self.scans) == 0:
            return

        os.makedirs(outputPath, exist_ok=True)
        fn = os.path.join(outputPath, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"scans": self.scans, "trans": self.trans, "ts": self.ts}, f, protocol=2)

    def processScan(self, msg, t):

        t = rospy.Time(t.secs, t.nsecs)

        if None in self.topicBuf:
            return

        msgs = copy.deepcopy(self.topicBuf)
        #self.topicBuf = [None, None, None]
        msgs.append(msg)

        #if not self.odometryMoved():
        #    return
        #print("Robot moved!", flush = True)

        combined, trans, ts = self.combineScans(msgs, t)
        if len(combined) == 0:
            return
        self.scans.append(combined)
        self.trans.append(trans)
        self.ts.append(ts)
        self.fileCounter += 1

if __name__ == "__main__":
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-4:] == ".bag":
                jobs.append(Extractor(files[0], filename))
    maxCores = 7
    limit = maxCores
    batch = maxCores 
    print("Spawned %i processes" % (len(jobs)), flush = True)
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()



