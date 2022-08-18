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

extractEveryFrame = False
datasetPath = "../data/rosbags/"
outputPath = "../data/extracted/"
lp = lg.LaserProjection()

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Extractor(multiprocessing.Process):
    def __init__(self, path, folder, filename):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.folder = folder
        self.filename = filename

        self.lastX, self.lastY = None, None
        self.distThresh = 0.1
        self.fileCounter = 0
        self.position = None
        self.scans = []
        self.trans = []
        self.ts = []

        self.scanTopics = ["/back_right/sick_safetyscanners/scan", 
                "/front/sick_safetyscanners/scan",
                "/back_low/scan",
                "/back_left/sick_safetyscanners/scan",
                "/back_middle/scan"]
        self.synchroniseToTopic = self.scanTopics[-1] #MUST always be last from list (back middle higher fps)
        self.topicBuf = []
        for _ in range(len(self.scanTopics)-1):
            self.topicBuf.append(None)

        self.pointcloudScanTopic = ["/os_cloud_node/points"]
        self.pointcloudScanBuf = None
        self.pointclouds = []

    def run(self):
        
        print("Process spawned for file %s" % (os.path.join(self.path, self.folder, self.filename)), flush = True)

        print("Reading tfs", flush = True)
        try:
            with suppress_stdout_stderr():
                self.bagtf = BagTfTransformer(os.path.join(self.path, self.folder, self.filename))
        except:
            print("Process finished, bag failed for file %s" % (self.filename), flush = True)
            return
        print("Reading bag", flush = True)
        for topic, msg, t in rosbag.Bag(os.path.join(self.path, self.folder, self.filename)).read_messages():
            if topic in self.pointcloudScanTopic:
                self.pointcloudScanBuf = msg
            if topic in self.scanTopics:
                if topic == self.synchroniseToTopic:
                    self.processScan(msg, msg.header.stamp)
                else:
                    for i in range(len(self.scanTopics)):
                        if topic == self.scanTopics[i]:
                            self.topicBuf[i] = msg
            if topic == "/odom":
                self.position = msg.pose.pose.position
                self.orientation = msg.pose.pose.orientation
        self.saveScans()
        print("Process finished for file %s, %i frames" % (self.filename, self.fileCounter), flush = True)

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
            print("Skipping saving empty bag")
            return

        os.makedirs(os.path.join(outputPath, self.folder), exist_ok=True)

        fn = os.path.join(outputPath, self.folder, self.filename + ".scans.pickle")
        print("Writing %s" % (fn))
        with open(fn, "wb") as f:
            pickle.dump({"scans": self.scans}, f)
        fn = os.path.join(outputPath, self.folder, self.filename + ".3d.pickle")
        print("Writing %s" % (fn))
        with open(fn, "wb") as f:
            pickle.dump({"pointclouds": self.pointclouds}, f)
        fn = os.path.join(outputPath, self.folder, self.filename + ".data.pickle")
        print("Writing %s" % (fn))
        with open(fn, "wb") as f:
            pickle.dump({"trans": self.trans, "ts": self.ts}, f)

    def processScan(self, msg, t):

        t = rospy.Time(t.secs, t.nsecs)

        if None in self.topicBuf:
            print("None in topic buf!")
            for idx in range(len(self.topicBuf)):
                if self.topicBuf[idx] is None:
                    print("Topic missing: ", self.scanTopics[idx])
            return
        if self.pointcloudScanBuf is None:
            print("Unable to grab 3d scan", self.filename)
            return

        msgs = copy.deepcopy(self.topicBuf)
        msgs.append(msg)

        if extractEveryFrame == False:
            if not self.odometryMoved():# and self.filename not in self.gtBags:
                return
        #print("Robot moved!", flush=True)

        combined, trans, ts = self.combineScans(msgs, t)
        if len(combined) == 0:
            return
        for key in combined.keys():
            combined[key] = np.array(combined[key])
            combined[key] = combined[key].reshape(combined[key].shape[0], 4)
        self.scans.append(combined)
        self.trans.append(trans)
        self.ts.append(ts)

        #pcpoints = self.cloudToArray(self.pointcloudScanBuf)
        #for p in pcpoints:
        #    self.pointcloudfile.write("%f %f %f, " % (p[0], p[1], p[2]))
        #self.pointcloudfile.write("\n")
        self.pointclouds.append(self.cloudToArray(self.pointcloudScanBuf))
        self.fileCounter += 1

        #self.pointcloudScanBuf = None

    def cloudToArray(self, msg):

        translation, quaternion = None, None

        try:
            with suppress_stdout_stderr():
                translation, quaternion = self.bagtf.lookupTransform("base_link", msg.header.frame_id, msg.header.stamp)
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

        gen = list(pc2.read_points(msg, skip_nans=True))
        gen = np.array(gen)

        out = []
        for idx, point in enumerate(gen):
            #x, y, z, intensity, t, reflectivity, ring, ambient, range
            transPoint = [*point[:3], 1]
            transPoint = np.matmul(mat, transPoint)
            point = [*transPoint[:3], *point[3:]]
            if (point[0]**2 + point[1]**2)**0.5 < 2.5:
                continue
            point = np.array(point)
            out.append(point)
        out = np.array(out) 

        return out

if __name__ == "__main__":

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-4:] == ".bag":
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Extractor(path, folder, filename))
    maxCores = 7
    limit = maxCores
    batch = maxCores 
    print("Spawned %i processes" % (len(jobs)), flush = True)
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
            time.sleep(0.2)
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()

