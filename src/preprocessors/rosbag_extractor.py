#!/usr/bin/python
import copy
import rospy
import pickle
import math
import sys
import os
import rosbag
import time
import concurrent
import concurrent.futures
import multiprocessing
import tqdm
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
import numpy as np

extractEveryFrame = False
gtPath = "../data/gt"
datasetPath = "../data/rosbags/"

scansPath = "../data/scans"
pointcloudsPath = "../data/pointclouds"
dataPath = "../data/meta"

lp = lg.LaserProjection()

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Extractor():
    def __init__(self, path, folder, filename, queue, gtBags):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue
        self.gtbags = gtBags

        self.lastX, self.lastY = None, None
        self.distThresh = 2
        self.fileCounter = 0
        self.position = None
        self.scans = []
        self.trans = []
        self.ts = []

        self.gtbag = None
        for i in gtBags:
            if i[2] == self.filename:
                self.gtbag = i
        if self.gtbag:
            with open(os.path.join(*self.gtbag[:2], self.gtbag[3]), "rb") as f:
                self.gtbag = pickle.load(f)[0]

        self.scanTopics = ["/back_right/sick_safetyscanners/scan", 
                "/front/sick_safetyscanners/scan",
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
        
        self.queue.put("%s: Process Spawned" % (self.filename))

        self.bagtf = BagTfTransformer(os.path.join(self.path, self.folder, self.filename))
        try:
            with suppress_stdout_stderr():
                self.bagtf = BagTfTransformer(os.path.join(self.path, self.folder, self.filename))
        except:
            self.queue.put("%s: Bag failed (1)" % (self.filename))
            return 0

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
        self.queue.put("complete")
        if len(self.scans) != 0:
            self.queue.put("%s: Finished writing %i frames" % (self.filename, len(self.scans)))
        return len(self.scans)

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
                self.queue.put("%s: Old Scan present" % (self.filename))
                return [], [], []

        for idx in range(len(msgs)):
            msgs[idx] = lp.projectLaser(msgs[idx])
            msg = pc2.read_points(msgs[idx])

            translation, quaternion = None, None
            
            try:
                with suppress_stdout_stderr():
                    translation, quaternion = self.bagtf.lookupTransform("base_link", msgs[idx].header.frame_id, msgs[idx].header.stamp)
            except:
                self.queue.put("%s: Error finding tf (1) from %s to %s" % (self.filename, msgs[idx].header.frame_id, "base_link"))
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
            self.queue.put("%s: Error finding tf (2) from %s to %s" % (self.filename, "base_link", "odom"))
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
            self.queue.put("%s: Skipping saving empty bag" % (self.filename))
            return

        os.makedirs(os.path.join(scansPath, self.folder), exist_ok=True)
        os.makedirs(os.path.join(pointcloudsPath, self.folder), exist_ok=True)
        os.makedirs(os.path.join(dataPath, self.folder), exist_ok=True)

        fn = os.path.join(scansPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"scans": self.scans}, f)
        fn = os.path.join(pointcloudsPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"pointclouds": self.pointclouds}, f)
        fn = os.path.join(dataPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"trans": self.trans, "ts": self.ts}, f)

    def processScan(self, msg, t):

        t = rospy.Time(t.secs, t.nsecs)

        if None in self.topicBuf:
            return

        msgs = copy.deepcopy(self.topicBuf)
        msgs.append(msg)

        if self.gtbag is None:
            if extractEveryFrame == False:
                if not self.odometryMoved():# and self.filename not in self.gtBags:
                    return
        else:
            tFound = False
            for frame in self.gtbag:
                gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
                if gttime == t:
                    tFound = True
                    break
            if tFound == False:
                return

        combined, trans, ts = self.combineScans(msgs, t)
        if len(combined) == 0:
            return
        for key in combined.keys():
            combined[key] = np.array(combined[key])
            combined[key] = combined[key].reshape(combined[key].shape[0], 4)
        self.scans.append(combined)
        self.trans.append(trans)
        self.ts.append(ts)

        if self.pointcloudScanBuf is not None:
            self.pointclouds.append(self.cloudToArray(self.pointcloudScanBuf))
        else:
            self.pointclouds.append([])
        self.fileCounter += 1

        self.pointcloudScanBuf = None
        for i in range(len(self.topicBuf)):
            self.topicBuf[i] = None

    def cloudToArray(self, msg):

        translation, quaternion = None, None

        try:
            with suppress_stdout_stderr():
                translation, quaternion = self.bagtf.lookupTransform("base_link", msg.header.frame_id, msg.header.stamp)
        except:
            self.queue.put("%s: Error finding tf (3) from %s to %s" % (self.filename, msg.header.frame_id, "base_link"))
            return []

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

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, None):
        if type(item) == int:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":
    
    gtBags = []
    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                origFn = fn
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag"
                gtBags.append([gtPath, files[0][len(gtPath)+1:], fn, origFn])

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-4:] == ".bag":
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Extractor(path, folder, filename, queue, gtBags))
    
    listenProcess = multiprocessing.Process(target=listener, args=(queue, len(jobs)))
    listenProcess.start()

    workers = 3
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)

        for future in futures:
            try:
                res = future.result()
                if res is not None:
                    queue.put(res)
            except Exception as e:
                queue.put("P Exception: " + str(e))

    results = []
    for i, job in enumerate(jobs):
        f = futures[i]
        value = [os.path.relpath(job.path, "../data/"), job.folder, job.filename, f.result()]
        results.append(value)

    with open(os.path.join(dataPath, "statistics.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    queue.put(None)
    listenProcess.join()
