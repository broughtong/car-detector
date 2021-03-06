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

datasetPath = "../data/rosbags/"
outputPath = "../data/static_tfs/"
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

        self.scanTopics = ["sick_back_left", 
                "sick_back_right", 
                "sick_back_middle",
                "sick_front"]

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
        print("Finished opening bag, reading...")

        tfs = {}
        for i in self.scanTopics:
            tfs[i] = {"translation": None, "quaternion": None, "mat": None}

            try:
                with suppress_stdout_stderr():
                    tftime = self.bagtf.waitForTransform("base_link", i, None)
                    translation, quaternion = self.bagtf.lookupTransform("base_link", i, tftime)
            except:
                print("Error finding tf here", flush = True)
                return

            r = R.from_quat(quaternion)
            mat = r.as_matrix()
            mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
            mat[0][-1] += translation[0]
            mat[1][-1] += translation[1]
            mat[2][-1] += translation[2]
            mat[3][-1] = 1

            tfs[i]["translation"] = translation
            tfs[i]["quaternion"] = quaternion
            tfs[i]["mat"] = mat

        os.makedirs(outputPath, exist_ok=True)
        fn = os.path.join(outputPath, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump(tfs, f, protocol=2)

        print("Process finished for file %s" % (self.filename), flush = True)

    def saveScans(self):

        if len(self.scans) == 0:
            return

        os.makedirs(outputPath, exist_ok=True)
        fn = os.path.join(outputPath, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"scans": self.scans, "trans": self.trans, "ts": self.ts}, f, protocol=2)

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



