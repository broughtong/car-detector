#!/usr/bin/python
import rospy
import pickle
import math
import cv2
import sys
import os
import rosbag
import time
import concurrent
import multiprocessing
import tqdm
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

class Extractor():
    def __init__(self, path, folder, filename, queue):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue

        self.scanTopics = ["sick_back_left", 
                "sick_back_right", 
                #"sick_back_low",
                "sick_back_middle",
                "sick_front"]
                #"os_sensor"]

    def run(self):
        
        self.queue.put("%s: Process spawned" % (self.filename))

        self.queue.put("%s: Reading tfs" % (self.filename))
        try:
            with suppress_stdout_stderr():
                fn = os.path.join(self.path, self.folder, self.filename)
                self.bagtf = BagTfTransformer(fn)
        except:
            self.queue.put("%s: Bag Failed (1)" % (self.filename))
            self.queue.put(1)
            return

        tfs = {}
        for i in self.scanTopics:
            tfs[i] = {"translation": None, "quaternion": None, "mat": None}

            try:
                with suppress_stdout_stderr():
                    tftime = self.bagtf.waitForTransform("base_link", i, None)
                    translation, quaternion = self.bagtf.lookupTransform("base_link", i, tftime)
            except:
                self.queue.put("%s: Error finding tf (2) from %s to %s" % (self.filename, i, "base_link"))
                self.queue.put(1)
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

        resultFolder = os.path.join(outputPath, self.folder)
        os.makedirs(resultFolder, exist_ok=True)
        fn = os.path.join(outputPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump(tfs, f)

        self.queue.put("%s: Process finished sucessfully" % (self.filename))
        self.queue.put(1)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, None):
        if type(item) == int:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":
    
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-4:] == ".bag":
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Extractor(path, folder, filename, queue))
    
    listenProcess = multiprocessing.Process(target=listener, args=(queue, len(jobs)))
    listenProcess.start()
    
    workers = 8
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)
            time.sleep(1)

        for future in futures:
            try:
                res = future.result()
                if type(res) is int:
                    res = str(res)
                if res is not None:
                    queue.put(res)
            except Exception as e:
                queue.put("P Exception: " + str(e))

    queue.put(None)
    listenProcess.join()
