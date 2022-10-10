#!/usr/bin/python
import shutil
import copy
import utils
import pickle
import math
import sys
import os
import time
import concurrent
import concurrent.futures
import multiprocessing
import tqdm
from os import devnull
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

annoField = "annotations"

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Temporal():
    def __init__(self, path, folder, filename, queue, detectionDistance, interpolateFrames, interpolateRequired, extrapolateFrames, extrapolateRequired, datasetPath, outputPath):

        self.path = path
        self.folder = folder
        self.filename = filename[:-12]
        self.queue = queue
        self.datasetPath = datasetPath
        self.op = outputPath

        self.detectionDistance = detectionDistance
        self.interpolateFrames = interpolateFrames
        self.interpolateRequired = interpolateRequired
        self.extrapolateFrames = extrapolateFrames
        self.extrapolateRequired = extrapolateRequired
        self.underRobotDistance = 1.8
        self.data = {}
        self.outputPath = outputPath
    
        os.makedirs(self.outputPath, exist_ok=True)
        shutil.copy(os.path.join(self.datasetPath, "statistics.pkl"), self.outputPath)

    def run(self):

        self.queue.put("Process spawned for file %s" % (os.path.join(self.path, self.folder, self.filename)))

        foldername = os.path.join(self.op, self.folder)
        if os.path.isfile(os.path.join(foldername, self.filename + ".data.pickle")):
            if os.path.getsize(os.path.join(foldername, self.filename + ".data.pickle")) > 0:
                #print("Skipping, exists...")
                #return
                pass

        basefn = os.path.join(self.path, self.folder, self.filename)
        os.makedirs(os.path.join(self.outputPath, self.folder), exist_ok=True)
        shutil.copy(basefn + ".scans.pickle", os.path.join(self.outputPath, self.folder))
        try:
            shutil.copy(basefn + ".3d.pickle", os.path.join(self.outputPath, self.folder))
        except:
            pass

        with open(basefn + ".data.pickle", "rb") as f:
            self.data.update(pickle.load(f))

        self.temporal()
        
        fn = os.path.join(self.outputPath, self.folder, self.filename + ".data.pickle")
        with open(fn, "wb") as f:
            pickle.dump(self.data, f)

    def temporal(self):

        trans = self.data["trans"]
        ts = self.data["ts"]
        annotations = self.data[annoField]
        annotationsOdom = []

        robotPoses = []

        #move everything into the odom frame
        for idx in range(len(annotations)):
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
        for i in range(len(annotations)):
            lerped.append([])
            lerpedOnly.append([])

        distThresh = 0.5
        minClusterSize = 5
        self.windowLength = 20
        filteredDets = []
        filteredOnly = []
        for mainIdx in range(len(annotations)):
            filteredDets.append([])
            filteredOnly.append([])
            clusters = []
            for windowIdx in range(mainIdx-self.windowLength, mainIdx):
                if windowIdx < 0:
                    continue
                for car in annotationsOdom[windowIdx]:
                    found = False
                    for i, v in enumerate(clusters):
                        avgX = 0
                        avgY = 0
                        clusterSize = 0
                        for c in v:
                            avgX += c[0]
                            avgY += c[1]
                            clusterSize += 1
                        avgX = avgX/clusterSize
                        avgY = avgY/clusterSize
                        diffX = car[0] - avgX
                        diffY = car[1] - avgY
                        dist = ((diffX**2) + (diffY**2))**0.5
                        if dist < distThresh:
                            clusters[i].append(car)
                            found = True
                            break
                    if found == False:
                        clusters.append([car])
                        
            for i, v in enumerate(clusters):
                if len(v) < minClusterSize:
                    continue
                avgX = 0
                avgY = 0
                clusterSize = 0
                for c in v:
                    avgX += c[0]
                    avgY += c[1]
                    clusterSize += 1
                avgX = avgX/clusterSize
                avgY = avgY/clusterSize
                found = False
                for car in annotationsOdom[mainIdx]:
                    diffX = car[0] - avgX
                    diffY = car[1] - avgY
                    dist = ((diffX**2) + (diffY**2))**0.5
                    if dist < distThresh:
                        found = True
                        break
                if found == False:
                    filteredDets[mainIdx].append([avgX, avgY, 0])
                    filteredOnly[mainIdx].append([avgX, avgY, 0])

        #add real detections to extrapolated
        for frameIdx in range(len(annotationsOdom)):
            for detectionIdx in range(len(annotationsOdom[frameIdx])):
                filteredDets[frameIdx].append(annotationsOdom[frameIdx][detectionIdx])

        self.data["annotationsOdom"]  = annotationsOdom
        self.data["filtered"] = []
        self.data["filteredOnly"] = []

        #move back to robot frame
        for idx in range(len(annotations)):
            mat = np.array(trans[idx])
            mat = np.linalg.inv(mat)
            r = R.from_matrix(mat[:3, :3])
            yaw = r.as_euler('zxy', degrees=False)
            yaw = yaw[0]
            filtered = []
            filteredDOnly = []
 
            for det in range(len(filteredOnly[idx])):
                print(filteredDets[idx][det])
                point = np.array([*filteredOnly[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = 0
                filteredDOnly.append([*point, orientation])

            for det in range(len(filteredDets[idx])):
                print(filteredDets[idx][det])
                point = np.array([*filteredDets[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = 0
                filtered.append([*point, orientation])
                
            self.data["filtered"].append(filtered)
            self.data["filteredOnly"].append(filteredDOnly)
        with suppress_stdout_stderr():
            self.data["filtered"] = np.array(self.data["filtered"])
            self.data["filteredOnly"] = np.array(self.data["filteredOnly"])

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

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, None):
        if type(item) == int:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    datasetPath = "../data/detector/"

    count = 0
    with open(os.path.join(datasetPath, "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]
                


    multi = 1
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count*multi))
    listenProcess.start()

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" in filename:
                if "drive" not in filename:
                    continue
                path = datasetPath
                folder = files[0][len(path):]
                outputPath = "../data/temporal/real"
                a = Temporal(path, folder, filename, queue, 0, 0, 0, 0, 0, datasetPath, outputPath)
                a.run()
                jobs.append(Temporal(path, folder, filename, queue, 0, 0, 0, 0, 0, datasetPath, outputPath))
                #distance thresh, interp window, interp dets req, extrap window, extrap dets req

    workers = 12
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)

        for future in futures:
            try:
                pass
                queue.put(str(future.result()))
            except Exception as e:
                queue.put("P Exception: " + str(e))

    queue.put(None)
    listenProcess.join()
