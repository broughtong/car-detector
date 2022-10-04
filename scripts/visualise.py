#!/usr/bin/python
import cv2
import shutil
import multiprocessing
import pickle
import os
import math
import numpy as np
import copy
import multiprocessing
import utils
import tqdm
import concurrent
import concurrent.futures

datasetPath = "../data/detector0.08"
scanField = "scans"
annotationField = "annotations"
outPath = "../visualisation/detector0.08"

class Visualise:
    def __init__(self, path, folder, filename, queue, outPath, scanField, annotationField):

        self.path = path
        self.folder = folder
        self.filename = filename[:-12]
        self.queue = queue
        self.outPath = outPath
        self.scanField = scanField
        self.annotationField = annotationField

        self.fileCounter = 0
        self.data = {}

    def run(self):
        
        self.queue.put("Process spawned for file %s" % (self.filename))

        basefn = os.path.join(self.path, self.folder, self.filename)
        with open(basefn + ".data.pickle", "rb") as f:
            self.data.update(pickle.load(f))
        with open(basefn + ".scans.pickle", "rb") as f:
            self.data.update(pickle.load(f))

        for frameIdx in range(len(self.data[self.scanField])):
            self.drawFrame(frameIdx)
            self.queue.put(1)

    def drawFrame(self, idx):

        
        scan = self.data[self.scanField][idx]
        #scans = combineScans([scans["sick_back_left"], scans["sick_back_right"], scans["sick_back_middle"], scans["sick_front"]])
        #scans = combineScans(self.data["scans"][idx])

        fn = os.path.join(self.outPath, self.folder, self.filename + "-" + str(idx) + ".png")
        os.makedirs(os.path.join(self.outPath, self.folder), exist_ok=True)
        
        scans = utils.combineScans({"sick_back_left": scan["sick_back_left"], "sick_back_right": scan["sick_back_right"]})
        utils.drawImgFromPoints(fn, scans, [], [], self.data[self.annotationField][idx], [], 1, False)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, None):
        if type(item) == int:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":
    
    count = 0
    with open(os.path.join(datasetPath, "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count))
    listenProcess.start()

    try:
        shutil.rmtree(outPath)
    except:
        pass
    os.makedirs(outPath, exist_ok=True)

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" not in filename:
                continue
            path = datasetPath
            folder = files[0][len(path)+1:]
            jobs.append(Visualise(path, folder, filename, queue, outPath, scanField, annotationField))

    workers = 2
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)

        for future in futures:
            try:
                queue.put(str(future.result()))
            except Exception as e:
                queue.put("P Exception: " + str(e))

    queue.put(None)
    listenProcess.join()
