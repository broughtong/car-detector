#!/usr/bin/python
import pickle
import math
import sys
import os
import time
import concurrent
import concurrent.futures
import multiprocessing
import tqdm
import numpy as np
import rosbag
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
from scipy.spatial.transform import Rotation as R

import vehicle_auto_annotation.src.utils.sentinel as sentinel

datasetPath = "../../data/rosbags/"
outputPath = "../../data/static_tfs/"

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
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
                "sick_back_middle",
                "sick_front",
                "os_sensor"]

    def run(self):
        
        self.queue.put("%s: Process spawned" % (self.filename))

        try:
            with suppress_stdout_stderr():
                fn = os.path.join(self.path, self.folder, self.filename)
                self.bagtf = BagTfTransformer(fn)
        except:
            self.queue.put("%s: Bag Failed [1]" % (self.filename))
            self.queue.put(sentinel.PROBLEM)
            return

        tfs = {}
        for i in self.scanTopics:

            try:
                with suppress_stdout_stderr():
                    tftime = self.bagtf.waitForTransform("base_link", i, None)
                    translation, quaternion = self.bagtf.lookupTransform("base_link", i, tftime)
            except:
                self.queue.put("%s: Warning, could not find tf from %s to %s [2]" % (self.filename, i, "base_link"))

            r = R.from_quat(quaternion)
            mat = r.as_matrix()
            mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
            mat[0][-1] += translation[0]
            mat[1][-1] += translation[1]
            mat[2][-1] += translation[2]
            mat[3][-1] = 1

            tfs[i] = {}
            tfs[i]["translation"] = translation
            tfs[i]["quaternion"] = quaternion
            tfs[i]["mat"] = mat

        resultFolder = os.path.join(outputPath, self.folder)
        os.makedirs(resultFolder, exist_ok=True)
        fn = os.path.join(outputPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump(tfs, f)

        self.queue.put("%s: Process finished sucessfully" % (self.filename))
        self.queue.put(sentinel.SUCCESS)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get):
        if item == sentinel.EXIT:
            break
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
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
            time.sleep(0.1)

        for future in futures:
            try:
                res = future.result()
                queue.put(res)
            except Exception as e:
                queue.put("Process exception: " + str(e))

    queue.put(sentinel.EXIT)
    listenProcess.join()
