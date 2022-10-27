#!/usr/bin/python
import math
import rospy
import pickle
import math
import cv2
import sys
import os
import rosbag
import multiprocessing
import time
from numpy.linalg import eig
from os import devnull
from sklearn.cluster import DBSCAN
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

combinePath = None
modelName = None
inferredPath = None
combinedOutPath = None

@contextmanager
def suppress_stdout_stderr():
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
def run():

    #now we merge those detections into the original file structures
    #basically bring it into a common format
    combinableFilenames = []
    for files in os.walk(os.path.join(inferredPath, modelName)):
        for filename in files[2]:
            #if filename[-12:] == ".annotations":
            combinableFilenames.append(os.path.join(files[0], filename.split(".")[0]))
    combinableFilenames = list(set(combinableFilenames))

    print("Combining %i files" % (len(combinableFilenames)))

    for base in combinableFilenames:

        print("Combining bag", base)
        modelPath = base.split("/")[-2]

        #open combinable file
        combineFile = ""
        combineFolder = ""
        for files in os.walk(combinePath):
            for filename in files[2]:
                if ".data.pickle" not in filename:
                    continue
                if filename.split(".")[0] == base.split("/")[-1]:
                    subPath = files[0][len(combinePath)+1:]
                    combineFolder = subPath
                    combineFile = os.path.join(subPath, filename)

        if combineFile == "":
            print("Error combining ", filename)
            break

        data = []
        with open(os.path.join(combinePath, combineFile), "rb") as f:
            data = pickle.load(f)

        readyFiles = []
        for files in os.walk(os.path.join(inferredPath, modelName)):
            for filename in files[2]:
                if filename.split(".")[0] == base.split("/")[-1]:
                    readyFiles.append(filename)

        print("N det frames, total f", len(readyFiles), len(data["ts"]))
        #if len(readyFiles) != len(data["ts"]): #dont worry, multi mask?
        #    print("Warning, frame mismatch", base)

        data["maskrcnn"] = []
        for i in range(len(data["ts"])):
            data["maskrcnn"].append([])

        #open each file, add it to the correct frame
        for filename in readyFiles:
            idx = int(filename.split(".pickle-")[1].split(".")[0])
            with open(os.path.join(inferredPath, modelPath, filename), "rb") as f:
                annotations = pickle.load(f)
                nmsanno = []
                for i in range(len(annotations)):
                    a = annotations[i]
                    similar = False
                    for j in range(i+1, len(annotations)):
                        b = annotations[j] 
                        diffX = a[0] - b[0]
                        diffY = a[1] - b[1]
                        dist = ((diffX**2) + (diffY**2))**0.5
                        if dist < 0.5:
                            similar = True
                            break
                    if similar == False:
                        nmsanno.append(a)
                data["maskrcnn"][idx] = nmsanno
        
        #for i in range(len(readyFiles)):
        #    if data["maskrcnn"][i] == None:
        #        print("Warning, empty frame found")


        os.makedirs(os.path.join(combinedOutPath, modelPath, combineFolder), exist_ok=True)
        print("Saving to %s" % (os.path.join(combinedOutPath, modelPath, combineFolder, base.split("/")[-1] + ".bag.data.pickle")))
        with open(os.path.join(combinedOutPath, modelPath, combineFolder, base.split("/")[-1] + ".bag.data.pickle"), "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":

    ctr = 0
    for files in os.walk("../data/maskrcnn/inference-all"):
        for modelName in files[1]:

            ctr += 1
            print(ctr)

            combinePath = "../data/temporal/temporal-0.6-20-10-20-100.6-20-10-20-10"
            inferredPath = "../data/maskrcnn/inference-all"
            combinedOutPath = "../data/maskrcnn/rectified-all"

            print(combinePath)
            print(inferredPath)
            print(combinedOutPath)
            print(modelName)

            run()

