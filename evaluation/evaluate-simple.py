#!/usr/bin/python
import rospy
import pickle
import os

datasetPath = "../data/results/simple"
gtPath = "../data/gt"

def evaluateFile(filename):

    fn = os.path.join(datasetPath, filename)
    if not os.path.isfile(fn):
        print("Unable to open file: %s" % (fn))

    gtfn = os.path.join(gtPath, filename)
    gtfn = gtfn[:-11] + "-lidar.pkl"
    if not os.path.isfile(gtfn):
        print("Unable to open GT file: %s" % (gtfn))

    data = []
    gtdata = []

    with open(fn, "rb") as f:
        data = pickle.load(f)
    with open(gtfn, "rb") as f:
        gtdata = pickle.load(f)

     

if __name__ == "__main__":

    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                evaluateFile(filename)

