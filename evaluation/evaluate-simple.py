#!/usr/bin/python
import math
import copy
import cv2
import numpy as np
import rospy
import pickle
import os

datasetPath = "../data/results/simple"
tfPath = "../data/static_tfs"
gtPath = "../data/gt"
visualisationPath = "../visualisation"

def combineScans(scans):

    newScans = scans[list(scans.keys())[0]]
    for key in list(scans.keys())[1:]:
        np.concatenate([newScans,scans[key]])

    return newScans

def evaluateFile(filename):

    fn = os.path.join(datasetPath, filename)
    if not os.path.isfile(fn):
        print("Unable to open file: %s" % (fn))
    
    tffn = os.path.join(tfPath, filename)
    if not os.path.isfile(tffn):
        print("Unable to open file: %s" % (tffn))

    gtfn = os.path.join(gtPath, filename)
    gtfn = gtfn[:-11] + "-lidar.pkl"
    if not os.path.isfile(gtfn):
        print("Unable to open GT file: %s" % (gtfn))

    data = []
    gtdata = []
    tfdata = []

    with open(fn, "rb") as f:
        data = pickle.load(f)
    with open(tffn, "rb") as f:
        tfdata = pickle.load(f)
    with open(gtfn, "rb") as f:
        gtdata = pickle.load(f)

    #analysis variables
    tp, fp, fn = 0, 0, 0
    gt_det_total = {}
    gt_det_hit = {}
    det_gt_total = {}
    det_gt_hit = {}
    for val in np.linspace(0.0, 5.0, num=100):
        gt_det_total[val] = 0
        gt_det_hit[val] = 0
        det_gt_total[val] = 0
        det_gt_hit[val] = 0

    fileCounter = 1
    for frame in gtdata[1]: #back middle sensor, higher framerate

        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        if gttime not in data["ts"]:
            print("Warning, no data for gt!")

        dataFrameIdx = data["ts"].index(gttime)
        dataFrame = data["scans"][dataFrameIdx]

        dataFrame = combineScans(dataFrame)

        frameAnnotations = []
        frameAnnotations.append(frame[0])
        for i in range(1, len(frame)):
            rotation = frame[i][2]
            a = list(frame[i])
            a[2] = 0
            a.append(1) 
            mat = tfdata["sick_back_middle"]["mat"]
            a = np.matmul(mat, a)
            rotationVec = np.array([0, 0, rotation, 1])#.transpose()
            #print(rotationVec)
            #rotation = np.matmul(mat, rotationVec)#np.array([[0], [0], [rotation]]))
            rotation = mat.dot(rotationVec)[2]
            #print(rotation)
            a = [*a[:2], rotation]
            frameAnnotations.append(a)

        ##todo confu matrix probably unused, precision recall graphs. range based data too? orientation? bi-directional  probability graphcs?

        detectionThreshold = 0.5

        #confusion matrix
        detections = copy.deepcopy(data["annotations"][dataFrameIdx])
        gts = copy.deepcopy(frameAnnotations)[1:]

        for gt in gts:
            found = False
            for j in detections:

                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < detectionThreshold:
                    found = True
                
            if found == True:
                tp += 1
            else:
                fn += 1

        for j in detections:
            found = False
            for gt in gts:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < detectionThreshold:
                    found = True
            if found == False:
                fp += 1

        #range based prob graphs

        #orientation graphs
        #get modulo ONE pi (to solve ambiuguity)
        #get abs diff between anotation and our value
        #if over 90, subtract 90 or whatever
        #graph of probbilit of difference in orientaiton?
        for j in detections:
            for gt in gts:
                jo = j[2] % math.pi
                gto = gt[2] % math.pi
                diff = abs(jo-gto)
                print(diff)
                if diff > (math.pi/2):
                    if jo > gto:
                        jo -= (math.pi/2)
                    else:
                        jo += (math.pi/2)
                    diff = abs(jo-gto)
                    print(diff, "corrected")
                break

        #bi directional prob graphs

        #given gt, prob of detection
        for gt in gts:
            for j in detections:


                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < detectionThreshold:
                    found = True
                
            if found == True:
                tp += 1
            else:
                fn += 1


        #given detection, prob of gt


        
        cols = [[255, 128, 128]] * len(frameAnnotations)
        pointsToImgsDraw(filename, fileCounter, dataFrame, "evaluation-simple", frameAnnotations[1:], cols)
        fileCounter += 1

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    print("Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
    print("Precision/Recall = %f %f" % (precision, recall))

def pointsToImgsDraw(filename, fileCounter, points, location, wheels, colours):

    res = 1024
    scale = 25
    accum = np.zeros((res, res, 3))
    accum.fill(255)

    for point in points:
        x, y = point[:2]
        x *= scale
        y *= scale
        x = int(x)
        y = int(y)
        try:
            accum[x+int(res/2), y+int(res/2)] = [0, 0, 0]
        except:
            pass

    for wheel in range(len(wheels)):
        x, y = wheels[wheel][:2]
        x *= scale
        y *= scale
        x = int(x)
        y = int(y)

        try:
            accum[x+int(res/2), y+int(res/2)] = colours[wheel]
        except:
            pass

        try:
            accum[x+int(res/2)+1, y+int(res/2)+1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2)+1, y+int(res/2)-1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2)-1, y+int(res/2)+1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2)-1, y+int(res/2)-1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2)-1, y+int(res/2)] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2), y+int(res/2)-1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2), y+int(res/2)+1] = colours[wheel]
        except:
            pass
        try:
            accum[x+int(res/2)+1, y+int(res/2)] = colours[wheel]
        except:
            pass
    fn = os.path.join(visualisationPath, location)
    os.makedirs(fn, exist_ok=True)
    fn = os.path.join(fn, filename + "-" + str(fileCounter) + ".png")
    cv2.imwrite(fn, accum)


if __name__ == "__main__":

    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                print("Evaluating %s" % (filename))
                evaluateFile(filename)

