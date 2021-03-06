#!/usr/bin/python
import matplotlib.pyplot as plt
import math
import copy
import cv2
import numpy as np
import rospy
import pickle
import os
import sys

datasetPath = "../data/results/temporal-s"
tfPath = "../data/static_tfs"
gtPath = "../data/gt"
visualisationPath = "../visualisation/eval-temporal-s"

if datasetPath[-1] == "/":
    datasetPath = datasetPath[:-1]
resultsPath = os.path.join("./results/", datasetPath.split("/")[-1])

def combineScans(scans):

    newScans = copy.deepcopy(scans[list(scans.keys())[0]])
    for key in list(scans.keys())[1:]:
        newScans = np.concatenate([newScans,scans[key]])
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
    tp_range = {}
    fn_range = {}
    fp_range = {}
    gt_det_total = 0
    gt_det_hit = {}
    det_gt_total = 0
    det_gt_hit = {}
    rot_total = 0
    rot_error = {}

    detectionGraphLimit = 5.0
    confusionGraphLimit = 20.0

    graph_resolution = 250
    for val in np.linspace(0.0, detectionGraphLimit, num=graph_resolution):
        gt_det_hit[val] = 0
        det_gt_hit[val] = 0
    for val in np.linspace(0.0, (math.pi/2) + 0.01, num=graph_resolution):
        rot_error[val] = 0
    for val in np.linspace(0.0, confusionGraphLimit, num=graph_resolution):
        tp_range[val] = 0
        fn_range[val] = 0
        fp_range[val] = 0

    fileCounter = 1
    ctr = 0
    for frame in gtdata[1]: #back middle sensor, higher framerate
        #ctr += 1
        #if ctr < 100:
        #    continue
        #if ctr > 1:
        #    continue

        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        if gttime not in data["ts"]:
            print("Warning, no data for gt!")

        dataFrameIdx = data["ts"].index(gttime)
        dataFrame = data["scans"][dataFrameIdx]
        dataFrame = combineScans(dataFrame)

        frameAnnotations = []
        frameAnnotations.append(frame[0])
        for i in range(1, len(frame)):
            rotation = frame[i][2] % math.pi
            position = [*frame[i][:2], 0, 1]
            mat = tfdata["sick_back_middle"]["mat"]
            position = np.matmul(mat, position)
            rotation = np.array([math.cos(rotation), math.sin(rotation), 0, 1])
            rotation = np.dot(mat, rotation)
            rotation = math.atan2(rotation[1], rotation[0])
            rotation = rotation % math.pi
            car = [*position[:2], rotation]
            frameAnnotations.append(car)

        ##todo confu matrix probably unused, precision recall graphs. range based data too? orientation? bi-directional  probability graphcs?

        detectionThreshold = 0.5

        #confusion matrix
        detections = copy.deepcopy(data["extrapolated"][dataFrameIdx])
        gts = copy.deepcopy(frameAnnotations)[1:]

        for rng in tp_range.keys():
            
            for gt in gts:
                dist = (gt[0]**2 + gt[1]**2)**0.5
                if dist > rng:
                    continue
                found = False
                for j in detections:
                    dx = gt[0] - j[0]
                    dy = gt[1] - j[1]
                    diff = ((dx**2) + (dy**2))**0.5
                    if diff < detectionThreshold:
                        found = True
                if found == True:
                    tp_range[rng] += 1
                else:
                    fn_range[rng] += 1

            for j in detections:
                found = False
                for gt in gts:
                    dx = gt[0] - j[0]
                    dy = gt[1] - j[1]
                    diff = ((dx**2) + (dy**2))**0.5
                    if diff < detectionThreshold:
                        found = True
                if found == False:
                    dist = (j[0]**2 + j[1]**2)**0.5
                    if dist > rng:
                        continue
                    fp_range[rng] += 1

        #orientation graphs
        for j in detections:
            for gt in gts:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < detectionThreshold :
                    jo = j[2] % (math.pi*2)
                    gto = gt[2] % (math.pi*2)
                    diff = abs(jo-gto)
                    if diff > (math.pi/2):
                        if jo > gto:
                            #jo -= (math.pi/2)
                            jo -= (math.pi)
                        else:
                            #jo += (math.pi/2)
                            gto -= (math.pi)
                        #print(gto, jo, diff, abs(jo-gto))
                        diff = abs(jo-gto)
                    else:
                        pass
                        #print(gto, jo, diff)
                    for key, _ in rot_error.items():
                        if diff < float(key):
                            rot_error[key] += 1
                    rot_total += 1
                    break

        #bi directional prob graphs

        #given gt, prob of detection
        for gt in gts:
            closestCar = 9999
            gt_det_total += 1
            for j in detections:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < closestCar:
                    closestCar = diff
            for key, _ in gt_det_hit.items():
                if float(key) > closestCar:
                    gt_det_hit[key] += 1

        #given detection, prob of gt
        for j in detections:
            closestGT = 9999
            det_gt_total += 1
            for gt in gts:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < closestGT:
                    closestGT = diff
            for key, _ in det_gt_hit.items():
                if float(key) > closestGT:
                    det_gt_hit[key] += 1
        
        cols = [[255, 128, 128]] * len(frameAnnotations)
        pointsToImgsDraw(filename, fileCounter, dataFrame, frameAnnotations[1:], cols)
        fileCounter += 1

    #recall distance
    recall_range = {}
    precision_range = {}
    lastVal = None
    for val in np.linspace(0.0, confusionGraphLimit, num=graph_resolution):
        if tp_range[val] == 0 and fn_range[val] == 0:
            recall_range[val] = None
        else:
            recall_range[val] = tp_range[val] / (tp_range[val]+fn_range[val])
        if tp_range[val] == 0 and fp_range[val] == 0:
            precision_range[val] = None
        else:
            precision_range[val] = tp_range[val] / (tp_range[val]+fp_range[val])
        lastVal = val
    precision = tp_range[lastVal] / (tp_range[lastVal] + fp_range[lastVal])
    recall = tp_range[lastVal] / (tp_range[lastVal] + fn_range[lastVal])
    tp = tp_range[lastVal]
    fn = fn_range[lastVal]
    fp = fp_range[lastVal]

    blanksRecall = [x for x in recall_range.keys() if recall_range[x] == None]
    blanksPrecision = [x for x in precision_range.keys() if precision_range[x] == None]
    recallIDs = [x for x in recall_range.keys() if x not in blanksRecall]
    precisionIDs = [x for x in precision_range.keys() if x not in blanksPrecision]
    filteredRecall = {}
    for key in recallIDs:
        filteredRecall[key] = recall_range[key]
    filteredPrecision = {}
    for key in precisionIDs:
        filteredPrecision[key] = precision_range[key]

    graphPath = os.path.join(resultsPath, "graph")
    os.makedirs(graphPath, exist_ok=True)

    makeGraph(gt_det_hit, gt_det_total, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, "gtdet-" + filename + ".png"))
    makeGraph(det_gt_hit, det_gt_total, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, "detgt-" + filename + ".png"))
    makeGraph(rot_error, rot_total, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, "rot-" + filename + ".png"))
    makeGraph(filteredRecall, 1.0, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "recall-" + filename + ".png"))
    makeGraph(filteredPrecision, 1.0, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "precision-" + filename + ".png"))

    print("Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
    print("Precision/Recall = %f %f" % (precision, recall))
    with open(os.path.join(resultsPath, "conf.pickle"), "wb") as f:
        pickle.dump({"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}, f, protocol=2)
    with open(os.path.join(resultsPath, "conf.txt"), "w") as f:
        f.write("tp %i fp %i fn %i precision %f recall %f" % (tp, fp, fn, precision, recall))

def makeGraph(hist, total, label, xlabel, ylabel, filename):

    if total < 1:
        print("Not enough vals for graph!")
        return

    x = []
    y = []
    for key, value in hist.items():
        x.append(key)
        y.append(value/total)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title='')
    ax.grid(linestyle="--")
    ax.legend(loc='lower right')
    plt.ylim([-0.02, 1.02])
    fig.savefig(filename)
    plt.close('all')

def pointsToImgsDraw(filename, fileCounter, points, wheels, colours):

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
    fn = os.path.join(visualisationPath)
    os.makedirs(fn, exist_ok=True)
    fn = os.path.join(fn, filename + "-" + str(fileCounter) + ".png")
    cv2.imwrite(fn, accum)

if __name__ == "__main__":

    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-7:] == ".pickle":
                print("Evaluating %s" % (filename))
                evaluateFile(filename)

