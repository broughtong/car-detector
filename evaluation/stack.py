#!/usr/bin/python
import matplotlib.pyplot as plt
import utils
import math
import copy
import cv2
import numpy as np
import rospy
import pickle
import os
import sys

def evaluateFile(path, folder, filename, datasetPath, tfPath, gtPath, annoField, resultsPath):

    fn = os.path.join(datasetPath, folder, filename)
    if not os.path.isfile(fn):
        print("Unable to open file: %s" % (fn))
        return
    
    tffn = "".join(filename.split(".data"))
    tffn = os.path.join(tfPath, folder, tffn)
    if not os.path.isfile(tffn):
        print("Unable to open TF file: %s" % (tffn))
        print(path, folder, filename)
        return

    gtfn = filename.split(".")[0] + "-lidar.pkl"
    for files in os.walk(gtPath):
        for filen in files[2]:
            if filen == gtfn:
                gtfn = os.path.join(files[0], gtfn)

    if not os.path.isfile(gtfn):
        #print("Unable to open GT file: %s %s" % (folder, gtfn))
        return

    data = []
    gtdata = []
    tfdata = []

    with open(fn, "rb") as f:
        data = pickle.load(f)
    scanfn = fn[:-12] + ".scans.pickle"
    with open(scanfn, "rb") as f:
        data.update(pickle.load(f))
    with open(tffn, "rb") as f:
        tfdata = pickle.load(f)
    with open(gtfn, "rb") as f:
        print(gtfn)
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
    for frame in gtdata[0]: #back middle sensor, higher framerate
        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        """
        if gttime not in data["ts"]:
            print("Warning, no data for gt!", gttime)
            continue
        """
        bestTime = 0
        bestIdx = None
        bestDiff = rospy.Duration(999999)
        for i, v in enumerate(data["ts"]):
            diff = abs(gttime - v)
            if diff < bestDiff:
                bestTime = v
                bestIdx = i
                bestDiff = diff
        if bestDiff.secs > 0:
            continue
        if bestDiff.nsecs > 200000000:
            continue
        if bestIdx == None:
            continue
        #print(bestDiff.secs, bestDiff.nsecs)

        #dataFrameIdx = data["ts"][bestIdx]
        dataFrameIdx = bestIdx
        dataFrame = data["scans"][dataFrameIdx]
        dataFrame = utils.combineScans(dataFrame)

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

        detectionThreshold = 0.75

        #confusion matrix
        detections = copy.deepcopy(data[annoField][dataFrameIdx])
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
                    #lets check if there are a certain number of points at least
                    carPoints, nonCarPoints = utils.getInAnnotation(dataFrame, [gt])
                    if len(nonCarPoints) > 15:
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
        #pointsToImgsDraw(filename, fileCounter, dataFrame, frameAnnotations[1:], cols)
        #fileCounter += 1

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
    
    precision = None
    recall = None
    try:
        precision = tp_range[lastVal] / (tp_range[lastVal] + fp_range[lastVal])
        recall = tp_range[lastVal] / (tp_range[lastVal] + fn_range[lastVal])
    except:
        pass
    tp = tp_range[lastVal]
    fn = fn_range[lastVal]
    fp = fp_range[lastVal]

    try:
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
    except:
        pass

    graphPath = os.path.join(resultsPath, "graph")
    os.makedirs(graphPath, exist_ok=True)

    makeGraph(gt_det_hit, gt_det_total, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, "gtdet-" + filename + ".png"))
    makeGraph(det_gt_hit, det_gt_total, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, "detgt-" + filename + ".png"))
    makeGraph(rot_error, rot_total, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, "rot-" + filename + ".png"))
    makeGraph(filteredRecall, 1.0, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "recall-" + filename + ".png"))
    makeGraph(filteredPrecision, 1.0, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "precision-" + filename + ".png"))

    if precision == None:
        precision = -1
    if recall == None:
        recall = -1

    print("Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
    print("Precision/Recall = %f %f" % (precision, recall))
    with open(os.path.join(resultsPath, "conf.pickle"), "wb") as f:
        pickle.dump({"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}, f, protocol=2)
    with open(os.path.join(resultsPath, "conf.txt"), "w") as f:
        f.write("tp %i fp %i fn %i precision %f recall %f" % (tp, fp, fn, precision, recall))

    results = [tp, fp, fn]
    results.append(tp_range)
    results.append(fn_range)
    results.append(fp_range)
    results.append(gt_det_total)
    results.append(gt_det_hit)
    results.append(det_gt_total)
    results.append(det_gt_hit)
    results.append(rot_total)
    results.append(rot_error)
    results.append(filteredRecall)
    results.append(filteredPrecision)
    results.append([path, folder, filename])
    return results

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

def makeGraphs(hist, total, label, xlabel, ylabel, filename):

    for i in total:
        if i < 1:
            print("Not evenough vals for graph!")
            return

    fig, ax = plt.subplots()
    for i, v in enumerate(hist):
        x = []
        y = []
        for key, value in v.items():
            x.append(key)
            y.append(value/total[i])
    
        ax.plot(x, y, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title='')
    ax.grid(linestyle="--")
    #ax.legend(loc='lower right')
    plt.ylim([-0.02, 1.02])
    fig.savefig(filename)
    plt.close('all')

def generateGraphs(results, evalName):

    resultsPath = os.path.join("./results/", evalName)
    os.makedirs(resultsPath, exist_ok=True)

    for i in range(len(results)):
        for j in range(i+1, len(results)):
            a = results[i]
            b = results[j]
            if a[-1][2] == b[-1][2]:
                graphPath = os.path.join(resultsPath, "graph")
                os.makedirs(graphPath, exist_ok=True)

                gt_det_hit = [a[7], b[7]]
                gt_det_total = [a[6], b[6]]
                det_gt_hit = [a[9], b[9]]
                det_gt_total = [a[8], b[8]]
                rot_error = [a[11], b[11]]
                rot_total = [a[10], b[10]]
                filteredRecall = [a[12], b[12]] 
                filteredPrecision = [a[13], b[13]]

                makeGraphs(gt_det_hit, gt_det_total, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, "gtdet.png"))
                makeGraphs(det_gt_hit, det_gt_total, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, "detgt.png"))
                makeGraphs(rot_error, rot_total, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, "rot.png"))
                makeGraphs(filteredRecall, [1.0, 1.0], "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "recall.png"))
                makeGraphs(filteredPrecision, [1.0, 1.0], "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "precision.png"))

def run(datasetPath, tfPath, gtPath, annoField):

    if datasetPath[-1] == "/":
        datasetPath = datasetPath[:-1]
    resultsPath = os.path.join("./results/", datasetPath.split("/")[-1] + "-" + annoField)
    os.makedirs(resultsPath, exist_ok=True)

    tp, fp, fn = 0, 0, 0
    results = []

    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".data.pickle" in filename:
                if "drive" not in filename:
                    continue
                path = datasetPath
                folder = files[0][len(path)+1:]
                vals = evaluateFile(path, folder, filename, datasetPath, tfPath, gtPath, annoField, resultsPath)
                if vals is not None:
                    tp += vals[0]
                    fp += vals[1]
                    fn += vals[2]
                    results.append(vals)
    precision = float('nan')
    recall = float('nan')
    f1 = float('nan')
    samples = tp + fp + fn
    try:
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = (2*tp)/((2*tp) + fp + fn)
    except:
        pass

    resultNameString = datasetPath.split("/")[-1]
    with open(os.path.join(resultsPath, "cumulative.txt"), "w") as f:
        f.write("f1 %f tp %i fp %i fn %i p %f r %f samples %i %s\n" % (f1, tp, fp, fn, precision, recall, samples, resultNameString))
    return results

if __name__ == "__main__":

    results = []

    tfPath = "../data/static_tfs"
    gtPath = "../data/gt"

    datasetPath = "../data/detector"
    annoField = "annotations"
    results += run(datasetPath, tfPath, gtPath, annoField)
    datasetPath = "../data/temporal-0.6-100-48-25-480.6-100-48-25-48"
    datasetPath = "../data/temporal-0.6-25-3-25-30.6-25-3-25-3"
    annoField = "extrapolated"
    datasetPath = "../data/detector"
    annoField = "annotations"
    results += run(datasetPath, tfPath, gtPath, annoField)
    

    """
    ctr = 0
    for files in os.walk("../data/temporal"):
        for filename in files[2]:
            if "driveby.bag.data.pickle" in filename:
                datasetPath = "/".join(files[0].split("/")[:-1])
                run(datasetPath, tfPath, gtPath, annoField)
                ctr += 1
                print(ctr)
    """

    generateGraphs(results, "eval-temp")

