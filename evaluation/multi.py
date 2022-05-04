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
    
detectionGraphLimit = 5.0
confusionGraphLimit = 20.0
graph_resolution = 250
detectionThreshold = 3

evalName = "full-complete"
datasetPaths = {"../data/results/detector-s": "annotations"}
datasetPaths["../data/results/temporal-new-0.8-50-6-75-6"] = "extrapolated"
datasetPaths["../data/results/maskrcnn_scans_rectified-debug/scans-25-04-22-18_08_16.pth"] = "maskrcnn"
#datasetPaths["../data/results/maskrcnn_scans_rectified-l"] = "maskrcnn"
#for i in range(1200, 2000, 100):
#    datasetPaths["../data/results/temporal-prc-%s-0.4-0" % (str(i))] = "extrapolated"
visualisationPath = "../visualisation/eval-" + evalName
tfPath = "../data/static_tfs"
gtPath = "../data/gt"

resultsPath = os.path.join("./results/", evalName)

tp_range = {}
fn_range = {}
fp_range = {}
gt_det_total = {}
gt_det_hit = {}
det_gt_total = {}
det_gt_hit = {}
rot_total = {}
rot_error = {}
recall_range = {}
precision_range = {}

lastVal = None
for key in datasetPaths.keys():
    if key[-1] == "/":
        print("Warning, please remove trailing slash")
        sys.exit(0)
    tp_range[key] = {}
    fn_range[key] = {}
    fp_range[key] = {}
    gt_det_total[key] = 0
    gt_det_hit[key] = {}
    det_gt_total[key] = 0
    det_gt_hit[key] = {}
    rot_total[key] = 0
    rot_error[key] = {}
    recall_range[key] = {}
    precision_range[key] = {}

def combineScans(scans):

    newScans = copy.deepcopy(scans[list(scans.keys())[0]])
    for key in list(scans.keys())[1:]:
        newScans = np.concatenate([newScans,scans[key]])
    return newScans

def findfile(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def evaluateFile(filename, method, filePart):
    global lastVal

    gtfn = filename.split(".")[0]
    gtfn = findfile(gtfn + "-lidar.pkl", gtPath) 
    if gtfn is None:
        print("no gt found %s %s" % (gtfn, filename))
        return

    print("Evaluating %s from %s (%s)" % (filename, method, filePart))

    fn = findfile(filename, method)
    if not os.path.isfile(fn):
        print("Unable to open data file: %s" % (fn))
        return
    
    tffn = os.path.join(tfPath, filename)
    if not os.path.isfile(tffn):
        print("Unable to open tf file: %s" % (tffn))
        return

    data = []
    gtdata = []
    tfdata = []

    with open(fn, "rb") as f:
        data = pickle.load(f)
    with open(tffn, "rb") as f:
        tfdata = pickle.load(f)
    with open(gtfn, "rb") as f:
        gtdata = pickle.load(f, encoding='latin1')

    for val in np.linspace(0.0, detectionGraphLimit, num=graph_resolution):
        gt_det_hit[method][val] = 0
        det_gt_hit[method][val] = 0
    for val in np.linspace(0.0, (math.pi/2) + 0.01, num=graph_resolution):
        rot_error[method][val] = 0
    for val in np.linspace(0.0, confusionGraphLimit, num=graph_resolution):
        tp_range[method][val] = 0
        fn_range[method][val] = 0
        fp_range[method][val] = 0

    frameCounter = 0
    for frame in gtdata[1]: #back middle sensor, higher framerate
        frameCounter += 1

        if frameCounter != 120:
            continue
        print("ninin")

        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        if gttime not in data["ts"]:
            print("Warning, no data for gt!")
            print(fn, gtfn)
            print(frame, frameCounter)
            print(frame[0].secs, frame[0].nsecs)
            print(gttime)
            continue

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

        #confusion matrix
        detections = copy.deepcopy(data[filePart][dataFrameIdx])
        gts = copy.deepcopy(frameAnnotations)[1:]

        for rng in tp_range[method].keys():
            pass
        for rng in [20]:
            
            for gt in gts:
                dist = (gt[0]**2 + gt[1]**2)**0.5
                if dist > rng:
                    continue
                found = False
                for j in detections:
                    dx = gt[0] - j[0]
                    dy = gt[1] - j[1]
                    diff = ((dx**2) + (dy**2))**0.5
                    if "scans" in method:
                        print("===")
                        #print(filename)
                        print(frameCounter - 1)
                        #print(method)
                        print(gt)
                        print(j)
                        print(diff)
                    if diff < detectionThreshold:
                        found = True
                if found == True:
                    tp_range[method][rng] += 1
                else:
                    fn_range[method][rng] += 1

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
                    fp_range[method][rng] += 1

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
                        pass #code path only used for comments
                        #print(gto, jo, diff)
                    for key, _ in rot_error[method].items():
                        if diff < float(key):
                            rot_error[method][key] += 1
                    rot_total[method] += 1
                    break

        #bi directional prob graphs

        #given gt, prob of detection
        for gt in gts:
            closestCar = 9999
            gt_det_total[method] += 1
            for j in detections:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < closestCar:
                    closestCar = diff
            for key, _ in gt_det_hit[method].items():
                if float(key) > closestCar:
                    gt_det_hit[method][key] += 1

        #given detection, prob of gt
        for j in detections:
            closestGT = 9999
            det_gt_total[method] += 1
            for gt in gts:
                dx = gt[0] - j[0]
                dy = gt[1] - j[1]
                diff = ((dx**2) + (dy**2))**0.5
                if diff < closestGT:
                    closestGT = diff
            for key, _ in det_gt_hit[method].items():
                if float(key) > closestGT:
                    det_gt_hit[method][key] += 1
        
        cols = [[255, 128, 128]] * len(frameAnnotations)
        pointsToImgsDraw(filename, frameCounter, dataFrame, frameAnnotations[1:], cols)

    #recall distance
    for val in np.linspace(0.0, confusionGraphLimit, num=graph_resolution):
        if tp_range[method][val] == 0 and fn_range[method][val] == 0:
            recall_range[method][val] = None
        else:
            recall_range[method][val] = tp_range[method][val] / (tp_range[method][val]+fn_range[method][val])
        if tp_range[method][val] == 0 and fp_range[method][val] == 0:
            precision_range[method][val] = None
        else:
            precision_range[method][val] = tp_range[method][val] / (tp_range[method][val]+fp_range[method][val])
        lastVal = val


def drawGraphs():

    graphPath = os.path.join(resultsPath, "graph")
    os.makedirs(graphPath, exist_ok=True)

    confusionPickle = {}
    confusionText = ""
    filteredRecall = {}
    filteredPrecision = {}

    for method in datasetPaths:
        print(method, lastVal, flush=True)
        tp = tp_range[method][lastVal]
        fn = fn_range[method][lastVal]
        fp = fp_range[method][lastVal]
        precision = "N/A"
        recall = "N/A"
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        except:
            print("Cannot generate pr/re %s %i %i %i" % (method, tp, fn, fp))

        try:
            blanksRecall = [x for x in recall_range[method].keys() if recall_range[method][x] == None]
            blanksPrecision = [x for x in precision_range[method].keys() if precision_range[method][x] == None]
            recallIDs = [x for x in recall_range[method].keys() if x not in blanksRecall]
            precisionIDs = [x for x in precision_range[method].keys() if x not in blanksPrecision]
            filteredRecall[method] = {}
            filteredPrecision[method] = {}
            for key in recallIDs:
                #print(recall_range.keys(), flush=True)
                #print(filteredRecall.keys(), flush=True)
                #print(key, flush=True)
                #print(recall_range[key].keys(), flush=True)
                #print(filteredRecall[key].keys(), flush=True)

                filteredRecall[method][key] = recall_range[method][key]
            for key in precisionIDs:
                filteredPrecision[method][key] = precision_range[method][key]        
        except:
            pass

        confusionPickle[method] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}
        try:
            confusionText += "tp %i fp %i fn %i precision %f recall %f method %s\n" % (tp, fp, fn, precision, recall, method)
        except:
            confusionText += "tp %i fp %i fn %i precision %s recall %s method %s\n" % (tp, fp, fn, precision, recall, method)


        print("===%s===" % (method))
        print("Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
        try:
            print("Precision/Recall = %f %f" % (precision, recall))
        except:
            print("Precision/Recall = %s %s" % (precision, recall))
            

    makeGraph(gt_det_hit, gt_det_total, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, "gtdet" + ".png"))
    makeGraph(det_gt_hit, det_gt_total, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, "detgt" + ".png"))
    makeGraph(rot_error, rot_total, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, "rot" + ".png"))
    try:
        makeGraph(filteredRecall, 1.0, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "recall" + ".png"))
        makeGraph(filteredPrecision, 1.0, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "precision" + ".png"))
    except:
        pass

    with open(os.path.join(resultsPath, "conf.pickle"), "wb") as f:
        pickle.dump(confusionPickle, f, protocol=2)
    with open(os.path.join(resultsPath, "conf.txt"), "w") as f:
        f.write(confusionText)

def makeGraph(hists, total, label, xlabel, ylabel, filename):

    fig, ax = plt.subplots()

    for histKey in hists.keys():
        if type(total) is not float:
            if total[histKey] < 1:
                print("Not enough vals for graph!")
                return

        x = []
        y = []
        for key, value in hists[histKey].items():
            x.append(key)
            if type(total) is not float:
                y.append(value/total[histKey])
            else:
                y.append(value)
        
        ax.plot(x, y, label=histKey.split("/")[-1])
    ax.set(xlabel=xlabel, ylabel=ylabel, title='')
    ax.grid(linestyle="--")
    ax.legend(loc='lower right')
    plt.ylim([-0.02, 1.02])
    fig.savefig(filename)
    plt.close('all')

def pointsToImgsDraw(filename, frameCounter, points, wheels, colours):

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
    fn = os.path.join(fn, filename + "-" + str(frameCounter) + ".png")
    cv2.imwrite(fn, accum)

if __name__ == "__main__":

    for method in datasetPaths:
        print("Evaluation method %s" % (method))
        for files in os.walk(method):
            #if files[0] is not method:
                for filename in files[2]:
                    if "17-13-47-41" in filename:
                        if filename[-7:] == ".pickle":
                            filePart = datasetPaths[method]
                            evaluateFile(filename, method, filePart)
    print("Generating Graphs")
    drawGraphs()

