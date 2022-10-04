#!/usr/bin/python
import utils
import matplotlib.pyplot as plt
import math
import copy
import cv2
import numpy as np
import rospy
import pickle
import os
import sys
    
detectionGraphLimit = 2.0
confusionGraphLimit = 15.0
graph_resolution = 250
detectionThreshold = 1.0
closestOnly = True

evalName = "full-train-5"
datasetPaths = {"../data/results/detector-s": "annotations"}
datasetPaths["../data/results/temporal-new-0.8-50-6-75-6"] = "extrapolated"
datasetPaths["../data/results/maskrcnn_scans_rectified-xy/scans-25-04-22-18_08_16.pth"] = "maskrcnn"
minimumMove = 0.1
closestOnly = False

evalName = "heval-temp-mask-plain"
datasetPaths = {}
#datasetPaths = {"../data/results/detector-s": "annotations"}
#datasetPaths["../data/results/temporal-new-0.8-50-5-50-10"] = "extrapolated"
datasetPaths["/home/broughtong/external/broughtong/maskrcnn_scans_rectified/scans-06-05-22-20_22_40.pth"] = "maskrcnn"
#datasetPaths["/home/broughtong/external/broughtong/maskrcnn_scans_rectified/lanoising-06-05-22-23_57_24.pth"] = "maskrcnn"
datasetPaths["/home/broughtong/external/broughtong/maskrcnn_scans_rectified/scans-06-05-22-20_18_46.pth"] = "maskrcnn"


#datasetPaths["/home/broughtong/external/broughtong/maskrcnn_scans_rectified/lanoising-07-05-22-00_09_06.pth"] = "maskrcnn"
#datasetPaths["../data/results/pns-eval-12"] = "pointnet"
#datasetPaths["../data/results/pns-eval-4"] = "pointnet"

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

    lastMove = None

    with open(fn, "rb") as f:
        data = pickle.load(f)
    with open(tffn, "rb") as f:
        tfdata = pickle.load(f)
    with open(gtfn, "rb") as f:
        gtdata = pickle.load(f, encoding='latin1')
    frameCounter = 0
    for frame in gtdata[1]: #back middle sensor, higher framerate
        frameCounter += 1
        gttime = rospy.Time(frame[0].secs, frame[0].nsecs)
        if gttime not in data["ts"]:
            print("Warning, no data for gt!", fn, gtfn)
            #print(fn, gtfn)
            #print(frame, frameCounter)
            #print(frame[0].secs, frame[0].nsecs)
            #print(gttime)
            continue

        dataFrameIdx = data["ts"].index(gttime)
        dataFrame = data["scans"][dataFrameIdx]
        dataFrame = combineScans(dataFrame)

        if lastMove is None:
            #no previous frame, set position
            lastMove = data["trans"][dataFrameIdx]
        else:
            #prevent stationary vehicle having multiple eval frames in same spot
            dx = lastMove[0][-1] - data["trans"][dataFrameIdx][0][-1]
            dy = lastMove[1][-1] - data["trans"][dataFrameIdx][1][-1]
            diff = ((dx**2) + (dy**2))**0.5
            if diff < minimumMove:
                continue
            lastMove = data["trans"][dataFrameIdx]

        frameAnnotations = []
        frameAnnotations.append(frame[0])
        closestCar = None

        for i in range(1, len(frame)):
            rotation = frame[i][2]
            position = [*frame[i][:2], 0, 1]
            dist = (frame[i][0]**2) + (frame[i][1]**2) ** 0.5
            if dist < 1:
                continue
            mat = tfdata["sick_back_middle"]["mat"]
            position = np.matmul(mat, position)
            quat = tfdata["sick_back_middle"]["quaternion"]
            qx = quat[0]
            qy = quat[1]
            qz = quat[2]
            qw = quat[3]
            yaw = math.atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
            car = [*position[:2], yaw-rotation]
            if not closestOnly:
                frameAnnotations.append(car)
            else:
                if closestCar == None:
                    closestCar = car
                else:
                    distOld = ((closestCar[0]**2) + (closestCar[1]**2)) ** 0.5
                    distNew = ((car[0]**2) + (car[1]**2)) ** 0.5
                    if distNew < distOld:
                        closestCar = car

        if closestOnly and closestCar is not None:
            frameAnnotations.append(closestCar)

        #confusion matrix
        detections = copy.deepcopy(data[filePart][dataFrameIdx])
        gts = copy.deepcopy(frameAnnotations)[1:]

        for rng in tp_range[method].keys():
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
                if diff < detectionThreshold:
                    jo = j[2] % (math.pi*2)
                    gto = gt[2] % (math.pi*2)
                    diff = abs(jo-gto)
                    if diff > (math.pi/2):
                        if jo > gto:
                            jo -= (math.pi)
                        else:
                            gto -= (math.pi)
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
        meth = method.split("/")[-1]
        utils.drawImgFromPoints("../visualisation/" + evalName + "/" + filename + "-" + str(meth) + "-" + str(frameCounter) + ".png", dataFrame, [], [], gts, detections, 3)

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
    makeGraph(rot_error, rot_total, "Rotational Error", "Rotational Error [rad]", "Probability", os.path.join(graphPath, "rot" + ".png"), invert=True)
    try:
        makeGraph(filteredRecall, 1.0, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "recall" + ".png"))
        makeGraph(filteredPrecision, 1.0, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "precision" + ".png"))
    except:
        pass

    with open(os.path.join(resultsPath, "conf.pickle"), "wb") as f:
        pickle.dump(confusionPickle, f, protocol=2)
    with open(os.path.join(resultsPath, "conf.txt"), "w") as f:
        f.write(confusionText)

def makeGraph(hists, total, label, xlabel, ylabel, filename, invert=False):

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
                if invert:
                    y.append(1-(value/total[histKey]))
                else:
                    y.append(value/total[histKey])
            else:
                label = histKey.split("/")[-1]
                if "06-05-22-20_18_46" in label:
                    y.append(value + 0.02)
                else:
                    y.append(value)
        
        label = histKey.split("/")[-1]
        if "06-05-22-20_18_46" in label:
            label = "Hand Annotated"
        else:
            label = "Auto Annotated"
        if label == "detector-s":
            label = "Simple Detector"
        if label == "temporal-new-0.8-50-5-50-10":
            label = "Temporal Filter"
        if "scans" in label:
            label = "Mask-RCNN"
        if "lanoi" in label:
            label = "Mask-RCNN + Lanoising"
        ax.plot(x, y, label=label)
    #ax.set(xlabel=xlabel, ylabel=ylabel, title='')#, fontsize=18)
    #ax.set_title('',fontweight="bold", size=20) # Title
    ax.set_ylabel(ylabel, fontsize = 16.0) # Y label
    ax.set_xlabel(xlabel, fontsize = 16) # X label
    ax.grid(linestyle="--")
    ax.set_xlim([5.75, 15])
    if invert:
        ax.legend(loc='upper right')
    else:
        ax.legend(loc='lower right')
    plt.ylim([-0.02, 1.02])
    fig.savefig(filename, dpi=250)
    plt.close('all')
    
if __name__ == "__main__":

    for method in datasetPaths:
        print("Evaluation method %s" % (method))

        for val in np.linspace(0.0, detectionGraphLimit, num=graph_resolution):
            gt_det_hit[method][val] = 0
            det_gt_hit[method][val] = 0
        for val in np.linspace(0.0, (math.pi/2) + 0.01, num=graph_resolution):
            rot_error[method][val] = 0
        for val in np.linspace(0.0, confusionGraphLimit, num=graph_resolution):
            tp_range[method][val] = 0
            fn_range[method][val] = 0
            fp_range[method][val] = 0

        for files in os.walk(method):
            for filename in files[2]:
                if "2022-02-15-15-08-59.bag" in filename:
                    continue #moving bag, ignore
                if filename[-7:] == ".pickle":
                    filePart = datasetPaths[method]
                    evaluateFile(filename, method, filePart)
    print("Generating Graphs")
    drawGraphs()

