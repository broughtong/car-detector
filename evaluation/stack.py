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

detectorThreshold = 0.92

def evaluateFile(path, folder, filename, datasetPath, tfPath, gtPath, annoField, resultsPath):

    fn = os.path.join(datasetPath, folder, filename)
    if not os.path.isfile(fn):
        print("Unable to open file: %s" % (fn))
        return
    
    tffn = "".join(filename.split(".data")[0]) + ".pickle"
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
    scanfn = os.path.join("../data/extracted/", folder, filename[:-12] + ".scans.pickle")
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

    detectionGraphLimit = 1.0
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

        detectionThreshold = 1.75

        #confusion matrix
        detections = copy.deepcopy(data[annoField][dataFrameIdx])
        filteredDetections = []
        for d in detections:
            if len(d) > 3:
                if d[3] < detectorThreshold:
                    continue
            filteredDetections.append(d)
        detections = filteredDetections
        gts = copy.deepcopy(frameAnnotations)[1:]

        utils.drawImgFromPoints("viz/" + filename + "-" + str(fileCounter) + ".png", dataFrame, cars = gts, cars2=detections)

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
                    #carPoints, nonCarPoints = utils.getInAnnotation(dataFrame, [gt])
                    #if len(carPoints) > 15:
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

    f1 = None
    try:
        f1 = (2*tp)/((2*tp) + fp + fn)
    except:
        pass

    print("Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
    print("Precision/Recall/F1 = %f %f %f" % (precision, recall, f1))
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
    results.append([path, folder, filename, annoField])
    return results

def makeGraph(hist, total, label, xlabel, ylabel, filename):

    if total < 1:
        print("Not enough vals for graph!")
        return

    newlabels = []
    for i in label:
        if "lanois" in i:
            newlabels.append("MaskRCNN + LaNoise")
        elif "scan" in i:
            newlabels.append("MaskRCNN")
        elif "dete" in i:
            newlabels.append("Clustering")
    label = newlabels            

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
        
def makeGraphs(hist, total, labels, title, xlabel, ylabel, filename):

    for i in total:
        if i < 1:
            print("Not evenough vals for graph!")
            return
    if len(hist) != len(total):
        print("Unequal!", title)

    newlabels = []
    for i in labels:
        if "lanois" in i:
            newlabels.append("MaskRCNN + LaNoise")
        elif "scan" in i:
            newlabels.append("MaskRCNN")
        elif "tempo" in i:
            newlabels.append("Clustering")
    labels = newlabels    

    fig, ax = plt.subplots()
    for i, v in enumerate(hist):
        x = []
        y = []
        last = 0
        for key, value in v.items():
            x.append(key)
            y.append(value/total[i])
            last = value/total[i]
    

        ax.plot(x, y, label=labels[i])
    #ax.set(xlabel=xlabel, ylabel=ylabel, title='')
    ax.legend(loc="lower right")
    ax.set_xlabel(xlabel, fontsize = 16)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.grid(linestyle="--")
    #ax.legend(loc='lower right')
    plt.ylim([-0.02, 1.02])
    if "ange" in title:
        plt.xlim([6, 20])
    fig.savefig(filename)
    plt.close('all')

def generateGraphs(results, evalName):

    resultsPath = os.path.join("./results/", evalName)
    os.makedirs(resultsPath, exist_ok=True)
    graphPath = os.path.join(resultsPath, "graph")
    os.makedirs(graphPath, exist_ok=True)

    usedIdxs = []
    total = []
    for i in range(len(results)):
        if i in usedIdxs:
            continue
        usedIdxs.append(i)
        method = results[i][-1][0].split("/")[-1] + "-" + results[i][-1][-1]
        filename = results[i][-1][2]
        
        gt_det_hit = [results[i][7]]
        gt_det_total = [results[i][6]]
        det_gt_hit = [results[i][9]]
        det_gt_total = [results[i][8]]
        rot_error = [results[i][11]]
        rot_total = [results[i][10]]
        filteredRecall = [results[i][12]] 
        filteredPrecision = [results[i][13]]
        labels = [method]
        
        for j in range(i+1, len(results)):
            if j in usedIdxs:
                continue
            if results[j][-1][2] == results[i][-1][2]:
                usedIdxs.append(j)
                nmethod = results[j][-1][0].split("/")[-1]  + "-" + results[i][-1][-1]
                nfilename = results[j][-1][2]

                gt_det_hit.append(results[j][7])
                gt_det_total.append(results[j][6])
                det_gt_hit.append(results[j][9])
                det_gt_total.append(results[j][8])
                rot_error.append(results[j][11])
                rot_total.append(results[j][10])
                filteredRecall.append(results[j][12])
                filteredPrecision.append(results[j][13])
                labels.append(nmethod)
                
        makeGraphs(gt_det_hit, gt_det_total, labels, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, filename + "-gtdet.png"))
        makeGraphs(det_gt_hit, det_gt_total, labels, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, filename + "-detgt.png"))
        makeGraphs(rot_error, rot_total, labels, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, filename + "-rot.png"))
        makeGraphs(filteredRecall, [1.0]*len(filteredRecall), labels, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, filename + "-recall.png"))
        makeGraphs(filteredPrecision, [1.0]*len(filteredPrecision), labels, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, filename + "-precision.png"))

    gt_det_hit = []
    gt_det_total = []
    det_gt_hit = []
    det_gt_total = []
    rot_error = []
    rot_total = []
    filteredRecall = []
    filteredPrecision = []
    labels = []
    tp_range = []
    fp_range = []
    fn_range = []

    for i in range(len(results)):
        method = results[i][-1][0].split("/")[-1] + "-" + results[i][-1][-1]

        if method not in labels:
            gt_det_hit.append(results[i][7])
            gt_det_total.append(results[i][6])
            det_gt_hit.append(results[i][9])
            det_gt_total.append(results[i][8])
            rot_error.append(results[i][11])
            rot_total.append(results[i][10])
            filteredRecall.append(results[i][12])
            filteredPrecision.append(results[i][13])
            labels.append(method)
            tp_range.append([results[i][3]])
            fp_range.append([results[i][4]])
            fn_range.append([results[i][5]])
        else:
            idx = labels.index(method)
            ngt_det_hit = {}
            for key, value in gt_det_hit[idx].items():
                ngt_det_hit[key] = value + results[i][7][key]
            gt_det_total[idx] += results[i][6]
            gt_det_hit[idx] = ngt_det_hit
            ndet_gt_hit = {}
            for key, value in det_gt_hit[idx].items():
                ndet_gt_hit[key] = value + results[i][9][key]
            det_gt_total[idx] += results[i][8]
            det_gt_hit[idx] = ngt_det_hit
            nrot_error = {}
            for key, value in rot_error[idx].items():
                nrot_error[key] = value + results[i][11][key]
            rot_total[idx] += results[i][10]
            rot_error[idx] = nrot_error
            tp_range[idx].append(results[i][3])
            fp_range[idx].append(results[i][4])
            fn_range[idx].append(results[i][5])

        #to get filtered recall precision,
        #you need to take the results[3,4,5] ie tp_range, fn, fp
        #and manually sum them, then regenerate teh prec/recall range from that

    filteredRecall = []
    filteredPrecision = []
    lastVal = None
    for methodIdx in range(len(tp_range)):
        totalTP = {}
        totalFP = {}
        totalFN = {}
        for fileIdx in range(len(tp_range[methodIdx])):
            tp = tp_range[methodIdx][fileIdx]
            fp = fp_range[methodIdx][fileIdx]
            fn = fn_range[methodIdx][fileIdx]
            if len(totalTP.keys()) == 0:
                for key, item in tp.items():
                    totalTP[key] = item
                    lastVal = key
                for key, item in fp.items():
                    totalFP[key] = item
                for key, item in fn.items():
                    totalFN[key] = item
            else:
                for key, item in tp.items():
                    totalTP[key] += item
                    lastVal = key
                for key, item in fp.items():
                    totalFP[key] += item
                for key, item in fn.items():
                    totalFN[key] += item

        recall_range = {}
        precision_range = {}
        for val in totalTP.keys():
            if totalTP[val] == 0 and totalFN[val] == 0:
                recall_range[val] = None
            else:
                recall_range[val] = totalTP[val] / (totalTP[val]+totalFN[val])
            if totalTP[val] == 0 and totalFP[val] == 0:
                precision_range[val] = None
            else:
                precision_range[val] = totalTP[val] / (totalTP[val]+totalFP[val])

        precision = None
        recall = None
        try:
            precision = totalTP[lastVal] / (totalTP[lastVal] + totalFP[lastVal])
            recall = totalTP[lastVal] / (totalTP[lastVal] + totalFN[lastVal])
        except:
            print("Didnt work")
        tp = totalTP[lastVal]
        fn = totalFP[lastVal]
        fp = totalFN[lastVal]

        try:
            blanksRecall = [x for x in recall_range.keys() if recall_range[x] == None]
            blanksPrecision = [x for x in precision_range.keys() if precision_range[x] == None]
            recallIDs = [x for x in recall_range.keys() if x not in blanksRecall]
            precisionIDs = [x for x in precision_range.keys() if x not in blanksPrecision]
            filteredRecall.append({})
            for key in recallIDs:
                filteredRecall[-1][key] = recall_range[key]
            filteredPrecision.append({})
            for key in precisionIDs:
                filteredPrecision[-1][key] = precision_range[key]
        except:
            print("dindit work")
            pass
        if precision == None:
            precision = -1
        if recall == None:
            recall = -1
        f1 = None
        try:
            f1 = (2*tp)/((2*tp) + fp + fn)
        except:
            pass

        print("====")
        print(str(labels[methodIdx]))
        print("Total Frame Confusion Matrix (tp/fp/fn):", tp, fp, fn)
        print("Total Precision/Recall/F1 = %f %f %f" % (precision, recall, f1))
        with open(os.path.join(graphPath, "..", str(labels[methodIdx]) + ".txt"), "w") as f:
            f.write("tp %i fp %i fn %i precision %f recall %f f1 %f \n" % (tp, fp, fn, precision, recall, f1))

    makeGraphs(gt_det_hit, gt_det_total, labels, "GT To Detection", "Distance [m]", "Probability", os.path.join(graphPath, "total-gtdet.png"))
    makeGraphs(det_gt_hit, det_gt_total, labels, "Detection To GT", "Distance [m]", "Probability", os.path.join(graphPath, "total-detgt.png"))
    makeGraphs(rot_error, rot_total, labels, "Rotational Error", "Rotation [rads]", "Probability", os.path.join(graphPath, "total-rot.png"))
    makeGraphs(filteredRecall, [1.0]*len(filteredRecall), labels, "Recall Range", "Distance [m]", "Recall", os.path.join(graphPath, "total-recall.png"))
    makeGraphs(filteredPrecision, [1.0]*len(filteredPrecision), labels, "Precision Range", "Distance [m]", "Precision", os.path.join(graphPath, "total-precision.png"))
            

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
                #if "drive" not in filename and "one" not in filename:
                #    continue
                #if "24-42" not in filename and "10-16" not in filename:
                #    continue
                bgs = ["08-42", "42-54", "22-07", "53-06", "10-16", "24-42"]
                f = False
                for i in bgs:
                    if i in filename:
                        f = True
                if f == False:
                    continue
                
                path = datasetPath
                folder = files[0][len(path)+1:]
                vals = evaluateFile(path, folder, filename, datasetPath, tfPath, gtPath, annoField, resultsPath)
                print(path, folder, filename, annoField)
                #print(vals)
                if vals is not None:
                    tp += vals[0]
                    fp += vals[1]
                    fn += vals[2]
                    results.append(vals)
                break
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
    with open(os.path.join(resultsPath, str(detectorThreshold) + "-cumulative.txt"), "w") as f:
        f.write("f1 %f tp %i fp %i fn %i p %f r %f samples %i %s\n" % (f1, tp, fp, fn, precision, recall, samples, resultNameString))
    return results

if __name__ == "__main__":

    results = []
    tfPath = "../data/static_tfs"
    gtPath = "../data/gt"

    #single temporal
    datasetPath = "../data/temporal/temporal-0.6-20-10-20-100.6-20-10-20-10"
    annoField = "annotations"
    results += run(datasetPath, tfPath, gtPath, annoField)
    generateGraphs(results, "eval-maskrcnn3")
    annoField = "extrapolated"
    results += run(datasetPath, tfPath, gtPath, annoField)
    generateGraphs(results, "eval-maskrcnn3")

    """
    #multi temporal
    annoField = "extrapolated"
    ctr = 0
    for files in os.walk("../data/temporal"):
        for filename in files[2]:
            if "statistics" not in filename:
                continue
            datasetPath = files[0] #"/".join(files[0].split("/")[:-1])
            results += run(datasetPath, tfPath, gtPath, annoField)
            ctr += 1
            print(ctr)

            generateGraphs(results, "eval-temp-all-tog-1.25-q")
    """

    print("fin")
    import sys
    sys.exit(0)

    #masrcnk
    annoField = "maskrcnn"
    ctr = 0
    for files in os.walk("../data/maskrcnn/rectified"):
        for filename in files[2]:
            if "statistics" not in filename:
                continue
            datasetPath = files[0] #"/".join(files[0].split("/")[:-1])
            if "32_40" not in datasetPath and "59_44" not in datasetPath:
                continue
            #if "22_32" in datasetPath or "56_23" in datasetPath or "59_44" in datasetPath or "25_55" in datasetPath or "29_17" in datasetPath or "36_04" in datasetPath:
            #    print("Skipping ", datasetPath) 
            #    continue
            results += run(datasetPath, tfPath, gtPath, annoField)
            ctr += 1
            print(ctr)

            generateGraphs(results, "eval-maskrcnn3")


    """
    #threshoold
    annoField = "maskrcnn"
    ctr = 0
    for files in os.walk("../data/maskrcnn/rectified"):
        for filename in files[2]:
            if "statistics" not in filename:
                continue
            datasetPath = files[0] #"/".join(files[0].split("/")[:-1])
            print(datasetPath, annoField)
            detectorThreshold = 0.9
            for i in range(87, 100, 1):
                detectorThreshold = i/100
                print("det threshd", detectorThreshold)
                results += run(datasetPath, tfPath, gtPath, annoField)
            ctr += 1
            print(ctr)

            generateGraphs(results, "eval-maskrcnn")
            break
    

    """
