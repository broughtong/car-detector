import cv2
import multiprocessing
import pickle
import os
import math
import numpy as np
import copy
import multiprocessing

def drawImgFromPoints(filename, points, otherPoints=[], otherColours=[], cars=[], cars2=[], dilation=None, renderAnnotations=False):

    cars = copy.deepcopy(cars)
    cars2 = copy.deepcopy(cars2)

    res = 1024
    scale = 25
    img = np.zeros((res, res, 3))
    img.fill(255)

    for car in cars:

        car[0] = int((car[0] * scale) + (res/2))
        car[1] = int((car[1] * scale) + (res/2))

        angle = car[2] % math.pi
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        width, height = int(4.5*scale), int(2.3*scale)

        a = [int(car[1] + alpha * height - beta * width), int(car[0] - beta * height - alpha * width)]
        b = [int(car[1] - alpha * height - beta * width), int(car[0] + beta * height - alpha * width)]
        c = [int(2 * car[1] - a[0]), int(2 * car[0] - a[1])]
        d = [int(2 * car[1] - b[0]), int(2 * car[0] - b[1])]

        contours = np.array([a, b, c, d])
        cv2.fillPoly(img, pts=[contours], color=[255, 0, 255])

        arrowLength = 50
        start = [int(car[1]), int(car[0])]
        end = [int(car[1]-arrowLength*math.sin(car[2])), int(car[0]-arrowLength*math.cos(car[2]))]
        img = cv2.arrowedLine(img, tuple(start), tuple(end), [255, 0, 0], 2, 1)

    for car in cars2:

        car[0] = int((car[0] * scale) + (res/2))
        car[1] = int((car[1] * scale) + (res/2))

        angle = car[2] % (math.pi * 2)
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        width, height = int(4.5*scale), int(2.3*scale)

        a = [int(car[1] + alpha * height - beta * width), int(car[0] - beta * height - alpha * width)]
        b = [int(car[1] - alpha * height - beta * width), int(car[0] + beta * height - alpha * width)]
        c = [int(2 * car[1] - a[0]), int(2 * car[0] - a[1])]
        d = [int(2 * car[1] - b[0]), int(2 * car[0] - b[1])]

        contours = np.array([a, b, c, d])
        cv2.fillPoly(img, pts=[contours], color=[0, 255, 0])

        arrowLength = 50
        start = [int(car[1]), int(car[0])]
        end = [int(car[1]-arrowLength*math.sin(car[2])), int(car[0]-arrowLength*math.cos(car[2]))]
        img = cv2.arrowedLine(img, tuple(start), tuple(end), [255, 0, 0], 2, 1)

    for point in points:
        x, y = point[:2]
        x *= scale
        y *= scale
        x = int(x)
        y = int(y)
        try:
            img[x+int(res/2), y+int(res/2)] = [0, 0, 0]
        except:
            pass

    if dilation is not None:
        kernel = np.ones((dilation, dilation), 'uint8')
        img = cv2.erode(img, kernel, iterations=1)

    for point in range(len(otherPoints)):
        x, y = otherPoints[point][:2]
        x *= scale
        y *= scale
        x = int(x)
        y = int(y)
        try:
            img[x+int(res/2), y+int(res/2)] = otherColours[point]
        except:
            pass

        try:
            img[x+int(res/2)+1, y+int(res/2)+1] = otherColours[point]
        except:
            pass
        try:
            img[x+int(res/2)+1, y+int(res/2)-1] = otherColours[point]
        except:
            pass
        try:
            img[x+int(res/2)-1, y+int(res/2)+1] = otherColours[point]
        except:
            pass
        try:
            img[x+int(res/2)-1, y+int(res/2)-1] = otherColours[point]
        except:
            pass

    if renderAnnotations:
        cv2.putText(img, "%s cars" % (str(len(cars))), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, 50)

    cv2.imwrite(filename, img)

class Visualise(multiprocessing.Process):
    def __init__(self, datasetPath, outPath, path, filename):
        multiprocessing.Process.__init__(self)

        self.filename = filename
        self.path = path
        self.fileCounter = 0
        self.datasetPath = datasetPath
        self.outPath = outPath

    def run(self):
        
        print("Process spawned for file %s" % (self.filename), flush = True)

        with open(os.path.join(self.path, self.filename), "rb") as f:
            self.data = pickle.load(f)

        for frameIdx in range(len(self.data["scans"])):
            self.drawFrame(frameIdx)

    def drawFrame(self, idx):

        scans = self.data["scans"][idx]

        try:
            scans = np.concatenate([scans["sick_back_left"], scans["sick_back_right"], scans["sick_back_middle"], scans["sick_front"]])
        except:
            return

        fn = os.path.join(self.outPath, self.filename + "-" + str(idx) + ".png")
        drawImgFromPoints(fn, scans)

if __name__ == "__main__":


    datasetPath = "../data/extracted/"
    outPath = "../visualisation/extractor"
    os.makedirs(outPath, exist_ok=True)

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            jobs.append(Visualise(datasetPath, outPath, files[0], filename))
    print("Spawned %i processes" % (len(jobs)), flush = True)
    maxCores = 1
    limit = maxCores
    batch = maxCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()

