import cv2
import multiprocessing
import pickle
import os
import math
import numpy as np
import copy
import multiprocessing

def drawPLY(filename, scan):
    header = """ply
format ascii 1.0
element vertex %i
property float x
property float y
property float z
property uint8 red
property uint8 green
property uint8 blue
end_header
"""
    
    with open(filename, "w") as f:
        f.write(header % (len(scan)))
        for p in scan:
            f.write("%f %f %f %i %i %i\n" % (p[0], p[1], p[2], 255, 0, 0))

def combineScans(arrOfScans):

    scans = []
    for key in arrOfScans.keys():
        arrOfScans[key] = np.array(arrOfScans[key])
        arrOfScans[key] = arrOfScans[key].reshape([arrOfScans[key].shape[0], 4])
        scans.append(arrOfScans[key])

    return np.concatenate(scans)

def drawImgFromPoints(filename, points, otherPoints=[], otherColours=[], cars=[], cars2=[], dilation=None, renderAnnotations=False):

    #filename
    #scan points
    #highlighted points
    #highlighted point colours
    #car position
    #second arr car position
    #dilate?
    #render annotations?

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
        width, height = int(4.9*scale), int(2.5*scale)

        a = [int(car[1] + alpha * height - beta * width), int(car[0] - beta * height - alpha * width)]
        b = [int(car[1] - alpha * height - beta * width), int(car[0] + beta * height - alpha * width)]
        c = [int(2 * car[1] - a[0]), int(2 * car[0] - a[1])]
        d = [int(2 * car[1] - b[0]), int(2 * car[0] - b[1])]

        contours = np.array([a, b, c, d])
        cv2.fillPoly(img, pts=[contours], color=[255, 0, 255])

    for car in cars2:

        car[0] = int((car[0] * scale) + (res/2))
        car[1] = int((car[1] * scale) + (res/2))

        angle = car[2] % (math.pi * 2)
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        width, height = int(4.85*scale), int(2.4*scale)

        a = [int(car[1] + alpha * height - beta * width), int(car[0] - beta * height - alpha * width)]
        b = [int(car[1] - alpha * height - beta * width), int(car[0] + beta * height - alpha * width)]
        c = [int(2 * car[1] - a[0]), int(2 * car[0] - a[1])]
        d = [int(2 * car[1] - b[0]), int(2 * car[0] - b[1])]

        contours = np.array([a, b, c, d])
        cv2.fillPoly(img, pts=[contours], color=[0, 255, 0])

    for car in cars:

        car[0] = int((car[0] * scale) + (res/2))
        car[1] = int((car[1] * scale) + (res/2))

        arrowLength = 50
        start = [int(car[1]), int(car[0])]
        end = [int(car[1]-arrowLength*math.sin(car[2])), int(car[0]-arrowLength*math.cos(car[2]))]
        img = cv2.arrowedLine(img, tuple(start), tuple(end), [255, 0, 0], 2, 1)

    for car in cars2:

        car[0] = int((car[0] * scale) + (res/2))
        car[1] = int((car[1] * scale) + (res/2))

        arrowLength = 50
        start = [int(car[1]), int(car[0])]
        end = [int(car[1]-arrowLength*math.sin(car[2])), int(car[0]-arrowLength*math.cos(car[2]))]
        img = cv2.arrowedLine(img, tuple(start), tuple(end), [255, 120, 0], 2, 1)

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

def getInAnnotation(scan, annotations):

    carPoints = []
    nonCarPoints = []

    for point in scan:

        inAnnotation = False

        for annotation in annotations:
            if isInsideAnnotation(point[:2], annotation):
                inAnnotation = True
                break

        if inAnnotation:
            carPoints.append(point)
        else:
            nonCarPoints.append(point)

    return carPoints, nonCarPoints

def isInsideAnnotation(pos, annotation):

    poly = getBoundaryPoints(annotation)

    counterPos = (9999, pos[1])
    intersects = 0

    for idx in range(len(poly)):
        pa = poly[idx]
        pb = poly[0]
        if idx != len(poly)-1:
            pb = poly[idx+1]

        print(pos, counterPos, pa, pb)
        if lineIntersect(pos, counterPos, pa, pb):
            print("int!")
            intersects += 1

    return intersects % 2

def getBoundaryPoints(poly):

    centreX = poly[0]#*scale) + (res//2)
    centreY = poly[1]#*scale) + (res//2)
    angle = poly[2] % (math.pi*2)
    height = 4.85# * scale
    width = 2.4# * scale

    alpha = math.cos(angle) * 0.5
    beta = math.sin(angle) * 0.5

    a = [centreX - beta * height - alpha * width, centreY + alpha * height - beta * width]
    b = [centreX + beta * height - alpha * width, centreY - alpha * height - beta * width]
    c = [2 * centreX - a[0], 2 * centreY - a[1]]
    d = [2 * centreX - b[0], 2 * centreY - b[1]]

    return a, b, c, d

def lineIntersect(a, b, c, d):
    o1 = tripletOrientation(a, b, c)
    o2 = tripletOrientation(a, b, d)
    o3 = tripletOrientation(c, d, a)
    o4 = tripletOrientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True

    if ((o1 == 0) and colinear(a, c, b)):
        return True
    if ((o2 == 0) and colinear(a, d, b)):
        return True
    if ((o3 == 0) and colinear(c, a, d)):
        return True
    if ((o4 == 0) and colinear(c, b, d)):
        return True
    return False

def tripletOrientation(a, b, c):
    orr = ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))
    if orr > 0:
        return 1
    elif orr < 0:
        return -1
    return 0

def colinear(a, b, c):
    if ((b[0] <= max(a[0], c[0])) and (b[0] >= min(a[0], c[0])) and (b[1] <= max(a[1], c[1])) and (b[1] >= min(a[1], c[1]))):
        return True
    else:
        return False

