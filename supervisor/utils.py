import cv2
import math
import numpy as np
import copy

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

