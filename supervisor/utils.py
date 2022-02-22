import cv2
import math
import numpy as np

def drawImgFromPoints(filename, points, otherPoints, otherColours, cars, dilation=None):

    res = 1024
    scale = 25
    img = np.zeros((res, res, 3))
    img.fill(255)

    for car in cars:

        angle = car[2] % (math.pi * 2)
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        width, height = int(2.3*scale), int(4.5*scale)

        a = [int(car[1] + alpha * height - beta * width), int(car[0] - beta * height - alpha * width)]
        b = [int(car[1] - alpha * height - beta * width), int(car[0] + beta * height - alpha * width)]
        c = [int(2 * car[1] - a[0]), int(2 * car[0] - a[1])]
        d = [int(2 * car[1] - b[0]), int(2 * car[0] - b[1])]

        contours = np.array([a, b, c, d])
        cv2.fillPoly(img, pts=[contours], color=[255, 0, 255])

        arrowLength = 50
        end = [int(car[0]+arrowLength*math.sin(car[2])), int(car[1]-arrowLength*math.cos(car[2]))]
        img = cv2.arrowedLine(img, tuple(car[:2]), tuple(end), [0, 0, 255], 2, 1)

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

    cv2.imwrite(filename, img)

