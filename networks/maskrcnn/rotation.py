import numpy
import math
import numpy as np
from numpy.linalg import eig
import cv2

def get(fn):

    img = cv2.imread(fn)
    img = img[:, :, 0]
    coords = []
    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x][y] == 255:
                coords.append([x, y])
    cov = np.cov(numpy.transpose(coords))

    w, v = eig(cov)

    bigIdx = 0
    if w[1] > w[0]:
        bigIdx = 1

    ev = v[bigIdx]
    angle = math.atan2(ev[1], ev[0])
    print("angle, ", angle)
    print("=====")

get("b.png")
get("c.png")
get("d.png")
get("e.png")

