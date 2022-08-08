import math

#functions
#isInsideAnnotation = give it a 2d point (1st arg) and an annotation, will return if inside annotation or not
#getinannotation function = give points and annotations, will return array of all annotated points (not by car), and an array of all non annotated points

#usage:
#format:
#scans = array of points from lidar
#annotations = array of car positions defined as
#[x, y, yaw] yaw is rads counter clockwise

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

