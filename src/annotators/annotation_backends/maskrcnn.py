import os
import math
import cv2
import utils
import numpy as np

class Annotator():

    def __init__(self, path, scanFields):

        self.path = path
        self.scanFields = scanFields

        for field in scanFields:
            os.makedirs(os.path.join(path, field, "maskrcnn", "all", "imgs"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "maskrcnn", "all", "annotations"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "maskrcnn", "all", "debug"), exist_ok=True)

    def annotate(self, filename, pc, annotations, fieldname):
        
        #pc only
        fn = os.path.join(self.path, fieldname, "maskrcnn", "all", "imgs", filename + ".png")
        utils.drawImgFromPoints(fn, pc, [], [], [], [], dilation=1)

        #masks
        fn = os.path.join(self.path, fieldname, "maskrcnn", "all", "masks", filename + ".png")
        carPoints, nonCarPoints = utils.getInAnnotation(pc, annotations)
        badAnnotation = self.drawAnnotation(fn, annotations)

        #debug
        fn = os.path.join(self.path, fieldname, "maskrcnn", "all", "debug", filename + ".png")
        utils.drawImgFromPoints(fn, pc, [], [], annotations, [], dilation=None)

    def drawAnnotation(self, filename, annotations):
        
        res = 1024
        scale = 25
        ctr = 0

        combineMasks = True

        if combineMasks == False:
            for annotation in annotations:

                accum = np.zeros((res, res, 3))
                
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(2.3*scale), int(4.5*scale)

                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img)
                fn = os.path.join(filename)
                if testImg(accum):
                    return True

                cv2.imwrite(fn, accum)
                ctr += 1

        else:
            accum = np.zeros((res, res, 1))
            annotationIdx = 1
            for annotation in annotations:
                x, y = int(annotation[0]*scale), int(annotation[1]*scale)
                width, height = int(2.4*scale), int(4.85*scale)
                #self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, img, annotationIdx)
                self.drawCar(x+(res//2), y+(res//2), width, height, annotation[2], accum, annotationIdx=annotationIdx)
                annotationIdx += 1
            #fn = os.path.join(foldername, self.filename + "-" + str(frame) + ".png")
            if self.testImg(accum):
                return True
            cv2.imwrite(filename, accum)

        return False

    def testImg(self, img):

        mask = np.array(img)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask[:,:,0] == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                return True
        return False



    def drawCar(self, centreX, centreY, fullHeight, fullWidth, angle, img, debugimg=None, annotationIdx=None):

        angle = angle % (math.pi * 2)
        alpha = math.cos(angle) * 0.5
        beta = math.sin(angle) * 0.5
        a = [int(centreY + alpha * fullHeight - beta * fullWidth), int(centreX - beta * fullHeight - alpha * fullWidth)]
        b = [int(centreY - alpha * fullHeight - beta * fullWidth), int(centreX + beta * fullHeight - alpha * fullWidth)]
        c = [int(2 * centreY - a[0]), int(2 * centreX - a[1])]
        d = [int(2 * centreY - b[0]), int(2 * centreX - b[1])]

        contours = np.array([a, b, c, d])
        if annotationIdx == None:
            colour = lambda : [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            cv2.fillPoly(img, pts = [contours], color = colour())
        else:
            colour = annotationIdx
            cv2.fillPoly(img, pts = [contours], color = colour)

        if debugimg is not None:
            cv2.fillPoly(debugimg, pts = [contours], color =(255,0,255))

            debugimg[centreX, centreY] = [255, 0, 255]
            debugimg[a[0], a[1]] = [125, 0, 128]
            debugimg[b[0], b[1]] = [125, 255, 0]
            debugimg[c[0], c[1]] = [125, 255, 0]
            debugimg[d[0], d[1]] = [125, 255, 0]


