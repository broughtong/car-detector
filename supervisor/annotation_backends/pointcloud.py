import os
import utils
import numpy as np

class Annotator():

    def __init__(self, path, scanFields):

        self.path = path
        self.scanFields = scanFields

        for field in scanFields:
            os.makedirs(os.path.join(path, field, "pointcloud-bin", "all", "cloud"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "pointcloud-bin", "all", "annotations"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "pointcloud-npy", "all", "cloud"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "pointcloud-npy", "all", "annotations"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "pointcloud-ply", "all", "cloud"), exist_ok=True)
            os.makedirs(os.path.join(path, field, "pointcloud-ply", "all", "annotations"), exist_ok=True)

    def annotate(self, filename, pc, annotations, fieldname):
        
        #pc only
        fn = os.path.join(self.path, fieldname, "pointcloud-bin", "all", "cloud", filename + ".bin")
        self.saveCloudBIN(fn, pc)
        fn = os.path.join(self.path, fieldname, "pointcloud-npy", "all", "cloud", filename + ".npy")
        self.saveCloudNPY(fn, pc)
        fn = os.path.join(self.path, fieldname, "pointcloud-ply", "all", "cloud", filename + ".ply")
        self.saveCloudPLY(fn, pc)

        #masks
        #fn = os.path.join(self.path, field, "maskrcnn", "all", "masks", filename + ".png")
        #carPoints, nonCarPoints = utils.getInAnnotation(newScan, newAnnotations)
        #badAnnotation = utils.drawAnnotation(fn, frame, newAnnotations)

    def saveCloudBIN(self, fn, pc):
        
        np.save(fn, pc)

    def saveCloudNPY(self, fn, pc):

        np.save(fn, pc)

    def saveCloudPLY(self, fn, pc):
        
        header = """ply
format ascii 1.0
element vertex %i
property float x
property float y
property float z
end_header
"""

        with open(fn, "w") as f:
            nPoints = len(pc)
            f.write(header % (nPoints))
            for p in pc:
                f.write("%f %f %f\n" % (p[0], p[1], p[2]))

