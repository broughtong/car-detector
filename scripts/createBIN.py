import utils
import multiprocessing
import pickle
import os
import numpy as np

datasetPath = "../data/temporal"
outPath = "../data/kitti-bin"
field = "pointclouds"

class Viz(multiprocessing.Process):
    def __init__(self, path, folder, filename):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.folder = folder
        self.filename = filename[:-14]

    def run(self):

        print("Process spawned for file %s" % (os.path.join(self.path, self.folder, self.filename)), flush=True)

        data = []
        with open(os.path.join(self.path, self.folder, self.filename + ".bag.3d.pickle"), "rb") as f:
            data = pickle.load(f)

        os.makedirs(os.path.join(outPath, self.folder), exist_ok=True)
        for frameIdx in range(len(data[field])):
            outfilename = os.path.join(outPath, self.folder, "%s-%i.npy" % (self.filename, frameIdx))
            frame = data[field][frameIdx]

            #x, y, z, intensity, t, reflectivity, ring, ambient, range 
            #bin is x, y, z, reflect
            frame = np.delete(frame, 8, 1)
            frame = np.delete(frame, 7, 1)
            frame = np.delete(frame, 6, 1)
            frame = np.delete(frame, 4, 1)
            frame = np.delete(frame, 3, 1)

            frame = frame.astype('float32')
            #print(outfilename, frame.shape, frame.type)
            #frame.tofile(outfilename)
            np.save(outfilename, frame)

if __name__ == "__main__":

    os.makedirs(outPath, exist_ok=True)
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".3d.pickle" in filename:
                if "2022-06-21-20-45-50" not in filename:
                    continue
                path = datasetPath
                folder = files[0][len(path)+1:]
                jobs.append(Viz(path, folder, filename))

    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 12
    limit = cpuCores
    batch = cpuCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
            limit += batch
            jobs[i].start()


