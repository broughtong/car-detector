import utils
import multiprocessing
import pickle
import os

datasetPath = "../data/extracted"
outPath = "../visualisation/preprocessed-plys"

class Viz(multiprocessing.Process):
    def __init__(self, path, folder, filename):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.folder = folder
        self.filename = filename[:-14]

    def run(self):

        print("Process spawned for file %s" % (os.path.join(self.path, self.folder, self.filename)), flush=True)

        #get annotation data
        data = []
        with open(os.path.join(self.path, self.folder, self.filename + ".bag.data.pickle"), "rb") as f:
            data = pickle.load(f)
        annotations = data["extrapolated"]

        data = []
        with open(os.path.join(self.path, self.folder, self.filename + ".bag.3d.pickle"), "rb") as f:
            data = pickle.load(f)

        os.makedirs(os.path.join(outPath, self.folder), exist_ok=True)
        for frameIdx in range(len(data["pointclouds"])):
            outfilename = os.path.join(outPath, self.folder, "%s-%i.ply" % (self.filename, frameIdx))
            utils.drawPLY(outfilename, data["pointclouds"][frameIdx])

if __name__ == "__main__":

    os.makedirs(outPath, exist_ok=True)
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".3d.pickle" in filename:
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

