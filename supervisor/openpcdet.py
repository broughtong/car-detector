import os
import shutil

def run(datasetPath):

    try:
        shutil.rmtree(os.path.join(datasetPath, "openpcdet"))
    except:
        pass
    os.makedirs(os.path.join(datasetPath, "openpcdet"), exist_ok=True)
    os.makedirs(os.path.join(datasetPath, "openpcdet", "ImageSets"), exist_ok=True)
    os.makedirs(os.path.join(datasetPath, "openpcdet", "points"), exist_ok=True)
    os.makedirs(os.path.join(datasetPath, "openpcdet", "labels"), exist_ok=True)

    with open(os.path.join(datasetPath, "openpcdet", "ImageSets", "train.txt"), "w") as ftrain:
        with open(os.path.join(datasetPath, "openpcdet", "ImageSets", "val.txt"), "w") as fval:
            
            for files in os.walk(os.path.join(datasetPath, "training", "cloud")):
                for filename in files[2]:
                    src = "../../training/cloud/" + filename
                    dst = os.path.join(datasetPath, "openpcdet", "points", filename)
                    os.symlink(src, dst)
                    src = "../../training/annotations/" + filename[:-4] + ".txt"
                    dst = os.path.join(datasetPath, "openpcdet", "labels", filename[:-4] + ".txt")
                    os.symlink(src, dst)
                    ftrain.write(filename[:-4] + "\n")

            for files in os.walk(os.path.join(datasetPath, "testing", "cloud")):
                for filename in files[2]:
                    src = "../../testing/cloud/" + filename
                    dst = os.path.join(datasetPath, "openpcdet", "points", filename)
                    os.symlink(src, dst)
                    src = "../../testing/annotations/" + filename[:-4] + ".txt"
                    dst = os.path.join(datasetPath, "openpcdet", "labels", filename[:-4] + ".txt")
                    os.symlink(src, dst)
                    fval.write(filename[:-4] + "\n")

run("../annotations/scans/pointcloud-npy")
