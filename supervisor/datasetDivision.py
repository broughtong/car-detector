import os
import shutil

def divide(testingSplit, datasetPath, gtPath):

    subfolders = []
    for files in os.walk(datasetPath):
        if files[0][-4:] == "/all":
            subfolders = files[1]

    for subfolder in subfolders:
        try:
            shutil.rmtree(os.path.join(datasetPath, "training", subfolder))
            shutil.rmtree(os.path.join(datasetPath, "testing", subfolder))
            shutil.rmtree(os.path.join(datasetPath, "evaluation", subfolder))
        except:
            pass
        os.makedirs(os.path.join(datasetPath, "training", subfolder), exist_ok=True)
        os.makedirs(os.path.join(datasetPath, "testing", subfolder), exist_ok=True)
        os.makedirs(os.path.join(datasetPath, "evaluation", subfolder), exist_ok=True)

    #get list of rosbags with gt
    evalBags = []
    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag"
                evalBags.append(fn)
    evalBags = list(set(evalBags))

    #get list of rosbags
    bags = []
    for files in os.walk(os.path.join(datasetPath, "all", subfolders[0])):
        filenames = sorted(files[2]) #replicate split
        for fn in filenames:
            bag = fn.split(".")[0]
            if bag not in evalBags:
                if bag not in bags:
                    bags.append(bag)

    testingBags = []
    trainingBags = []
    for bagIdx in range(len(bags)):
        if bagIdx % testingSplit == 0:
            testingBags.append(bags[bagIdx])
        else:
            trainingBags.append(bags[bagIdx])

    print("The following bags will be used for training: ")
    print(trainingBags)
    print("The following bags will be used for testing: ")
    print(testingBags)
    print("The following bags will be used for evaluation: ")
    print(evalBags)
    print("Creating symlinks...")

    dryRun = False
    for subfolder in subfolders:
        for files in os.walk(os.path.join(datasetPath, "all", subfolder)):
            for filename in files[2]:
                bag = filename.split(".")[0]
                if bag in trainingBags:
                    src = os.path.join("../../", "all", subfolder, filename)
                    dst = os.path.join(datasetPath, "training", subfolder, filename)
                    if dryRun == False:
                        os.symlink(src, dst)
                elif bag in testingBags:
                    src = os.path.join("../../", "all", subfolder, filename)
                    dst = os.path.join(datasetPath, "testing", subfolder, filename)
                    if dryRun == False:
                        os.symlink(src, dst)
                elif bag in evalBags:
                    src = os.path.join("../../", "all", subfolder, filename)
                    dst = os.path.join(datasetPath, "evaluation", subfolder, filename)
                    if dryRun == False:
                        os.symlink(src, dst)

if __name__ == "__main__":

    #divide(testingSplit=6, datasetPath="../annotations/lanoising/mask", gtPath="../data/gt")
    #divide(testingSplit=6, datasetPath="../annotations/scans/mask", gtPath="../data/gt")
    #divide(testingSplit=8, datasetPath="../annotations/lanoising/mask", gtPath="../data/gt")
    divide(testingSplit=8, datasetPath="../annotations/scans/maskrcnn", gtPath="../data/gt")
    divide(testingSplit=8, datasetPath="../annotations/scans/pointcloud-ply", gtPath="../data/gt")
    divide(testingSplit=8, datasetPath="../annotations/scans/pointcloud-npy", gtPath="../data/gt")
    divide(testingSplit=8, datasetPath="../annotations/scans/pointcloud-bin", gtPath="../data/gt")
