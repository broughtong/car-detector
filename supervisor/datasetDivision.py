import os

testingSplit = 10 # 1bag in 10

datasetPath = "../annotations/maskrcnn"
gtPath = "../data/gt"

os.makedirs(os.path.join(datasetPath, "training", "imgs"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "training", "annotations"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "testing", "imgs"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "testing", "annotations"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "evaluation", "imgs"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "evaluation", "annotations"), exist_ok=True)

#get list of rosbags with gt
evalBags = []
for files in os.walk(gtPath):
    for fn in files[2]:
        fn = fn.split("-")[:-1]
        fn = "-".join(fn)
        fn += ".bag"
        evalBags.append(fn)
evalBags = list(set(evalBags))

#get list of rosbags
bags = []
for files in os.walk(os.path.join(datasetPath, "all", "imgs")):
    filenames = sorted(files[2]) #replicate split
    for fn in filenames:
        bag = fn.split(".pickle")[0]
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

for files in os.walk(os.path.join(datasetPath, "all", "imgs")):
    for filename in files[2]:
        bag = filename.split(".pickle")[0]
        if bag in trainingBags:
            src = os.path.join("../../", "all", "imgs", filename)
            dst = os.path.join(datasetPath, "training", "imgs", filename)
            os.symlink(src, dst)
        elif bag in testingBags:
            src = os.path.join("../../", "all", "imgs", filename)
            dst = os.path.join(datasetPath, "testing", "imgs", filename)
            os.symlink(src, dst)
        elif bag in evalBags:
            src = os.path.join("../../", "all", "imgs", filename)
            dst = os.path.join(datasetPath, "evaluation", "imgs", filename)
            os.symlink(src, dst)

for files in os.walk(os.path.join(datasetPath, "all", "annotations")):
    for filename in files[2]:
        bag = filename.split(".pickle")[0]
        if bag in trainingBags:
            src = os.path.join("../../", "all", "annotations", filename)
            dst = os.path.join(datasetPath, "training", "annotations", filename)
            os.symlink(src, dst)
        elif bag in testingBags:
            src = os.path.join("../../", "all", "annotations", filename)
            dst = os.path.join(datasetPath, "testing", "annotations", filename)
            os.symlink(src, dst)
        elif bag in evalBags:
            src = os.path.join("../../", "all", "annotations", filename)
            dst = os.path.join(datasetPath, "evaluation", "annotations", filename)
            os.symlink(src, dst)
