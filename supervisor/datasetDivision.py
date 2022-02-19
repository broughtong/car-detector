import os

testingSplit = 10 # 1bag in 10

datasetPath = "../annotations/maskrcnn"

os.makedirs(os.path.join(datasetPath, "training", "imgs"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "training", "annotations"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "testing", "imgs"), exist_ok=True)
os.makedirs(os.path.join(datasetPath, "testing", "annotations"), exist_ok=True)

#get list of rosbags
bags = []
for files in os.walk(os.path.join(datasetPath, "all", "imgs")):
    filenames = sorted(files[2]) #replicate split
    for fn in filenames:
        bag = fn.split(".pickle")[0]
        if bag not in bags:
            bags.append(bag)

evalBags = []
trainingBags = []
for bagIdx in range(len(bags)):
    if bagIdx % testingSplit == 0:
        #testing bag
        evalBags.append(bags[bagIdx])
    else:
        #training bag
        trainingBags.append(bags[bagIdx])

print("The following bags will be used for testing: ")
print(evalBags)
print("The following bags will be used for training: ")
print(trainingBags)
print("Creating symlinks...")

for files in os.walk(os.path.join(datasetPath, "all", "imgs")):
    for filename in files[2]:
        bag = filename.split(".pickle")[0]
        if bag in evalBags:
            os.symlink(os.path.join("../../", "all", "imgs", filename), os.path.join(datasetPath, "testing", "imgs", filename))
        else:
            os.symlink(os.path.join("../../", "all", "imgs", filename), os.path.join(datasetPath, "training", "imgs", filename))
for files in os.walk(os.path.join(datasetPath, "all", "annotations")):
    for filename in files[2]:
        bag = filename.split(".pickle")[0]
        if bag in evalBags:
            os.symlink(os.path.join("../../", "all", "annotations", filename), os.path.join(datasetPath, "testing", "annotations", filename))
        else:
            os.symlink(os.path.join("../../", "all", "annotations", filename), os.path.join(datasetPath, "training", "annotations", filename))

