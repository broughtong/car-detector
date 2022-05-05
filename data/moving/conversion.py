import pickle

data = []

with open("images_info.csv", "r") as f:
    data = f.read()

data = data.split("\n")
data = filter(None, data)

gt = {}

for line in data:
    tokens = line.split(",")
    frame = int(tokens[0].split("-")[-1].split(".")[0])
    x = int(tokens[1])
    y = int(tokens[2])

    annotation = []

    if x != -1:
        annotation.append([x, y, 1])

    gt[frame] = annotation

print(gt)
with open("2022-02-15-15-08-59.bag.gt", "wb") as f:
    pickle.dump(gt, f)
