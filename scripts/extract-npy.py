import numpy as np
import pickle

f = "../data/temporal/3/2022-06-21-20-31-21.bag.3d.pickle"
idx = 530
out = "../a.npy"

data = None
with open(f, "rb") as f:
    data = pickle.load(f)

data = data["pointclouds"][idx]
data = np.delete(data, 4, 1)
data[:, 3] = 0

np.save(out, data)
