import os
import h5py
import numpy as np


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    labels = f['labels'][:]
    return data, labels


def save_h5(filename, data, labels):
    f = h5py.File(filename, 'w')
    lab = f.create_dataset("labels", data=labels)
    dat = f.create_dataset("data", data=data)
    f.close()


path = "bags/janota_regenerated/"
bags = []
for folders in os.walk(path):
    if folders[0] == path:
        bags = folders[2]
        break

data = np.empty((0, 1526, 3))
labels = np.empty((0, 1526))

for i in range(len(bags)):
    tmp_data, tmp_labels = load_h5(path + bags[i])
    data = np.vstack((data, tmp_data))
    labels = np.vstack((labels, tmp_labels))

print(data.shape, labels.shape)
save_h5("bags/trn_dataset.h5", data, labels)
