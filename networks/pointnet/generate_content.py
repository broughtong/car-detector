import os
import os.path
import numpy as np
import sys
import h5py


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    labels = f['labels'][:]
    f.close()
    return data, labels


path = "../../../janota/lanoising-ts4-scans"
f_out = open("trn_list.txt", "w")

dirs = os.listdir(path)
print(dirs)
for j in range(len(dirs)):

    file = dirs[j]

    if ".txt" in file:
        continue

    if j > len(dirs)-3:
        f_out.close()
        f_out = open("val_list.txt", "w")

    data, labels = load_h5(path+"/"+file)

    print(len(data))

    for i in range(len(data)):
        f_out.write(file + "_" + str(i)+"\n")

f_out.close()
