import numpy as np

def combineScans(arrOfScans):

    scans = []
    for key in arrOfScans.keys():
        arrOfScans[key] = np.array(arrOfScans[key])
        arrOfScans[key] = arrOfScans[key].reshape([arrOfScans[key].shape[0], 4])
        scans.append(arrOfScans[key])

    return np.concatenate(scans)

