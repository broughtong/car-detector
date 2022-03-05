import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


datasets_path = Path("/datafast/janota/")


def convert_bag(bag):
    with h5py.File(bag, 'r') as f:
        points = np.array(f["data"])
        labels = np.array(f["labels"])
      
    np.save(bag.with_suffix(".h5.points.npy"), points, allow_pickle=False)
    np.save(bag.with_suffix(".h5.labels.npy"), labels, allow_pickle=False)



with ThreadPoolExecutor(20) as ex:
    fs = []
    for dataset in datasets_path.glob("*"):
        for bag in dataset.glob("*.h5"):
            fs.append(ex.submit(convert_bag, bag))
    for f in fs:
        f.result()

