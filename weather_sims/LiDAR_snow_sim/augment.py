#! /usr/bin/env python

import os
import multiprocessing
import pickle
import numpy as np

from lib.LiDAR_fog_sim.fog_simulation import ParameterSet, simulate_fog
from tools.snowfall.simulation import augment

datasetPath = "../../data/temporal"
outputPath = "../../data/snow"

def callback(frame):

    #wants nx5 np array (xyz intensity, channel)
    frame = np.delete(frame, 8, 1)
    frame = np.delete(frame, 7, 1)
    frame = np.delete(frame, 5, 1)
    frame = np.delete(frame, 4, 1)

    #snow
    """
    filename = f'{self.mode}_{rain_rate}_{occupancy}'
    noise_floor = 0.7
    stats, pc = augment(pc=frame, 
        only_camera_fov=False,
        particle_file_prefix=filename, 
        noise_floor=noise_floor,
        beam_divergence=0.18,
        shuffle=True, 
        show_progressbar=True)
    return pc
    """


    #fog
    parameters = {'n': 500, 'n_min': 100, 'n_max': 1000, 'r_range': 69.90491, 'r_range_min': 50, 'r_range_max': 250, 'alpha': 0.06, 'alpha_min': 0.003, 'alpha_max': 0.5, 'alpha_scale': 1000, 'mor': 49.92887122589985, 'beta': 0.000921310633919122, 'beta_min': 0.000460655316959561, 'beta_max': 0.001842621267838244, 'beta_scale': 49928.87122589985, 'p_0': 80, 'p_0_min': 60, 'p_0_max': 100, 'tau_h': 2e-08, 'tau_h_min': 5e-09, 'tau_h_max': 8e-08, 'tau_h_scale': 1000000000.0, 'e_p': 1.6e-06, 'a_r': 0.25, 'a_r_min': 0.01, 'a_r_max': 0.1, 'a_r_scale': 1000, 'l_r': 0.05, 'l_r_min': 0.01, 'l_r_max': 0.1, 'l_r_scale': 100, 'c_a': 1873702.8625, 'linear_xsi': True, 'D': 0.1, 'ROH_T': 0.01, 'ROH_R': 0.01, 'GAMMA_T_DEG': 2, 'GAMMA_R_DEG': 3.5, 'GAMMA_T': 0.03490658503988659, 'GAMMA_R': 0.061086523819801536, 'r_1': 0.9, 'r_1_min': 0, 'r_1_max': 10, 'r_1_scale': 10, 'r_2': 1.0, 'r_2_min': 0, 'r_2_max': 10, 'r_2_scale': 10, 'r_0': 30, 'r_0_min': 1, 'r_0_max': 200, 'gamma': 1e-06, 'gamma_min': 1e-07, 'gamma_max': 1e-05, 'gamma_scale': 10000000, 'beta_0': 3.1830988618379064e-07}
    p = ParameterSet()
    for key, value in parameters.items():
        p.key = value
    noise = 10
    gain = True
    noise_variant = "v4"
    pc, fog_pc, stats = simulate_fog(p, frame, noise, gain, noise_variant)

    return pc

class Augment(multiprocessing.Process):
    def __init__(self, path, folder, filename):
        multiprocessing.Process.__init__(self)

        self.path = path
        self.folder = folder
        self.filename = filename

    def run(self):

        path = self.path
        folder = self.folder
        filename = self.filename
        #print("Opening %s %s %s" % (path, folder, filename))
        data = []
        with open(os.path.join(path, folder, filename), "rb") as f:
            data = pickle.load(f)

        newScans = []
        for frameIdx in range(len(data["pointclouds"])):
            print("Frame %i/%i" % (frameIdx, len(data["pointclouds"])))
            newFrame = callback(data["pointclouds"][frameIdx])
            newScans.append(newFrame)
        data["snow"] = newScans

        print("Saving File %s %s %s" % (path, folder, filename))
        os.makedirs(os.path.join(outputPath, folder), exist_ok=True)
        with open(os.path.join(outputPath, folder, filename), "wb") as f:
            pickle.dump(data, f)

if __name__ == '__main__':
    
    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if ".3d.pickle" in filename:
                path = datasetPath
                folder = files[0][len(path)+1:]
                jobs.append(Augment(path, folder, filename))

    print("Spawned %i processes" % (len(jobs)), flush = True)
    cpuCores = 15
    limit = cpuCores
    batch = cpuCores
    for i in range(len(jobs)):
        if i < limit:
            jobs[i].start()
        else:
            for j in range(limit):
                jobs[j].join()
                limit += batch
                jobs[i].start()

