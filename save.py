import numpy as np
import os

datapath = "datasets/tcsvt/gt_result2022"
outquatdir = "result1"
camera = "android"
Benchmark_Scenes = ["atrium", "corridor", "exhibition-hall", "indoor-office-room", "outdoor-office-park", "stairs", "whole-floor"]
for scene in Benchmark_Scenes:
    Benchmark_scene_path = os.path.join(datapath, scene)
    Scene_Sequences = os.listdir(Benchmark_scene_path)
    for squence in Scene_Sequences:
        seqname = scene + "-" + squence
        path = os.path.join(outquatdir,seqname+'-mono.csv')
        if not os.path.exists(path):
            continue
        traj_ref0 = np.loadtxt(path, delimiter=' ') #t wxyz
        traj_ref0[:, 0] = traj_ref0[:, 0]/1000000000.0
        print(traj_ref0)
        np.savetxt(os.path.join(outquatdir,seqname+'-mono1.csv'),traj_ref0,delimiter=' ')
        
