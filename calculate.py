import numpy as np
import os
from evo.tools import file_interface
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation
import evo.main_ape as main_ape
import evo.core.geometry as geometry
gt_path = "/home/dlr/Project/DROID-SLAM/datasets/tcsvt/gt_result2022/atrium/A0/android/gba_pose.csv"
est_path = "/home/dlr/Project/DROID-SLAM/result1/atrium-A0-mono.csv"

def calculate(gt_path, est_path):
    traj_ref = file_interface.read_euroc_csv_trajectory(gt_path)
    traj_est = np.loadtxt(est_path, delimiter=' ')[:, 1:]
    tstamps = np.loadtxt(est_path, delimiter=' ')[:, 0]/1000000000.0
    traj_est = PoseTrajectory3D(
        positions_xyz=1.10 * traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.02)

    traj_imu_se = PoseTrajectory3D(
        positions_xyz=traj_est.positions_xyz,
        orientations_quat_wxyz=traj_est.orientations_quat_wxyz, # w x y z
        timestamps=traj_est.timestamps) #s
    traj_imu_sim = PoseTrajectory3D(
        positions_xyz=traj_est.positions_xyz,
        orientations_quat_wxyz=traj_est.orientations_quat_wxyz, # w x y z
        timestamps=traj_est.timestamps) #s 
    resultsim3 = main_ape.ape(traj_ref, traj_imu_sim, est_name='traj',  #-as
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    print('SIM3-----\n',resultsim3)
    
    r_a, t_a, s_KF = geometry.umeyama_alignment(traj_est.positions_xyz.T, traj_ref.positions_xyz.T, True)
    print('scale-----\n',s_KF)
    resultse3 = main_ape.ape(traj_ref, traj_imu_se, est_name='trajse',  #-a
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    print('SE3-----\n',resultse3)

if __name__ == "__main__":
    gt_root = "/home/dlr/Project/DROID-SLAM/datasets/tcsvt/gt_result2022/"
    Benchmark_Scenes = ["atrium", "corridor", "exhibition-hall", "indoor-office-room", "outdoor-office-park", "stairs", "whole-floor"]
    for scene in Benchmark_Scenes:
        Benchmark_scene_path = os.path.join(gt_root, scene)
        Scene_Sequences = os.listdir(Benchmark_scene_path)
        for squence in Scene_Sequences:
            print("*"*32)
            print(squence)
            gt_path = os.path.join(gt_root, scene, squence, "android/gba_pose.csv")
            est_path = "/home/dlr/Project/DROID-SLAM/result1/"+scene+"-"+squence+"-mono.csv"
            if not os.path.exists(est_path):
                continue
            calculate(gt_path, est_path)
    