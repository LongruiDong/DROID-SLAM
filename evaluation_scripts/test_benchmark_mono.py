import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import yaml

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F



import yaml

import evaluation.transformation as tf


def image_stream(folder_path, intrinsics_vec, distortion_vec,image_size=[384, 512], stereo=False, stride=1):
    """ image generator """
    ht0, wd0 = [480, 640]
    K_l = np.array([intrinsics_vec[0], 0.0, intrinsics_vec[2], 0.0, intrinsics_vec[1], intrinsics_vec[3], 0.0, 0.0, 1.0]).reshape(3,3)
    # read all png images in folder
    images = sorted(glob.glob(os.path.join(folder_path, 'images/*.png')) + glob.glob(os.path.join(folder_path, 'images/*.jpg')))
    d_l = np.array(distortion_vec)
    for t, img in enumerate(images):
        image = cv2.imread(img)
        image = cv2.undistort(image, K_l, d_l)
        cv2.imwrite("t.jpg", image)
        image = cv2.resize(image, (image_size[1], image_size[0]))
        image = torch.from_numpy(image).permute(2,0,1)

        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0
        yield t, image[None], intrinsics


def transformref(traj_raw, Tr):
    # Tr : T_Brect (rect)相机到imu
    Trinv = np.linalg.inv(Tr) # T_rectB
    flen  = traj_raw.shape[0]
    traj_new = []
    for i in range(flen):
        qwxyz = traj_raw[i,3:] # wxyz
        qxyzw = [qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]] # ! -> xyzw
        t = traj_raw[i,:3]
        pos_quat = [t[0],t[1],t[2],qxyzw[0],qxyzw[1],qxyzw[2],qxyzw[3]]
        SEarr = tf.pos_quat2SE_matrice(pos_quat) # imu gt T
        newSE = (Tr.dot(SEarr)).dot(Trinv) # 到imu下
        newpos_quat = tf.SE2pos_quat(newSE) # xyz Q: X Y Z W 注意这里的四元数输出
        nquat = [newpos_quat[0],newpos_quat[1],newpos_quat[2],
        newpos_quat[6],newpos_quat[3],newpos_quat[4],newpos_quat[5]] # 变回w x y z
        traj_new.append(nquat)
    traj_out=np.array(traj_new)

    return traj_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True, help="path to dataset")
    parser.add_argument("--gt", required=True, help="path to gt file")
    parser.add_argument("--seq_name", required=True, help="seq_name", default = "android/atrium/A0")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)
    #Benchmark_Scenes = ["atrium", "corridor", "exhibition-hall", "indoor-office-room", "outdoor-office-park", "stair", "whole-floor"]
    #for scene in Benchmark_Scenes:
        #Benchmark_scene_path = os.path.join(args.datapath, scene)
        #Scene_Sequences = os.listdir(Benchmark_scene_path)
        #for squence in Scene_Sequences:
    torch.cuda.empty_cache()
    droid = Droid(args)
    folder_path = os.path.join(args.datapath, "camera")
    skip_lines = 2
    with open(os.path.join(folder_path, "sensor.yaml")) as f:
        for i in range(skip_lines):
            f.readline()
        y = yaml.safe_load(f)
        intrinsics_vec = y["intrinsic"]["camera"]
        distortion_vec = y["intrinsic"]["distortion"]
        q_imu0 = y["extrinsic"]["q"]
        q_imu = [q_imu0[1], q_imu0[2], q_imu0[3], q_imu0[0]]
        t_imu = y["extrinsic"]["p"]

    for (tstamp, image, intrinsics) in tqdm(image_stream(folder_path, intrinsics_vec, distortion_vec)):
        droid.track(tstamp, image, intrinsics=intrinsics)
    traj_est = droid.terminate(image_stream(folder_path, intrinsics_vec, distortion_vec))
    Tr = tf.pos_quat2SE_matrice(t_imu + q_imu)
    traj_est = transformref(traj_est, Tr) # camera-> imu

    # evaluation
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    images_list = sorted(glob.glob(os.path.join(folder_path, 'images/*.png')) + glob.glob(os.path.join(folder_path, 'images/*.jpg')))
    tstamps = [float(x.split('/')[-1][:-4])/1000000000.0 for x in images_list] # ns->s !
    traj_est0 = traj_est #arr w x y z
    tstamparr = np.array(tstamps).reshape(-1, 1)#*1000000000.0 #转列向量  ns
    traj_est_time = np.hstack((tstamparr, traj_est0))
    # 保存结果
    outquatdir = 'result1'
    folder = os.path.exists(outquatdir)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(outquatdir)
    seqname = args.datapath.split('/')[-3]+"-"+args.datapath.split('/')[-2]
    np.savetxt(os.path.join(outquatdir,seqname+'-mono_ios.csv'),traj_est_time,delimiter=' ') # 还有time也保存
    traj_ref0 = np.loadtxt(args.gt, delimiter=',')[:, :8] #t wxyz
    np.savetxt(os.path.join(outquatdir,seqname+'-monogt_ios.csv'),traj_ref0,delimiter=',')


    traj_est = PoseTrajectory3D(
        positions_xyz=1.10 * traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps))


    evaluator = TartanAirEvaluator()

    traj_ref = file_interface.read_euroc_csv_trajectory(args.gt)
    traj_ref0 = np.loadtxt(args.gt, delimiter=',')[:, [1, 2, 3, 5, 6, 7, 4]] # -> xyz w
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.02)


    traj_ref_xyz = traj_ref.positions_xyz #(N,3)
    traj_ref_xyzw = traj_ref.orientations_quat_wxyz[:,[1,2,3,0]] # wxyz-> xyzw
    traj_ref_1 = np.hstack((traj_ref_xyz, traj_ref_xyzw))
    traj_est_xyz = traj_est.positions_xyz
    traj_est_xyzw = traj_est.orientations_quat_wxyz[:,[1,2,3,0]] # wxyz-> xyzw
    traj_est_1 = np.hstack((traj_est_xyz, traj_est_xyzw))
    # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
    results0 = evaluator.evaluate_one_trajectory(traj_ref_1, traj_est_1, #也会画轨迹
                scale=True, title=seqname+'-mono_ios')
    print(results0)
    ate_list = [] # save ATE rmse
    ate_list.append(results0["ate_score"])

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)
            





