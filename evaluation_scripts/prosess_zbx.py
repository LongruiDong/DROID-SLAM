'''
把给定的结果进行坐标变换
'''
import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')
# from tqdm import tqdm
import numpy as np
# import torch
# import lietorch
# import cv2
import os
import glob 
import time
import argparse

# from torch.multiprocessing import Process
# from droid import Droid

# import torch.nn.functional as F

global R_l

R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3) # 矫正矩阵 rect 和cam不同

from evaluation.transformation import pos_quat2SE_matrice, SE2pos_quat

def transformref(traj_raw, Tr):
    # Tr : T_Brect (rect)相机到imu
    Trinv = np.linalg.inv(Tr) # T_rectB
    flen  = traj_raw.shape[0]
    traj_new = []
    for i in range(flen):
        # qwxyz = traj_raw[i,3:] # wxyz
        # qxyzw = [qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]] # ! -> xyzw
        qxyzw = traj_raw[i,3:] # 本来就是xyzw
        t = traj_raw[i,:3]
        pos_quat = [t[0],t[1],t[2],qxyzw[0],qxyzw[1],qxyzw[2],qxyzw[3]]
        SEarr = pos_quat2SE_matrice(pos_quat) # imu gt T
        # newSE = (Tr.dot(SEarr)).dot(Trinv) # 到imu下
        newSE = SEarr.dot(Trinv) # haomin指正 Twrect TrectB
        newpos_quat = SE2pos_quat(newSE) # xyz Q: X Y Z W 注意这里的四元数输出
        # nquat = [newpos_quat[0],newpos_quat[1],newpos_quat[2],
        # newpos_quat[6],newpos_quat[3],newpos_quat[4],newpos_quat[5]] # 变回w x y z
        nquat = newpos_quat # 还按ijkw输出
        traj_new.append(nquat)
    traj_out=np.array(traj_new)

    return traj_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    # parser.add_argument("--gt", help="path to gt file")
    
    args = parser.parse_args()

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    seqname = args.datapath.split('/')[-1]
    estpath = os.path.join('result/euroc',seqname+'-s-mono.csv') # -mono
    traj_est = np.loadtxt(estpath, delimiter=' ') # t(s) x y z w i j k camera坐标系
    print('load est pose {}'.format(traj_est.shape[0]))
    time_est = traj_est[:, 0].reshape(-1, 1)
    qt_est = traj_est[:,1:]
    rawgtpath = os.path.join(args.datapath,'mav0/state_groundtruth_estimate0/data.csv')
    rawgt = np.loadtxt(rawgtpath, delimiter=',', comments='#')[:,:8] # t(ns) x y z w i j k imu坐标系
    print('load raw gt pose {}'.format(rawgt.shape[0]))
    # 读取相机到imu的变换 TBcam
    sensor_path = os.path.join(args.datapath,'mav0/cam0/sensor.yaml')
    import yaml
    with open(sensor_path, 'r') as f:
        temp = yaml.safe_load(f.read())
        T_BC = np.array(temp['T_BS']['data']).reshape(4,4)
    print('load T_BC: \n', T_BC)
    T_l43 = np.vstack((R_l, np.array([0,0,0])))
    T_l = np.hstack((T_l43, np.array([[0],[0],[0],[1]]))) #Trect_l
    T_l_inv = np.linalg.inv(T_l) #Tl_rect
    global T_Brect
    T_Brect = T_BC.dot(T_l_inv) # 矫正相机系 到 imu(gt)
    # 使用
    print('load T_Brect: \n', T_Brect) # 输入的 是 i j k w
    imuqt_est = transformref(qt_est, T_Brect) # x y z w x y z 也改为输出xyz ijkw
    # 再和时间拼接起来
    imu_est = np.hstack((time_est, imuqt_est))
    estsave = os.path.join('result/euroc',seqname+'-srectimu2-mono.csv') #-mono
    np.savetxt(estsave,imu_est,delimiter=' ')

    # 把rawgt 也保存
    rawsave = os.path.join('data/rawgt', seqname+'.txt')
    np.savetxt(rawsave,rawgt,delimiter=',')
    # ### run evaluation ###

    # import evo
    # from evo.core.trajectory import PoseTrajectory3D
    # from evo.tools import file_interface
    # from evo.core import sync
    # import evo.main_ape as main_ape
    # from evo.core.metrics import PoseRelation
    # import evo.core.geometry as geometry
    # # from thirdparty.tartanair_tools.evaluation.tartanair_evaluator import TartanAirEvaluator
    # # from evaluation.tartanair_evaluator import TartanAirEvaluator #对定位结果评估的class

    # images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    # tstamps0 = [float(x.split('/')[-1][:-4]) for x in images_list]
    # print('all data image: {}'.format(len(tstamps0))) # 图像总数

    # tstamps = [float(fname) for (_, _, _, fname) in input_data] # ns 有效输入 才和实际输入对的上
    # # for (_, _, _, fname) in input_data:

    
    # tstamparr = np.divide(np.array(tstamps), 1e9).reshape(-1, 1) # s
    # est_euroc = np.hstack((tstamparr, traj_est))
    # # 保存估计位子 euroc format  其实还是tum s    空格
    # np.savetxt(os.path.join('result/euroc',seqname+'-s.csv'),est_euroc,delimiter=' ')
    # traj_est0 = PoseTrajectory3D(
    #     positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
    #     orientations_quat_wxyz=traj_est[:,3:],
    #     timestamps=np.divide(np.array(tstamps), 1e9))

    # traj_est_se = PoseTrajectory3D(
    #     positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
    #     orientations_quat_wxyz=traj_est[:,3:],
    #     timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds
    
    # traj_est_sim = PoseTrajectory3D(
    #     positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
    #     orientations_quat_wxyz=traj_est[:,3:],
    #     timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds

    # # traj_ref = file_interface.read_tum_trajectory_file(args.gt) # 单位还是ns
    # traj_ref = file_interface.read_euroc0_csv_trajectory(args.gt) # time 单位s
    # # 用numpy 读取 并保存真正euroc格式的gt 主要是分隔符
    # traj_ref0 = np.loadtxt(args.gt, delimiter=' ') #t wxyz
    
    # np.savetxt(os.path.join('data/eurocgt',seqname+'.txt'),traj_ref0,delimiter=',')
    # traj_ref, traj_est_sim = sync.associate_trajectories(traj_ref, traj_est_sim)
    # # 记录有多少个实际参与评估
    # num_valid = traj_ref.positions_xyz.shape[0]
    # result = main_ape.ape(traj_ref, traj_est_sim, est_name='traj', 
    #     pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    # print('valid frames pair: {}'.format(num_valid))
    # print('SIM3-----\n',result)

    # # 没有尺度对齐的结果
    # traj_ref, traj_est_se = sync.associate_trajectories(traj_ref, traj_est_se)
    # resultse3 = main_ape.ape(traj_ref, traj_est_se, est_name='trajse',  #-a
    #     pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    # print('SE3-----\n',resultse3)

    # traj_ref, traj_est0 = sync.associate_trajectories(traj_ref, traj_est0)
    # # 尺度因子
    # r_a, t_a, s_KF = geometry.umeyama_alignment(traj_est0.positions_xyz.T, traj_ref.positions_xyz.T, True)
    # print('scale-----\n',s_KF)


