'''
把给定的结果进行坐标变换
zju a
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
    estpath = os.path.join('result/zju',seqname+'.csv')
    traj_est = np.loadtxt(estpath, delimiter=' ') # t(s) x y z i j k w camera坐标系
    print('load est pose {}'.format(traj_est.shape[0]))
    time_est = traj_est[:, 0].reshape(-1, 1)
    qt_est = traj_est[:,1:]
    rawgtpath = os.path.join(args.datapath,'groundtruth/euroc_gt.csv')
    rawgt = np.loadtxt(rawgtpath, delimiter=',', comments='#')[:,:8] # t(ns) x y z w i j k imu坐标系
    print('load raw gt pose {}'.format(rawgt.shape[0]))
    # 把rawgt 也保存
    rawsave = os.path.join('data/zjugt', seqname+'.txt')
    np.savetxt(rawsave,rawgt,delimiter=',')
    # 读取相机到imu的变换 TBcam
    sensor_path = os.path.join(args.datapath,'camera/sensor.yaml')
    import yaml
    with open(sensor_path, 'r') as f:
        temp = yaml.safe_load(f.read())
        q_bc = np.array(temp['extrinsic']['q']).reshape(4) # i j k w
        t_bc = np.array(temp['extrinsic']['p']).reshape(3)
        pos_quat = [t_bc[0],t_bc[1],t_bc[2],q_bc[0],q_bc[1],q_bc[2],q_bc[3]]
        # 转矩阵
        T_BC = pos_quat2SE_matrice(pos_quat)
        
    print('load T_BC: \n', T_BC)
    
    imuqt_est = transformref(qt_est, T_BC) # x y z w x y z 也改为输出xyz ijkw
    # 再和时间拼接起来
    imu_est = np.hstack((time_est, imuqt_est))
    estsave = os.path.join('result/zju',seqname+'-simu.csv') #-mono
    np.savetxt(estsave,imu_est,delimiter=' ')

    
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


