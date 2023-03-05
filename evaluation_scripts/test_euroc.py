import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

global R_l

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
    """ image generator """

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3) 
    
    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
    
    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
         0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3,3)
    # 从这里也能看出 baseline 0.11m
    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])
        # print('read img: {}'.format(imgL.split('/')[-1][:-4]))
        try:
            images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        except: # euroc数据损坏
            print('bad png at {}, continue...'.format(imgL.split('/')[-1][:-4]))
            continue
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0
        # 对于euroc 多返回png filename
        yield stride*t, images, intrinsics, imgL.split('/')[-1][:-4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024) # 512 | 1024
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = Droid(args)
    time.sleep(5)
    # 对于euroc 多返回png filename
    for (t, image, intrinsics, _) in tqdm(image_stream(args.datapath, stereo=args.stereo, stride=2)):
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath, stride=1))
    print('filled frame: {}'.format(traj_est.shape[0])) # 填好的普通帧总数
    ### run evaluation ###

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    import evo.core.geometry as geometry
    # from thirdparty.tartanair_tools.evaluation.tartanair_evaluator import TartanAirEvaluator
    # from evaluation.tartanair_evaluator import TartanAirEvaluator #对定位结果评估的class

    images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    tstamps0 = [float(x.split('/')[-1][:-4]) for x in images_list]
    print('all data image: {}'.format(len(tstamps0))) # 图像总数
    input_data = tqdm(image_stream(args.datapath, stride=1))
    tstamps = [float(fname) for (_, _, _, fname) in input_data] # ns 有效输入 才和实际输入对的上
    # for (_, _, _, fname) in input_data:

    seqname = args.datapath.split('/')[-1]
    tstamparr = np.divide(np.array(tstamps), 1e9).reshape(-1, 1) # s
    est_euroc = np.hstack((tstamparr, traj_est))
    # 保存估计位子 euroc format  其实还是tum s    空格
    np.savetxt(os.path.join('result/euroc',seqname+'-s-mono.csv'),est_euroc,delimiter=' ')
    traj_est0 = PoseTrajectory3D(
        positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9))

    traj_est_se = PoseTrajectory3D(
        positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds
    
    traj_est_sim = PoseTrajectory3D(
        positions_xyz=1.10 *traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds

    # traj_ref = file_interface.read_tum_trajectory_file(args.gt) # 单位还是ns
    traj_ref = file_interface.read_euroc0_csv_trajectory(args.gt) # time 单位s
    # 用numpy 读取 并保存真正euroc格式的gt 主要是分隔符
    traj_ref0 = np.loadtxt(args.gt, delimiter=' ') #t wxyz
    
    # np.savetxt(os.path.join('data/eurocgt',seqname+'.txt'),traj_ref0,delimiter=',')
    traj_ref, traj_est_sim = sync.associate_trajectories(traj_ref, traj_est_sim)
    # 记录有多少个实际参与评估
    num_valid = traj_ref.positions_xyz.shape[0]
    result = main_ape.ape(traj_ref, traj_est_sim, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    print('valid frames pair: {}'.format(num_valid))
    print('SIM3-----\n',result)

    # 没有尺度对齐的结果
    traj_ref, traj_est_se = sync.associate_trajectories(traj_ref, traj_est_se)
    resultse3 = main_ape.ape(traj_ref, traj_est_se, est_name='trajse',  #-a
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    print('SE3-----\n',resultse3)

    traj_ref, traj_est0 = sync.associate_trajectories(traj_ref, traj_est0)
    # 尺度因子
    r_a, t_a, s_KF = geometry.umeyama_alignment(traj_est0.positions_xyz.T, traj_ref.positions_xyz.T, True)
    print('scale-----\n',s_KF)


