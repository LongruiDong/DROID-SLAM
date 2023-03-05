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



def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

# from tartanair
def image_stream(datapath, image_size=[480, 640], intrinsics_vec=[493.0167, 491.55953, 317.97856, 242.392], stereo=False):
    """ image generator """
    # 内参格式 fx fy cx cy
    intrinsics_vec=[493.0167, 491.55953, 317.97856, 242.392] # for A
    # read all png images in folder 测试不需要深度
    ht0, wd0 = [480, 640] # 原始数据大小 之后对长款0.8缩放 datasets/zju/A0/camera/images/771812250517066.png
    images_left = sorted(glob.glob(os.path.join(datapath, 'camera/images/*.png')))
    # images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    data = []
    for t in range(len(images_left)):
        # images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ] #list (384,512,3)
        try:
            images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ] #list (384,512,3)
        except: # 数据损坏
            print('bad png at {}, continue...'.format(images_left.split('/')[-1][:-4]))
            continue
        # if stereo:
        #     images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]
        # array 转 tensor 堆叠 维度换位
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2) #array (2,3,384,512)
        intrinsics = torch.as_tensor(intrinsics_vec) # .8 * 
        # 输入数据的元组 List t只是伪时间
        data.append((t, images, intrinsics))

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024) # 512 | 1024
    parser.add_argument("--image_size", default=[480, 640])
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
    # , stereo=args.stereo , stride=2
    for (t, image, intrinsics) in tqdm(image_stream(args.datapath)):
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath))

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

    images_list = sorted(glob.glob(os.path.join(args.datapath, 'camera/images/*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]
    seqname = args.datapath.split('/')[-1]
    tstamparr = np.divide(np.array(tstamps), 1e9).reshape(-1, 1) # s
    est_euroc = np.hstack((tstamparr, traj_est))
    # 保存估计位子 euroc format  其实还是tum(i j k w) s    空格
    np.savetxt(os.path.join('result/zju',seqname+'.csv'),est_euroc,delimiter=' ')
    
    traj_est0 = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9))

    traj_est_se = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds
    
    traj_est_sim = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3], # 这个比例？ 1.10 * 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.divide(np.array(tstamps), 1e9)) # np.divide(mat[:, 0], 1e9)  # n x 1  -  nanoseconds to seconds

    # traj_ref = file_interface.read_tum_trajectory_file(args.gt) # 单位还是ns
    traj_ref = file_interface.read_euroc_csv_trajectory(args.gt) # time 单位s

    traj_ref, traj_est_sim = sync.associate_trajectories(traj_ref, traj_est_sim)

    result = main_ape.ape(traj_ref, traj_est_sim, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print('SIM3-----\n',result)

    # 没有尺度对齐的结果
    traj_ref, traj_est_se = sync.associate_trajectories(traj_ref, traj_est_se)
    resultse3 = main_ape.ape(traj_ref, traj_est_se, est_name='trajse',  #-a
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    print('SE3-----\n',resultse3)

    # 尺度因子
    r_a, t_a, s_KF = geometry.umeyama_alignment(traj_est0.positions_xyz.T, traj_ref.positions_xyz.T, True)
    print('scale-----\n',s_KF)

