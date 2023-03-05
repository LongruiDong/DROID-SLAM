# -*- coding:utf8 -*-
"""
test scannet
"""
from copy import copy
import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')
import evaluation.transformation as tf

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse

from droid import Droid
import copy
"""
scannet color 968x1296
depth 480x640 (same with tartanair)
把color resize 到depth先 这时就可用intrinsic_depth
然后在用image_stream得到 384,512
"""

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[577.590698, 578.729797, 318.905426, 242.683609], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640] #以此作为哦输入 60，80
    images_left = sorted(glob.glob(os.path.join(datapath, 'color/*.jpg')))
    # images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    data = []
    for t in range(len(images_left)):
        # rawimgs = cv2.resize(cv2.imread(images_left[t]), (wd0, ht0)) # to depth img size (image_size[1], image_size[0])
        images = [ cv2.resize(cv2.imread(images_left[t]), (wd0, ht0)) ] # to 384,512

        # if False:
        #     images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = 1.0 * torch.as_tensor(intrinsics_vec) # 640，480 -》 384,512 .8 

        data.append((t, images, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/mnt/lustre/share_data/scannet/public_datalist_185/scans")
    parser.add_argument("--sceneid", type=str) # scans中某个序列
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000) #对于kitti 大一些 1300 1800 2000
    parser.add_argument("--image_size", default=[480,640]) # 384,512 480,640
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--stereo", action="store_true")
    # parser.add_argument("--depth", action="store_true") # 对于kitti 没有深度图

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()
    print('hello??')
    torch.multiprocessing.set_start_method('spawn')

    from data_readers.tartan import test_split
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")
    
    #遍历scans下所有scene dir
    test_scenes = sorted(os.listdir(args.datapath))
    if args.sceneid != None:
        test_scenes = [ args.sceneid ]

    ate_list = []
    for scene in test_scenes:
        scenedir = os.path.join(args.datapath, scene)
        print("Performing evaluation on {}".format(scenedir))
        # outdir = "result/scannet"
        outposedir = os.path.join(scenedir, 'posedroid')
        outkfdepthdir = os.path.join(scenedir, 'depthdroidsm') #1/8
        outkfupdepthdir = os.path.join(scenedir, 'depthdroid') #上采样后的
        if not os.path.isdir(outposedir):
            os.mkdir(outposedir)
        if not os.path.isdir(outkfdepthdir):
            os.mkdir(outkfdepthdir)
        if not os.path.isdir(outkfupdepthdir):
            os.mkdir(outkfupdepthdir)
        # 读入depth intrinsic
        intrinsics_arr = np.loadtxt(os.path.join(scenedir, "intrinsic/intrinsic_depth.txt"))
        intrinsics_vec = [intrinsics_arr[0][0], intrinsics_arr[1][1], intrinsics_arr[0][2], intrinsics_arr[1][2]]
        print('intrinsics_vec:\n',intrinsics_vec)
        torch.cuda.empty_cache()
        droid = Droid(args)

        for (tstamp, image, intrinsics) in tqdm(image_stream(scenedir, stereo=False,intrinsics_vec=intrinsics_vec)):
            droid.track(tstamp, image, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA add path to save kf depth
        traj_est = droid.terminate(image_stream(scenedir),outkfdepthdir=outkfupdepthdir)

        ### do evaluation ###
        evaluator = TartanAirEvaluator()
        # gt_file = os.path.join(scenedir, "pose_left.txt")
        # traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz
        # 读取scannet格式的pose
        # scale depths to balance rot & trans
        DEPTH_SCALE = 5.0
        length = len(os.listdir(os.path.join(scenedir, 'color'))) #该scene总帧数
        trajs = []
        for t in range(length):
            Tp = np.loadtxt(os.path.join(scenedir, "pose", str(t)+".txt"))
            Ts2c = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
            Tc2s = np.linalg.inv(Ts2c)
            Tp_cam = np.dot(np.dot(Ts2c, Tp), Tc2s)
            trajs.append(tf.SE2pos_quat(Tp_cam))
        poses = np.array(trajs)
        poses[:,:3] /= DEPTH_SCALE
        traj_ref = copy.deepcopy(poses)
        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
        
        print(results)
        ate_list.append(results["ate_score"])
        # 保存输出轨迹
        trajoptpath = os.path.join(outposedir,scene+'.txt')
        print('save out traj at: \t', trajoptpath)
        np.savetxt(trajoptpath,traj_est,delimiter=' ')

    print("Results")
    print(ate_list)

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.show()

