# -*- coding:utf8 -*-
import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm #进度条库
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

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """
    # 内参格式 fx fy cx cy
    # read all png images in folder 测试不需要深度
    ht0, wd0 = [480, 640] # 原始数据大小 之后对长款0.8缩放
    images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
    images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ] #list (384,512,3)
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]
        # array 转 tensor 堆叠 维度换位
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2) #array (2,3,384,512)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)
        # 输入数据的元组 List t只是伪时间
        data.append((t, images, intrinsics))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/TartanAir")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512]) #H,W
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4) # 输入帧 充分运动的阈值
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
    torch.multiprocessing.set_start_method('spawn') #torch use cuda 多任务

    from data_readers.tartan import test_split #tartanair 测试数据的部分
    from evaluation.tartanair_evaluator import TartanAirEvaluator #对定位结果评估的class

    if not os.path.isdir("figures"): # save traj plot  
        os.mkdir("figures")

    if args.id >= 0: #只测试test 中的某个序列
        test_split = [ test_split[args.id] ]

    ate_list = [] # save ATE rmse
    for scene in test_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()
        droid = Droid(args) #slam 核心算法代码 类 所有

        scenedir = os.path.join(args.datapath, scene) # (id=21) 'datasets/TartanAir/oldtown/oldtown/Easy/P007'
        input_data = image_stream(scenedir, stereo=args.stereo)
        countf = 0
        for (tstamp, image, intrinsics) in tqdm(input_data): #遍历每一帧数据 显示tracking 每一帧进度条
            droid.track(tstamp, image, intrinsics=intrinsics)
            countf = countf+1
            if countf>350: # 临时限制每个序列长度 避免显存不够
                break
                # pass

        # fill in non-keyframe poses + global BA #只在所有track和lba之后全局ba?
        traj_est = droid.terminate(image_stream(scenedir))

        ### do evaluation ###
        evaluator = TartanAirEvaluator()
        gt_file = os.path.join(scenedir, "pose_left.txt")
        traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz

        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
        
        print(results)
        ate_list.append(results["ate_score"])

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

