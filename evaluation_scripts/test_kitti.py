# -*- coding:utf8 -*-
# 测试kitti数据
import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools') #用于评估

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

import torch.nn.functional as F
from droid import Droid

import matplotlib.pyplot as plt


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)
#[370,1226]  -> [240,800]
def image_stream(datapath, intrinsics_vec=[707.0912, 707.0912, 601.8873, 183.1104], stereo=False):
    """ image generator """ # change for kitti

    # ht0, wd0 = [370, 1226] # 原始数据大小 所有序列不完全相同！
    images_left = sorted(glob.glob(os.path.join(datapath, 'image_2/*.png')))
    images_right = sorted(glob.glob(os.path.join(datapath, 'image_3/*.png')))
    # fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
    # image_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
    # depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]

    data = []
    for t in range(len(images_left)):
        image = cv2.imread(images_left[t])
        h0, w0, _ = image.shape
        
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))#243
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))#, 807
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        images = [ image ] 
        
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (w1, h1))[:h1-h1%8, :w1-w1%8] ]
        # array 转 tensor 堆叠 维度换位
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2) #array (2,3,240,800)
        intrinsics = torch.as_tensor(intrinsics_vec)
        intrinsics[0::2] *= (w1 / w0) # fx cx
        intrinsics[1::2] *= (h1 / h0) # fy cy
        if t==0:
            print('kitti image shape: [{:d}, {:d}]'.format(h0, w0))
            print('resize to: [{:d}, {:d}]'.format(h1, w1))
            print('input size: [{:d}, {:d}]'.format(image.shape[0], image.shape[1]))
            print('raw intrinsics: ', intrinsics_vec)
            print('new intrinsics: ', intrinsics)
        
        # 输入数据的元组 List t只是伪时间
        data.append((t, images, intrinsics))
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/kitti")
    parser.add_argument("--id", type=int, default=-1) # kitti中某个序列
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1400) #对于kitti 大一些 1300 1800 2000
    parser.add_argument("--image_size", default=[240, 800])
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15.0)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--stereo", action="store_true")
    # parser.add_argument("--depth", action="store_true") # 对于kitti 没有深度图

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')#torch use cuda 多任务
    test_seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    dict_intrinsics = {'00': [718.856, 718.856, 607.1928, 185.2157],
                       '01': [718.856, 718.856, 607.1928, 185.2157],
                       '02': [718.856, 718.856, 607.1928, 185.2157],
                       '03': [721.5377, 721.5377, 609.5593, 172.854],
                       '04': [707.0912, 707.0912, 601.8873, 183.1104],
                       '05': [707.0912, 707.0912, 601.8873, 183.1104],
                       '06': [707.0912, 707.0912, 601.8873, 183.1104],
                       '07': [707.0912, 707.0912, 601.8873, 183.1104],
                       '08': [707.0912, 707.0912, 601.8873, 183.1104],
                       '09': [707.0912, 707.0912, 601.8873, 183.1104],
                       '10': [707.0912, 707.0912, 601.8873, 183.1104] }
    if args.id >= 0: #只测试test 中的某个序列
        test_seqs = [ test_seqs[args.id] ]
    # seqid = '%02d'%args.id #kitti指定序列 07
    ate_list = [] # save ATE rmse 
    for seqid in test_seqs:
        # if seqid == '00' or seqid == '02' or seqid == '08' or seqid == '05' or seqid == '09': #bf=3000前提下 gba时显存不够
        #     continue   
        seqpath = os.path.join(args.datapath,seqid)
        seq_intrinsics = dict_intrinsics[seqid] #根据序列选择内参
        print("Running evaluation on {}".format(seqpath))
        print(args)
        if not os.path.isdir("figures"): # save traj plot  
            os.mkdir("figures")
        countf = 0
        droid = Droid(args) #slam 核心算法代码 类 所有
        input_data = image_stream(seqpath, intrinsics_vec=seq_intrinsics ,stereo=args.stereo)
        for (tstamp, image, intrinsics) in tqdm(input_data): #遍历每一帧数据 显示tracking 每一帧进度条
                droid.track(tstamp, image, intrinsics=intrinsics)
                countf = countf+1
                if countf>500: # 临时限制每个序列长度 避免显存不够
                    # break
                    pass
        
        # fill in non-keyframe poses + global BA #只在所有track和lba之后全局ba?
        traj_est = droid.terminate(image_stream(seqpath, intrinsics_vec=seq_intrinsics ,stereo=args.stereo))

        ### run evaluation ###

        print("Results KITTI "+seqid+"... ")

        # from evaluation.evaluator_base import ATEEvaluator, RPEEvaluator, KittiEvaluator, transform_trajs, quats2SEs
        from evaluation.tartanair_evaluator import TartanAirEvaluator #对定位结果评估的class
        from evaluation.transformation import SE_matrices2pos_quats
        
        evaluator = TartanAirEvaluator()
        gt_file = os.path.join(seqpath, "pose.txt")
        traj_ref_SE = np.loadtxt(gt_file, delimiter=' ') #(N,12)
        traj_ref = np.array(SE_matrices2pos_quats(traj_ref_SE)) #得到 对应四元数格式的gt
        results = evaluator.evaluate_one_trajectory(
                traj_ref, traj_est, scale=True, title='KITTI '+seqid)
        print(results)
        ate_list.append(results["ate_score"])

