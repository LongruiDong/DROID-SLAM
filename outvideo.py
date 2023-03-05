# -*- coding:utf8 -*-
"""[从图像序列生成视频]
"""
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob 

KITTI_PATH='datasets/kitti'

#[370,1226]  -> [240,800]
def image_stream(datapath):
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
        
        # if stereo:
        #     images += [ cv2.resize(cv2.imread(images_right[t]), (w1, h1))[:h1-h1%8, :w1-w1%8] ]
        # array 转 tensor 堆叠 维度换位
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2) #array (2,3,240,800)
        
        if t==0:
            print('kitti image shape: [{:d}, {:d}]'.format(h0, w0))
            print('resize to: [{:d}, {:d}]'.format(h1, w1))
            print('input size: [{:d}, {:d}]'.format(image.shape[0], image.shape[1]))
        
        # 输入数据的元组 List t只是伪时间
        data.append((t, image))
    return data

def main(): 
    seq = '07'   
    flowdir = "visresult/flow/"+seq #/flow/03
    depthdir = "visresult/depth/"+seq
    wtxdir = "visresult/weightx/"+seq
    wtydir = 'visresult/weighty/'+seq
    seqpath = os.path.join(KITTI_PATH,seq)
    depthimlist = sorted(glob.glob(os.path.join(depthdir, '*.png')))
    kfnum = len(depthimlist)
    fps = 10          # 视频帧率 30 10
    input_data = image_stream(seqpath)
    
    for (tstamp, image) in tqdm(input_data): #遍历每一帧数据 显示tracking 每一帧进度条
        ht, wd, _ = image.shape
        size = (2*wd, ht) # 需要转为视频的图片的尺寸
        break
    video_depth = cv2.VideoWriter("visresult/"+seq+"_depth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    video_flow = cv2.VideoWriter("visresult/"+seq+"_flow.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    video_wt = cv2.VideoWriter("visresult/"+seq+"_wt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for kf in range(kfnum):
        filename = depthimlist[kf][-10:] #*.png
        # print(filename)
        leftimgf = os.path.join(seqpath,'image_2',filename)
        print(leftimgf)
        image = cv2.imread(leftimgf)
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))#243
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))#, 807
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8] #(ht,wd)
        depthimf = os.path.join(depthdir, filename)
        flowimf = os.path.join(flowdir, filename)
        wtximf = os.path.join(wtxdir, filename)
        wtyimf = os.path.join(wtydir, filename)
        depthim = cv2.imread(depthimf)
        flowim = cv2.imread(flowimf)
        wtxim = cv2.imread(wtximf)
        wtyim = cv2.imread(wtyimf)
        # 拼接
        #横向
        depth_out = np.concatenate((image, depthim), axis=1)
        flow_out = np.concatenate((image, flowim), axis=1)
        wt_out = np.concatenate((image, wtxim), axis=1)
        video_depth.write(depth_out)
        video_flow.write(flow_out)
        video_wt.write(wt_out)
    
     
    # video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    # for i in range(1041):      
    #     image_path = data_path + "%010d_color_labels.png" % (i+1)
    #     print(image_path)
    #     img = cv2.imread(image_path)
    #     video.write(img)
    
    video_depth.release()
    video_flow.release()
    video_wt.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()