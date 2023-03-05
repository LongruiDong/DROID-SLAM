# -*- coding:utf8 -*-
"""[从图像序列生成视频]
"""
from turtle import left
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import glob 
import yaml

KITTI_PATH='datasets/kitti'

#[370,1226]  -> [240,800]
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

def main(): 
    camera = "android"
    DATA_PATH = ""
    Benchmark_Scenes = ["atrium", "corridor", "exhibition-hall", "indoor-office-room", "outdoor-office-park", "stairs"]
    for scene in Benchmark_Scenes:  
        Benchmark_scene_path = os.path.join("visresult/flow/"+camera, scene)
        Scene_Sequences = os.listdir(Benchmark_scene_path)
        for squence in Scene_Sequences:
            print("begin: "+scene+"-"+squence)
            seq = os.path.join(camera, scene, squence)
            flowdir = "visresult/flow/"+seq #/flow/03
            depthdir = "visresult/depth/"+seq
            wtxdir = "visresult/weightx/"+seq
            wtydir = 'visresult/weighty/'+seq
            
            datapath = os.path.join("datasets/tcsvt/", scene, squence, camera)
            folder_path = os.path.join(datapath, "camera")
            depthimlist = sorted(glob.glob(os.path.join(depthdir, '*.png')))
            kfnum = len(depthimlist)
            fps = 10          # 视频帧率 30 10
            size = (2*512, 384) # 需要转为视频的图片的尺寸
            if not os.path.exists(os.path.join("visresult", camera, scene)):
                os.mkdir(os.path.join("visresult", camera, scene))
            print("visresult/"+seq+"_depth.mp4")
            video_depth = cv2.VideoWriter("visresult/"+seq+"_depth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            video_flow = cv2.VideoWriter("visresult/"+seq+"_flow.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            video_wt = cv2.VideoWriter("visresult/"+seq+"_wt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            images_list = sorted(glob.glob(os.path.join(folder_path, 'images/*.png')) + glob.glob(os.path.join(folder_path, 'images/*.jpg')))
            for kf in range(kfnum):
                filename = depthimlist[kf][-10:] #*.png
                # print(filename)
                timestamp = int(filename[:-4])
                filename1 = images_list[timestamp]
                leftimgf = filename1
                image = cv2.imread(leftimgf)
                image = cv2.resize(image, (512, 384))
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

            video_depth.release()
            video_flow.release()
            video_wt.release()
            cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()