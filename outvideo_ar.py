# -*- coding:utf8 -*-
"""[从可视化结果图像序列生成视频]
for ar glass mono
"""
import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import glob 
import yaml
import torch.nn.functional as F

KITTI_PATH='datasets/tcsvt'

def undistort_fisheye(img_path,K,D,DIM,scale=1,imshow=False):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_LINEAR)
    Knew = K.copy()
    # if scale:#change fov  这里pengzhen说可以s>1 之后尝试
    #     Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_32FC1) #CV_32FC1 CV_16SC2
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) #BORDER_CONSTANT BORDER_TRANSPARENT  INTER_LINEAR INTER_CUBIC
    # if imshow:
    #     cv2.imshow("undistorted/", undistorted_img)
    return undistorted_img

#[400,640]  -> [320,512]
def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1): #.../glass
    """ image generator """

    tmproot = 'tmpvis' #保存处理后的图像
    unroot = 'undistort'
    sename = datapath[-8:-6]
    tmpath = os.path.join(tmproot, sename)
    folder = os.path.exists(tmpath)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tmpath)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    distdir = os.path.join(unroot, sename)
    if not os.path.exists(distdir):
        os.makedirs(distdir)

    # 读取该数据下sensor.yaml
    skip_lines = 2
    leftymp = os.path.join(datapath, 'cam0', 'sensor.yaml')
    rightymp = os.path.join(datapath, 'cam1', 'sensor.yaml')
    with open(leftymp) as f:
        for i in range(skip_lines):
            f.readline()
        yl = yaml.safe_load(f)
        raw_size = yl['resolution'] # [W, H] 640,400
        intrinsics_vec = yl["intrinsics"] # fx fy cx cy
        distortion_vec = yl["distortion_coefficients"]
        T_Bl = np.array(yl['T_BS']['data']).reshape(4,4)
        K_l = np.array([intrinsics_vec[0], 0.0, intrinsics_vec[2], 0.0, intrinsics_vec[1], intrinsics_vec[3], 0.0, 0.0, 1.0]).reshape(3,3)
        d_l = np.array(distortion_vec).reshape(8)
        
    with open(rightymp) as f:
        for i in range(skip_lines):
            f.readline()
        yr = yaml.safe_load(f)
        intrinsics_vec = yr["intrinsics"]
        distortion_vec = yr["distortion_coefficients"]
        T_Br = np.array(yr['T_BS']['data']).reshape(4,4)
        K_r = np.array([intrinsics_vec[0], 0.0, intrinsics_vec[2], 0.0, intrinsics_vec[1], intrinsics_vec[3], 0.0, 0.0, 1.0]).reshape(3,3)
        d_r = np.array(distortion_vec).reshape(8)
    # print('\n')
    # print('K_l: \n',K_l)
    # print('K_r: \n',K_r)
    # print('d_l: \n', d_l)
    # print('d_r: \n', d_r)
    # print('T_Bl: \n', T_Bl)
    # print('T_Br: \n', T_Br)

    T_rl = np.linalg.inv(T_Br).dot(T_Bl)
    R_rl = T_rl[0:3,0:3]
    t_rl = T_rl[0:3,3]

    R_l, R_r, P_l, P_r, _,_,_= cv2.stereoRectify(K_l, np.zeros(4), K_r, np.zeros(4), (raw_size[0], raw_size[1]), R_rl, t_rl,
                                                 flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=(raw_size[0], raw_size[1])) #为了不改变fov 就设为1  0就是去掉所有黑边
    intrinsics_vec = [P_l[0,0], P_l[1,1], P_l[0,2], P_l[1,2]] #？
    map_l = cv2.initUndistortRectifyMap(K_l, np.zeros(5), R_l, P_l[:3,:3], (raw_size[0], raw_size[1]), cv2.CV_32FC1) #需要W,H [:3,:3]
    map_r = cv2.initUndistortRectifyMap(K_r, np.zeros(5), R_r, P_r[:3,:3], (raw_size[0], raw_size[1]), cv2.CV_32FC1)
    T_l43 = np.vstack((R_l, np.array([0,0,0])))
    T_l = np.hstack((T_l43, np.array([[0],[0],[0],[1]]))) #Trect_l
    T_l_inv = np.linalg.inv(T_l) #Tl_rect
    global T_Brect
    T_Brect = T_Bl.dot(T_l_inv) # 矫正相机系 到 imu(gt)
    if stereo:
        print("T_rl: \n",T_rl)
        print("R_l: \n",R_l)
        print("R_r: \n",R_r)
        print("P_l: \n",P_l)
        print("P_r: \n",P_r)
        print("intrinsics vec: \n", intrinsics_vec)
        print("T_Brect: \n", T_Brect)
    
    ht0, wd0 = [raw_size[1],raw_size[0]] #480, 752euroc原始数据size  benchmark 400,640  若像tartanair 那样 0.8后就是 320 512 先按这个来

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4]) 
        #测试 先去畸变 得到正常双目 再正常双目校正
        imgl_un = undistort_fisheye(imgL,K_l,d_l[:4],raw_size,scale=1,imshow=False) 
        imgr_un = undistort_fisheye(imgR,K_r,d_r[:4],raw_size,scale=1,imshow=False) 
             
        images = [cv2.remap(imgl_un, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)] #(400,640,3)

        # print("an img size : ", images[0].shape)
        if stereo:
            distortst = np.hstack((imgl_un, imgr_un))
            images += [cv2.remap(imgr_un, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
            testimgs = np.hstack((images[0], images[1]))
            testfile = str(tstamp)+'.png'
            # cv2.imwrite(os.path.join(tmpath,testfile), testimgs)
            # cv2.imwrite(os.path.join(distdir,testfile), distortst)
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False) #插值到给定大小 H,W

        yield stride*t, images

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datapath", help="path to euroc sequence") # .../glass 
    parser.add_argument("--seq_name", required=True, help="seq_name", default = "glassst/atrium/A0") #用来保存中间depth flow等可视化结果
    parser.add_argument("--image_size", default=[320,512]) # raw size glass W 640 H 400
    
    args = parser.parse_args()

    
    print(args) 
    seq = args.seq_name   #直接到可视化结果图片
    seqpath = seq.split('/')[-2]+'/'+seq.split('/')[-1] #atrium/A0
    seqid = seq.split('/')[-1] #A0
    datapath = os.path.join(KITTI_PATH,seqpath,'glass') #
    print("Running video on {}".format(datapath))
    flowdir = "visresult/flow/"+seq #/flow/03
    depthdir = "visresult/depth/"+seq
    wtxdir = "visresult/weightx/"+seq
    wtydir = 'visresult/weighty/'+seq
    print("flow dir {}".format(flowdir))
    # seqpath = os.path.join(KITTI_PATH,seq)
    depthimlist = sorted(glob.glob(os.path.join(depthdir, '*.png')))
    kfnum = len(depthimlist)
    fps = 5          # 视频帧率 30 10
    input_data = image_stream(datapath)
    inimgtup = [] #list
    for (tstamp, image) in tqdm(input_data): #遍历每一帧数据 显示tracking 每一帧进度条
        # print(image.shape)
        _,_,ht, wd= image.shape #1,3,320,512
        size = (2*wd, ht) # 需要转为视频的图片的尺寸
        # break
        imagearr = image.cpu().numpy()[0] #ndarry 3,320,512 float32
        imagearr1 = imagearr.transpose(1,2,0) #320，512，3
        inimgtup.append((tstamp, imagearr1))
    
    video_depth = cv2.VideoWriter("visresult/"+seq.split('/')[-3]+"-"+seqid+"_depth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    video_flow = cv2.VideoWriter("visresult/"+seq.split('/')[-3]+"-"+seqid+"_flow.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    video_wt = cv2.VideoWriter("visresult/"+seq.split('/')[-3]+"-"+seqid+"_wt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for kf in range(kfnum):
        filename = depthimlist[kf][-10:] #*.png
        # print(filename)
        
        # leftimgf = os.path.join(seqpath,'image_2',filename)
        # print(leftimgf)
        # image = cv2.imread(leftimgf)
        # h0, w0, _ = image.shape
        # h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))#243
        # w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))#, 807
        # image = cv2.resize(image, (w1, h1))
        # image = image[:h1-h1%8, :w1-w1%8] #(ht,wd)
        
        filestamp = int(depthimlist[kf][-10:-4]) #kf id
        tstamp, image = inimgtup[filestamp] #拿出输入droid的图像  1,3,320,512
        if tstamp != filestamp:
            print("ERROR: tstamp: {:d} != filestamp : {:d}".format(tstamp, filestamp))
            break
        # print(filestamp)
        image=image.astype( np.uint8 )
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