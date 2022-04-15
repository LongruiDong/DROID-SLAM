import sys
sys.path.append('droid_slam')
# -*- coding: utf-8 -*-

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

def fisheyerectify(K_l, d_l, K_r, d_r, image_size=[400, 640]):
    T_Bl = np.array([9.9997329711914062e-01, 6.6418354399502277e-03,
       -3.0394839122891426e-03, -9.8180193454027176e-03,
       6.6309631802141666e-03, -9.9997162818908691e-01,
       -3.5731517709791660e-03, 1.2429017573595047e-02,
       -3.0631299596279860e-03, 3.5529017914086580e-03,
       -9.9998897314071655e-01, -7.7026826329529285e-03, 0., 0., 0., 1.
       ]).reshape(4,4) #用来变换gt
    T_Br = np.array([
        9.9982339143753052e-01, 1.8695123493671417e-02,
       1.9050934351980686e-03, 6.9406457245349884e-02,
       1.8734551966190338e-02, -9.9955159425735474e-01,
       -2.3359693586826324e-02, 1.3070842251181602e-02,
       1.4675267739221454e-03, 2.3391259834170341e-02,
       -9.9972528219223022e-01, -7.3286700062453747e-03, 0., 0., 0., 1.
    ]).reshape(4,4)
    T_rl = np.linalg.inv(T_Br).dot(T_Bl)
    R_rl = T_rl[0:3,0:3]
    t_rl = T_rl[0:3,3]

    R_l, R_r, P_l, P_r, Q= cv2.fisheye.stereoRectify(K_l, d_l[:4], K_r, d_r[:4], (image_size[1], image_size[0]), R_rl, t_rl,
    flags=cv2.fisheye.CALIB_ZERO_DISPARITY, newImageSize=(640, 400))

    T_l43 = np.vstack((R_l, np.array([0,0,0])))
    T_l = np.hstack(T_l43, np.array([0,0,0,1])) #Trect_l
    T_l_inv = np.linalg.inv(T_l) #Tl_rect
    T_Brect = T_Bl.dot(T_l_inv) # 矫正相机系 到 imu(gt)

    return R_l, R_r, P_l, P_r, T_rl, T_Brect

def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1): #.../glass
    """ image generator """

    tmproot = 'tmpvis' #保存处理后的图像
    sename = datapath[-8:-6]
    tmpath = os.path.join(tmproot, sename)
    folder = os.path.exists(tmpath)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(tmpath)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    # T_Bl = np.array([9.9997329711914062e-01, 6.6418354399502277e-03,
    #    -3.0394839122891426e-03, -9.8180193454027176e-03,
    #    6.6309631802141666e-03, -9.9997162818908691e-01,
    #    -3.5731517709791660e-03, 1.2429017573595047e-02,
    #    -3.0631299596279860e-03, 3.5529017914086580e-03,
    #    -9.9998897314071655e-01, -7.7026826329529285e-03, 0., 0., 0., 1.
    #    ]).reshape(4,4) #用来变换gt
    # T_Br = np.array([
    #     9.9982339143753052e-01, 1.8695123493671417e-02,
    #    1.9050934351980686e-03, 6.9406457245349884e-02,
    #    1.8734551966190338e-02, -9.9955159425735474e-01,
    #    -2.3359693586826324e-02, 1.3070842251181602e-02,
    #    1.4675267739221454e-03, 2.3391259834170341e-02,
    #    -9.9972528219223022e-01, -7.3286700062453747e-03, 0., 0., 0., 1.
    # ]).reshape(4,4)
    # T_rl = np.linalg.inv(T_Br).dot(T_Bl)
    # R_rl = T_rl[0:3,0:3]
    # t_rl = T_rl[0:3,3]

    K_l = np.array([2.8565679257789628e+02, 0.0, 3.2484074110933352e+02, 0.0, 2.8589271507372979e+02, 2.0060895004526657e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-1.3631494865479999e-03, -7.9948184639614859e-04, 8.4645787044929904e-03, -4.0112405620408706e-03, 0.0]).reshape(5)
    K_r = np.array([2.8965297624144807e+02, 0.0, 3.2409910815283644e+02, 0.0, 2.8985944343659531e+02, 2.0176534871317821e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    d_r = np.array([-1.7782802426685011e-02, 4.3753190194342768e-02, -4.6115411506519849e-02, 1.8216015142547858e-02, 0.0]).reshape(5)
    # print('K_l: \n',K_l)
    # print('K_r: \n',K_r)
    # print('d_l: \n', d_l)
    # print('d_l size: \n', d_l.shape)
    # print('d_r: \n', d_r)
    # print('d_r size: \n', d_r.shape)

    R_l, R_r, P_l, P_r, T_rl, T_Brect = fisheyerectify(K_l, d_l, K_r, d_r, image_size=[400, 640])
    # R_l, R_r, P_l, P_r, Q= cv2.fisheye.stereoRectify(K_l, d_l[:4], K_r, d_r[:4], (640, 400), R_rl, t_rl,
    # flags=cv2.fisheye.CALIB_ZERO_DISPARITY, newImageSize=(640, 400))
    # R_l = np.array([
    #      0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
    #      0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
    #     -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    # ]).reshape(3,3) #这个参数？
    # R_r = np.array([
    #      0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
    #      0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
    #     -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    # ]).reshape(3,3)
    
    # P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    # P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    
    map_l = cv2.fisheye.initUndistortRectifyMap(K_l, d_l[:4], R_l, P_l[:3,:3], (640, 400), cv2.CV_32F)
    map_r = cv2.fisheye.initUndistortRectifyMap(K_r, d_r[:4], R_r, P_r[:3,:3], (640, 400), cv2.CV_32F)

    intrinsics_vec = [P_l[0,0], P_l[1,1], P_l[0,2], P_l[1,2]] #？
    # T_l43 = np.vstack((R_l, np.array([0,0,0])))
    # T_l = np.hstack(T_l43, np.array([0,0,0,1])) #Trect_l
    # T_l_inv = np.linalg.inv(T_l) #Tl_rect
    # T_Brect = T_Bl.dot(T_l_inv) # 矫正相机系 到 imu(gt)
    if stereo:
        print("T_rl: \n",T_rl)
        print("R_l: \n",R_l)
        print("R_r: \n",R_r)
        print("P_l: \n",P_l)
        print("P_r: \n",P_r)
        print("intrinsics vec: \n", intrinsics_vec)
        print("T_Brect: \n", T_Brect)
    
    ht0, wd0 = [400,640] #480, 752euroc原始数据size  benchmark 400,640  若像tartanair 那样 0.8后就是 320 512 先按这个来

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)] #(400,640,3)

        # print("an img size : ", images[0].shape)
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
            testimgs = np.hstack((images[0], images[1]))
            testfile = str(tstamp)+'.png'
            cv2.imwrite(os.path.join(tmpath,testfile), testimgs)
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False) #插值到给定大小
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics

from evaluation.transformation import pos_quat2SE_matrice, SE2pos_quat

def transformref(traj_raw, Tr):
    # Tr : T_Brect (rect)相机到imu
    Trinv = np.linalg.inv(Tr) # T_rectB
    flen  = traj_raw.shape[0]
    traj_new = []
    for i in range(flen):
        qwxyz = traj_raw[i,3:] # wxyz
        qxyzw = [qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]] # ! -> xyzw
        t = traj_raw[i,:3]
        pos_quat = [t[0],t[1],t[2],qxyzw[0],qxyzw[1],qxyzw[2],qxyzw[3]]
        SEarr = pos_quat2SE_matrice(pos_quat) # imu gt T
        newSE = (Tr.dot(SEarr)).dot(Trinv) # 到imu下
        newpos_quat = SE2pos_quat(newSE) # xyz Q: X Y Z W 注意这里的四元数输出
        nquat = [newpos_quat[0],newpos_quat[1],newpos_quat[2],
        newpos_quat[6],newpos_quat[3],newpos_quat[4],newpos_quat[5]] # 变回w x y z
        traj_new.append(nquat)
    traj_out=np.array(traj_new)

    return traj_out




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence") # .../glass 
    parser.add_argument("--gt", help="path to gt file") #... gba_pose.csv
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=14000) # 512 14000
    parser.add_argument("--image_size", default=[320,512]) # raw size glass W 640 H 400
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")

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

    for (t, image, intrinsics) in tqdm(image_stream(args.datapath, stereo=args.stereo, stride=2)):
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datapath, stride=1)) #(rectified)相机系

    K_l = np.array([2.8565679257789628e+02, 0.0, 3.2484074110933352e+02, 0.0, 2.8589271507372979e+02, 2.0060895004526657e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-1.3631494865479999e-03, -7.9948184639614859e-04, 8.4645787044929904e-03, -4.0112405620408706e-03, 0.0]).reshape(5)
    K_r = np.array([2.8965297624144807e+02, 0.0, 3.2409910815283644e+02, 0.0, 2.8985944343659531e+02, 2.0176534871317821e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    d_r = np.array([-1.7782802426685011e-02, 4.3753190194342768e-02, -4.6115411506519849e-02, 1.8216015142547858e-02, 0.0]).reshape(5)
    R_l, R_r, P_l, P_r, T_rl, T_Brect = fisheyerectify(K_l, d_l, K_r, d_r, image_size=[400, 640])
    traj_imu = transformref(traj_est, T_Brect) # x y z w x y z
    ### run evaluation ###

    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    images_list = sorted(glob.glob(os.path.join(args.datapath, 'cam0/data/*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    traj_imu = PoseTrajectory3D(
        positions_xyz=1.10 * traj_imu[:,:3],
        orientations_quat_wxyz=traj_imu[:,3:], # w x y z
        timestamps=np.array(tstamps))

    
    traj_ref = file_interface.read_euroc_csv_trajectory(args.gt) #euroc那里就没再事先变换坐标系

    traj_ref, traj_imu = sync.associate_trajectories(traj_ref, traj_imu, max_diff=0.02)

    result = main_ape.ape(traj_ref, traj_imu, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)


