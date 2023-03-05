import sys
sys.path.append('droid_slam')
# -*- coding: utf-8 -*-
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
import yaml

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


T_Brect = np.eye(4) #全局变量初始化

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def fisheyerectify(K_l, d_l, K_r, d_r, T_Bl, T_Br, rawimage_size): #W,H
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
    T_rl = np.linalg.inv(T_Br).dot(T_Bl)
    R_rl = T_rl[0:3,0:3]
    t_rl = T_rl[0:3,3]
    T_lr = np.linalg.inv(T_rl)
    R_lr = T_lr[0:3,0:3]
    t_lr = T_lr[0:3,3]
    # W,H ?
    R_l, R_r, P_l, P_r, Q= cv2.fisheye.stereoRectify(K_l, d_l[:4], K_r, d_r[:4], (rawimage_size[0], rawimage_size[1]), R_lr, t_lr,
    flags=cv2.fisheye.CALIB_ZERO_DISPARITY, newImageSize=(0, 0))

    T_l43 = np.vstack((R_l, np.array([0,0,0])))
    T_l = np.hstack((T_l43, np.array([[0],[0],[0],[1]]))) #Trect_l
    T_l_inv = np.linalg.inv(T_l) #Tl_rect
    global T_Brect
    T_Brect = T_Bl.dot(T_l_inv) # 矫正相机系 到 imu(gt)

    return R_l, R_r, P_l, P_r, T_rl, T_Brect

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
    # K_l = np.array([2.8565679257789628e+02, 0.0, 3.2484074110933352e+02, 0.0, 2.8589271507372979e+02, 2.0060895004526657e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_l = np.array([-1.3631494865479999e-03, -7.9948184639614859e-04, 8.4645787044929904e-03, -4.0112405620408706e-03, 0.0]).reshape(5)
    # K_r = np.array([2.8965297624144807e+02, 0.0, 3.2409910815283644e+02, 0.0, 2.8985944343659531e+02, 2.0176534871317821e+02, 0.0, 0.0, 1.0]).reshape(3,3)
    # d_r = np.array([-1.7782802426685011e-02, 4.3753190194342768e-02, -4.6115411506519849e-02, 1.8216015142547858e-02, 0.0]).reshape(5)
    print('\n')
    print('K_l: \n',K_l)
    print('K_r: \n',K_r)
    print('d_l: \n', d_l)
    print('d_r: \n', d_r)
    print('T_Bl: \n', T_Bl)
    print('T_Br: \n', T_Br)

    T_rl = np.linalg.inv(T_Br).dot(T_Bl)
    R_rl = T_rl[0:3,0:3]
    t_rl = T_rl[0:3,3]
    # T_lr = np.linalg.inv(T_rl)
    # R_lr = T_lr[0:3,0:3]
    # t_lr = T_lr[0:3,3]
    # _, _, _, _, T_rl, _ = fisheyerectify(K_l, d_l, K_r, d_r, T_Bl, T_Br,rawimage_size=raw_size) #W,H
    
    
    # P_l = np.array([K_l[0,0], 0, K_l[0,2], 0,  0, K_l[1,1], K_l[1,2], 0,  0, 0, 1, 0]).reshape(3,4)
    # P_r = np.array([K_r[0,0], 0, K_r[0,2], -47.90639384423901, 0, K_r[1,1], K_r[1,2], 0,  0, 0, 1, 0]).reshape(3,4)

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
            cv2.imwrite(os.path.join(tmpath,testfile), testimgs)
            cv2.imwrite(os.path.join(distdir,testfile), distortst)
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False) #插值到给定大小 H,W
        
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
    parser.add_argument("--seq_name", required=True, help="seq_name", default = "glassst/atrium/A0") #用来保存中间depth flow等可视化结果
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=14000) # 512 14000
    parser.add_argument("--image_size", default=[320,512]) # raw size glass W 640 H 400
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20) #前端局部 窗口 帧数目
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
    countf = 0
    for (t, image, intrinsics) in tqdm(image_stream(args.datapath, stereo=args.stereo, stride=2)):
        droid.track(t, image, intrinsics=intrinsics)
        countf = countf+1
        # if countf>500: # 为了调试
        #     break
        #     # pass

    traj_est = droid.terminate(image_stream(args.datapath, stride=1)) #(rectified)相机系

    #test
    print('again T_Brect: \n', T_Brect)
    traj_imu = transformref(traj_est, T_Brect) # x y z w x y z
    ### run evaluation ###

    import evo
    from evo.core.trajectory import PoseTrajectory3D, PosePath3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    from evo.core import trajectory
    from evo.core import metrics
    from evo.core.metrics import StatisticsType
    import evo.core.geometry as geometry
    seqname = args.datapath.split('/')[-3]+'-'+args.datapath.split('/')[-2]
    images_list = sorted(glob.glob(os.path.join(args.datapath, 'cam0/data/*.png')))
    # tstamps = [float(x.split('/')[-1][:-4])/1000000000.0 for x in images_list] # ns->s !
    # 发现对于 outdoor-office-park 它的image name 不是和gt对应的时间戳 要读取 data.csv
    imgtimef = os.path.join(args.datapath, 'cam0/data.csv')
    imgtime = np.loadtxt(imgtimef, delimiter=',', usecols=[0])/1000000000.0 # s (N,)
    traj_imu0 = traj_imu #arr w x y z
    tstamparr = imgtime.reshape(-1, 1) #*1000000000.0 #转列向量  ns  对于ape对齐 用s
    traj_imu_time = np.hstack((tstamparr, traj_imu0))
    # 保存结果
    outquatdir = 'result'
    folder = os.path.exists(outquatdir)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(outquatdir)
    
    newseqname = seqname
    if args.stereo: #表示双目glass for pose result.csv   figure/traj  ate.pdf
        newseqname = seqname+'-st'
    
    #     np.savetxt(os.path.join(outquatdir,seqname+'-st.csv'),traj_imu_time,delimiter=' ')
    # else:
    np.savetxt(os.path.join(outquatdir,newseqname+'.csv'),traj_imu_time,delimiter=' ') # 还有time也保存s  对于ape 用空格
    traj_imu = PoseTrajectory3D(
        positions_xyz=1.0 * traj_imu[:,:3],
        orientations_quat_wxyz=traj_imu[:,3:], # w x y z
        timestamps=imgtime) #s np.array(tstamps)

    from evaluation.tartanair_evaluator import TartanAirEvaluator #对定位结果评估的class
    evaluator = TartanAirEvaluator()
    traj_ref = file_interface.read_euroc_csv_trajectory(args.gt) #w x y z
    
    # traj_ref0 = np.loadtxt(args.gt, delimiter=',')[:, [1, 2, 3, 5, 6, 7, 4]] # -> xyz w
    traj_ref0 = np.loadtxt(args.gt, delimiter=',')[:, :8] #t wxyz
    np.savetxt(os.path.join(outquatdir,seqname+'-gt.csv'),traj_ref0,delimiter=',')
    traj_ref, traj_imu = sync.associate_trajectories(traj_ref, traj_imu, max_diff=0.02) #注意单位ns 0.02s= ns
    # 要求是shape相同 故需要posetraj3d 转为 arry 注意都不含time gt需要是xyzw to do
    traj_ref_xyz = traj_ref.positions_xyz #(N,3)
    traj_ref_xyzw = traj_ref.orientations_quat_wxyz[:,[1,2,3,0]] # wxyz-> xyzw
    traj_ref_1 = np.hstack((traj_ref_xyz, traj_ref_xyzw))
    traj_imu_xyz = traj_imu.positions_xyz
    traj_imu_xyzw = traj_imu.orientations_quat_wxyz[:,[1,2,3,0]] # wxyz-> xyzw
    traj_imu_1 = np.hstack((traj_imu_xyz, traj_imu_xyzw))
    # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
    results0 = evaluator.evaluate_one_trajectory(traj_ref_1, traj_imu_1, #也会画轨迹
                scale=True, title=newseqname) #/home/dlr/anaconda3/envs/droidenv5/lib/python3.9/site-packages/numpy/core/fromnumeric.py:3440: RuntimeWarning: Mean of empty slice.
#   return _methods._mean(a, axis=axis, dtype=dtype,
# /home/dlr/anaconda3/envs/droidenv5/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
    print(results0)
    ate_list = [] # save ATE rmse
    ate_list.append(results0["ate_score"])
    
    traj_imu_se = PoseTrajectory3D(
        positions_xyz=traj_imu.positions_xyz,
        orientations_quat_wxyz=traj_imu.orientations_quat_wxyz, # w x y z
        timestamps=traj_imu.timestamps) #s
    traj_imu_sim = PoseTrajectory3D(
        positions_xyz=traj_imu.positions_xyz,
        orientations_quat_wxyz=traj_imu.orientations_quat_wxyz, # w x y z
        timestamps=traj_imu.timestamps) #s 
    resultsim3 = main_ape.ape(traj_ref, traj_imu_sim, est_name='traj',  #-as
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    print('SIM3-----\n',resultsim3)
    
    # #from pengzhen
    # traj_ape_kf = 
    # traj_imu.align(traj_ref, False, False)
    # traj_ape_kf_sim3 
    # r_a, t_a, s = traj_imu.align(traj_ref, True, False)
    
    # ape_metric2 = metrics.APE()
    # data_ape2 = (traj_ref, traj_ape_kf)
    # ape_metric2.process_data(data_ape2)
    # ape2 = ape_metric2.get_statistic(StatisticsType.rmse)

    # ape_metric_sim = metrics.APE()
    # data_ape_sim = (traj_ref, traj_ape_kf_sim3)
    # ape_metric_sim.process_data(data_ape_sim)
    # ape2_sim3 = ape_metric_sim.get_statistic(StatisticsType.rmse)
    
    r_a, t_a, s_KF = geometry.umeyama_alignment(traj_imu.positions_xyz.T, traj_ref.positions_xyz.T, True)
    print('scale-----\n',s_KF)
    resultse3 = main_ape.ape(traj_ref, traj_imu_se, est_name='trajse',  #-a
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
    print('SE3-----\n',resultse3)
    
    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        # save ate 的线图
        plt.savefig("figures/%s_ATE.pdf"%newseqname)
        # plt.show()
        


