import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import sys

from PIL import Image
from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()

sys.path.append('thirdparty/tartanair_tools')
import evaluation.transformation as tf


class ScanNet(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.folder = "scans" if self.mode=='training' else 'scans_test'
        self.n_frames = 2
        self.raw_size = (640,480)
        self.new_size = (512,384)
        super(ScanNet, self).__init__(name='ScanNet', **kwargs)

    # todo...
    @staticmethod 
    def is_test_scene(scene):
        return any(x in scene for x in test_split)


    def _build_dataset(self):
        from tqdm import tqdm
        print("Building ScanNet dataset")

        scene_info = {}
        scenes = os.listdir(os.path.join(self.root, self.folder))[:2]
        for scene in tqdm(sorted(scenes)):
            scene_folder = osp.join(self.root, self.folder, scene)
            length = len(os.listdir(osp.join(scene_folder, 'color')))
            images = [osp.join(scene_folder, 'color', str(i)+'.jpg') for i in range(length)]
            depths = [osp.join(scene_folder, 'depth', str(i)+'.png') for i in range(length)]
            trajs = []

            for t in range(length):
                Tp = np.loadtxt(osp.join(scene_folder, "pose", str(t)+".txt"))
                Ts2c = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
                Tc2s = np.linalg.inv(Ts2c)
                Tp_cam = np.dot(np.dot(Ts2c, Tp), Tc2s)
                trajs.append(tf.SE2pos_quat(Tp_cam))
            poses = np.array(trajs)
            poses[:,:3] /= ScanNet.DEPTH_SCALE
            

            intrinsics_arr = np.loadtxt(os.path.join(scene_folder, "intrinsic/intrinsic_depth.txt"))
            intrinsics_vec = [intrinsics_arr[0][0], intrinsics_arr[1][1], intrinsics_arr[0][2], intrinsics_arr[1][2]]
            print(intrinsics_vec)
            intrinsics = [np.array(intrinsics_vec)] * len(images)
            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)
            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info


    @staticmethod
    def image_read(image_file):
        img = Image.open(image_file)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 480))
        return img

    @staticmethod
    def depth_read(depth_file):
        depth = np.array(Image.open(depth_file)) / (ScanNet.DEPTH_SCALE * 1000.0)
        depth[depth==0] = 1.0
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


class TartanAirStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/TartanAir'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class TartanAirTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(TartanAirTestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)