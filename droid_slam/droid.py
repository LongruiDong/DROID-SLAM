import torch
import lietorch
import numpy as np
from torch._C import device

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process
# -*- coding:utf8 -*-
import geom.projective_ops as pops
from droid_net import upsample_disp, cvx_upsample
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import os
from flow_vis import flow_to_image
# https://github.com/JiawangBian/SC-SfMLearner-Release/blob/7a1fdc5f108f484c66fe022f81c99281ae8b8048/eval_depth.py
def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_depth = data
    vmax = np.percentile(inv_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights) # 载入trained模型参数
        self.args = args
        self.disable_vis = args.disable_vis # 是否可视化地图

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process track 和  lba?
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process gba
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet() #该类包含所有的网络入口吧
        state_dict = OrderedDict([ # module. 被替换为空
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict) #加载模型
        self.net.to("cuda:0").eval() # 不训练 限制在卡0

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map 
            salm前端 跟踪主线程 会更改 frame graph
        """

        with torch.no_grad(): #不计算梯度
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

            # global bundle adjustment
            # self.backend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)
        
        torch.cuda.empty_cache() #for vis
        self.backend.onlyvis(steps=1, seq_name=self.args.seq_name) 
           
        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
