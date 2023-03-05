# -*- coding:utf8 -*-
import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidFrontend:
    def __init__(self, net, video, args): #net: DroidNet()
        self.video = video
        self.update_op = net.update # , seq = args.seq_name
        self.graph = FactorGraph(video, net.update, max_factors=48) # 表示优化图结构， 用于能量项的帧间关系？

        # local optimization window #局部优化窗口
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup # 默认12 对于tartanair
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh #和关键帧有关
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def __update(self):
        """ add edges, perform update """

        self.count += 1
        self.t1 += 1 # 上个有效帧

        if self.graph.corr is not None: #去掉age太大的
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)
        #根据距离增加新边
        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)
        # disp_sens？
        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1): #4次
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                self.graph.update(None, None, use_inactive=True)

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True

    def __initialize(self):
        """ initialize the SLAM system """ # 直到通过warm up

        self.t0 = 0
        self.t1 = self.video.counter.value #warm up 有效帧的阈值12 也是两个id
        # create edge win r t  都是在factor graph操作的 用顶点保存边
        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)
        #在初始12帧 都作为kf 做了2轮 每轮8次调用 每次1步 在factor graph的更新
        for itr in range(8):
            self.graph.update(1, use_inactive=True) #t0=1

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)


        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1-1].clone()
        self.last_disp = self.video.disps[self.t1-1].clone()
        self.last_time = self.video.tstamp[self.t1-1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization 直到经过 12个有效输入(motion check)后才会进行初始化
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
            
        # do update 12后直到有新的有效帧(kf?)
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
