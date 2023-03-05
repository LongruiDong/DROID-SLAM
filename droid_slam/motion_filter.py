import cv2
import torch
import lietorch
# -*- coding:utf8 -*-
from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update # DroidNet().update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """ # (1,1,256,48,64) 拆分对特征维度对半拆分 两个(1,1,128,48,64)
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0) #和fbet 不同各自激活函数 双曲正切

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze() # t,q:(x,y,z,w) 单位阵
        ht = image.shape[-2] // 8 #再缩放8倍 in: (1,3,384,512) feature map 是 (48,64)
        wd = image.shape[-1] // 8

        # normalize images 前后区别？
        inputs = image[None, :, [2,1,0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV) #(1,1,3,384,512) 为啥要多一维

        # extract features # 光流特征map (1,128,48,64)
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ### 首帧 总不会剔除 加入系统的depthvideo中
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]]) #两个 (1,128,48,64)
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0,0], inp[0,0]) # (48,64) 首个通道值map表示啥

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume (48,64,2)[None,None] 增加两维 看作是光流 每处的值其实就是索引
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None] #(1,1,48,64,2) #gmap[None,[0]] (1,1,128,48,64)
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0) #第一个是首帧(初始化之前) 或 上个kf 第二个是当前input frame 4层拼接 (1,1,4*49,48,64)
            # LOOK UP corr中 后两维是image i ， 表示i中每个位置 在j中对应位置 7x7林域内 4个level的加权后的相似度
            # approximate flow magnitude using 1 update iteration 用的是上个有效帧的两组特征 和当前帧的对应关系
            _, delta, weight = self.update(self.net[None], self.inp[None], corr) #(1,1,48,64,2) in droid.net.py UpdateModule.forward()

            # check motion magnitue / add new frame to video #所以delta是预测的光流 每个像素位移范数的平均
            if delta.norm(dim=-1).mean().item() > self.thresh: 
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]]) #有效帧才进行context 提取
                self.net, self.inp, self.fmap = net, inp, gmap #保存的是上个有效帧 （是kf吗）
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])

            else:
                self.count += 1 #似乎是距离上次有效帧之间的被抛弃的帧数




# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        
#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:                
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1

