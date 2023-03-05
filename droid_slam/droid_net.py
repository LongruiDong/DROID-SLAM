import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# -*- coding:utf8 -*-
from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean

#(_,48,64,1) (_,576,48,64)
def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data
# 对深度图上采样 #disp(1,_,48,64) upm(1,_,576,48,64)
def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1) #(_,48,64,1)
    mask = mask.view(batch*num, -1, ht, wd) #(_,576,48,64)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)

#?
class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus()) #out (#v,1,48,64)

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0)) #out (#v,576,48,64)

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape #(1,#edge60,128,48,64)
        net = net.view(batch*num, ch, ht, wd)#(#edge60,128,48,64)
        # 原始数据中的每个元素在新生成的独立元素张量中的索引输出
        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net)) #(#edge,128,48,64)

        net = net.view(batch, num, 128, ht, wd)#(1,#edge60,128,48,64)
        net = scatter_mean(net, ix, dim=1)#(1,#vertex,128,48,64) 每个顶点所有边 后三维平均
        net = net.view(-1, 128, ht, wd) #(#vertex,128,48,64) 初始 12frame

        net = self.relu(self.conv2(net))#(#vertex,128,48,64)

        eta = self.eta(net).view(batch, -1, ht, wd) #(1,#v,48,64) 输出作为damp
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)#(1,#v,576,48,64)

        return .01 * eta, upmask #(1,#v,48,64),(1,#v,576,48,64) 


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2 #? 196 就是近邻半径为3时inde_corr #channels  
        # input  context 直接输入 paper figure10
        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)) # out:(1,128,48,64)

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)) # out:(1,64,48,64)
        # output
        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid()) #(1,2,48,64)

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip()) #(1,2,48,64)

        self.gru = ConvGRU(128, 128+128+64) # update hidden state 
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """ #可以只有两个输入 contexture index_corr
        # net inp (1,#edge,128,48,64)
        batch, num, ch, ht, wd = net.shape 

        if flow is None: #没有光流 就用0 (1,#edge,4,48,64) 为啥通道是4 不是2？光流后附加了 表示优化前后的变化量
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd) #(1,128,48,64) 变回4维矩阵
        inp = inp.view(batch*num, -1, ht, wd) # net,inp 是contexture encoder的输出       
        corr = corr.view(batch*num, -1, ht, wd) #(1,196,48,64)
        flow = flow.view(batch*num, -1, ht, wd)#(1,4,48,64)
        # paper fig10
        corr = self.corr_encoder(corr) #(1,128,48,64)
        flow = self.flow_encoder(flow) #(1,64,48,64)
        net = self.gru(net, inp, corr, flow) #这里net作用 和 inp有何功能区别  (1,128,48,64) net 看作是 gru的hidden state

        ### update variables ### 变为5维
        delta = self.delta(net).view(*output_dim) #(#edge,2,48,64)->(1,#edge,2,48,64)
        weight = self.weight(net).view(*output_dim) #同上

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous() #(1,1,48,64,2)  [...,:2]? 含义？取最后一维 2个数
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous() #同上 为何权重也要最后维度是 2

        net = net.view(*output_dim) #(1,#edge,128,48,64)

        if ii is not None: #graph 上当前边的左顶点
            eta, upmask = self.agg(net, ii.to(net.device)) #(1,#v,48,64),(1,#v,576,48,64) 
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight #(1,#edge,128,48,64) (1,#edge,48,64,2)(1,1,48,64,2)


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        #需要学习的网络参数来自一下：
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance') # flow feature 128d paper 图9 结构图与之对应
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none') #context feature
        self.update = UpdateModule() #update opertor


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj) #(1,_,48,64,2)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj) #upm(1,#v,576,48,64)

            target = coords1 + delta

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask)) #disp(1,_,48,64) upm(1,_,576,48,64)
            residual_list.append(valid_mask * residual)


        return Gs_list, disp_list, residual_list