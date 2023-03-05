import torch
import torch.nn.functional as F
# -*- coding:utf8 -*-
from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1) #->(1,60,1,1,4)-> 每个 (1,60,1,1)
#
def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid( #生成网格 和feature map 尺寸有关 (48,64)
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())
    # 增加一个维度 (48,64,2)
    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """ #相机模型
    ht, wd = disps.shape[2:] # 48,64
    fx, fy, cx, cy = extract_intrinsics(intrinsics) #4个值 每个(1,60,1,1)
    
    y, x = torch.meshgrid( #坐标网格
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps) #all 1 (1,60,48,64)
    X = (x - cx) / fx # invK * (u,v) 得到相机系下归一化坐标 (48,64)-(1,60,1,1) ->(1,60,48,64)
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1) #新维度拼接 (1,60,48,64,4) 就是三维点的4d坐标 line42表明了

    if jacobian:
        J = torch.zeros_like(pts) #0初始化 (1,60,48,64,4)
        J[...,-1] = 1.0 #(1,60,48,64)
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """ #(1,60,48,64,4) (1,#edge=60,4)
    fx, fy, cx, cy = extract_intrinsics(intrinsics)##4个值 每个(1,60,1,1)
    X, Y, Z, D = Xs.unbind(dim=-1) #每个都是(1,60,48,64) 是每个三维点的各分量 还不是齐次！

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z) #把小于阈值的用1替代
    d = 1.0 / Z #d 是inverse depth
    #投像 (1,60,48,64)
    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1) #(1,60,48,64,2) 每个位置像素坐标

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """ #对点云坐标变换 不同的本地相机系之间 i->j
    X1 = Gij[:,:,None,None] * X0 #(1,60,7)->(1,60,1,1,7)*(1,60,48,64,4) 元素乘法 得到j坐标系下所有3d点
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """ map points from ii->jj """ # (1,1000,6) (1,1000,48,64) (1,1000,4) (#edge) (#edge) depths 传入的是 disps

    # inverse project (pinhole) (1,#edge,48,64) (1,#edge,4) -- (1,60,48,64,4)
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    
    # transform (1,#edge,6) Gjw*Gwi
    Gij = poses[:,jj] * poses[:,ii].inv()
    # .data:(1,60,7)  把两顶点一样的设为无效pose 但好像本来就没有吧
    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian) #点云坐标变换 (1,60,48,64,4)
    
    # project (pinhole) 正向投影 (1,60,48,64,2)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera [...,2]表示取最后一维的Z
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float() #(1,60,48,64) 0表示too close
    valid = valid.unsqueeze(-1) #增加一维标记flag(1,60,48,64,1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid #(1,60,48,64,2),(1,60,48,64,1)

def induced_flow(poses, disps, intrinsics, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid

