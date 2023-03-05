import torch
import lietorch
import numpy as np
import os
import cv2
from flow_vis import flow_to_image
import matplotlib as mpl
import matplotlib.cm as cm
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

# https://www.zhihu.com/question/274926848/answer/784905939
def show_wtonimg(img, mask, filepath):
    """[summary]
    Args:
        img ([ndarray]): [HxWx3]
        mask ([ndarray]): [HxW]
        filepath ([str]): [filepath]
    """
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET) #(h0,w0,3)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(filepath, np.uint8(255*cam))

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops
from droid_net import upsample_disp, cvx_upsample


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1,
    sceneid="scene0000_00"):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.sceneid = sceneid

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """


        with self.video.get_lock():
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_sens[ix] = self.video.disps_sens[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
        self.age += 1


    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, _ = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)


                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP
            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            # dense bundle adjustment
            self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True


    @torch.cuda.amp.autocast(enabled=False)
    def vis_lowmem(self, use_inactive=False, outkfdepthdir=None):
        """  """

        sceneid = self.sceneid
        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        # for step in range(steps):
        # print("Global BA Iteration #{}".format(step+1))
        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        s = 8
        for i in range(0, self.jj.max()+1, s):
            v = (self.ii >= i) & (self.ii < i + s)
            iis = self.ii[v]
            jjs = self.jj[v]

            ht, wd = self.coords0.shape[0:2]
            corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

            with torch.cuda.amp.autocast(enabled=True):
                
                net, delta, weight, damping, upmask = \
                    self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)


            self.net[:,v] = net
            self.target[:,v] = coords1[:,v] + delta.float()
            self.weight[:,v] = weight.float()
            self.damping[torch.unique(iis)] = damping #(1,8,48,64)
            upmask = upmask.to("cuda:1") #for vis
            self.upmask[torch.unique(iis)] = upmask.float() #(1,8,576,48,64) # for vis

        damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + 1e-7
        upmask = self.upmask[torch.unique(self.ii)].contiguous() ##(#v,576,48,64) #for vis
        target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
        weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

        # dense bundle adjustment
        self.video.ba(target, weight, damping, self.ii, self.jj, 1, t, 
            itrs=2, lm=1e-5, ep=1e-2, motion_only=False)
        

        self.video.dirty[:t] = True
        depthdir = 'visresult/depth/'+sceneid # seq
        flowdir = 'visresult/flow/'+sceneid
        weightxdir = 'visresult/weightx/'+sceneid
        weightydir = 'visresult/weighty/'+sceneid
        if not os.path.exists(depthdir):
            os.makedirs(depthdir)
        if not os.path.exists(flowdir):
            os.makedirs(flowdir)
        if not os.path.exists(weightxdir):
            os.makedirs(weightxdir)
        if not os.path.exists(weightydir):
            os.makedirs(weightydir)
        #这里已经进行完所有优化了 在这里拿出 disp flow weight 吧！
        fnum = torch.unique(self.ii).shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.video.reproject(self.ii, self.jj) # --(1,#e,48,64,2),(1,60,48,64,1)
        flow = coords1-self.coords0 # -(ht,wd,2)
        batch, edgenum, _, _, dim = flow.shape
        flow = flow.view(batch*edgenum,ht,wd,dim) # (#edge,48,64,2)
        # edgemask = upmask[self.ii,:,:,:] # 本来是和顶点数相同 这次按每条边的顶点i索引 就得到边数
        weight = self.weight.view(-1, ht, wd, 2).contiguous() #(#edge,2,48,64)
        # weight = weight.permute(0,2,3,1)#(#edge,48,64,2)
        
        #对上采样后的光流可视化 和 权重两方向可视化
        for fi in range(fnum):
            torch.cuda.empty_cache() 
            rawatmp = int(self.video.tstamp[fi].data.cpu().numpy()) #原始输入的id
            # 从边的集合中找到.ii==fi的 首个边吧
            siiarr = self.ii.data.cpu().numpy()
            edgeindex = np.argwhere( siiarr==fi ) #(,1)
            edgeindex = edgeindex[:,0]
            edgevjjs = self.jj.data.cpu().numpy()[edgeindex]
            distedge = np.abs(edgevjjs - fi)
            bigind = np.where(distedge>=3)
            
            if bigind[0].shape[0] > 0:
                selectedg = bigind[0][0]
            else:
                selectedg = 0
            # 把上采样放这里 节省cuda
            weighti = weight[edgeindex[selectedg]][None]
            flowi = flow[edgeindex[selectedg]][None] #(1,ht,wd,2)
            # edgemki = edgemask[edgeindex[0,0]][None] #(1,576,ht,wd)
            edgemki = upmask[fi][None] #for vis
            flowi = flowi.to("cuda:1")
            weighti = weighti.to("cuda:1")
            flowifi = cvx_upsample(flowi,edgemki)
            upweighti = cvx_upsample(weighti,edgemki) #(#e,h0,w0,2)
            # flowifi = upflow[edgeindex[0]] #(h0,w0,2)
            flowfi = flowifi.data.cpu().numpy()[0]
            upwtfi = upweighti.data.cpu().numpy()[0]
            upwtix = upwtfi[:,:,0] # (h0,w0,1)
            upwtiy = upwtfi[:,:,1]
            # map flow to rgb image
            flo = flow_to_image(flowfi) #(h0,w0,3)
            flo = flo[:, :, [2,1,0]] #为啥纯黑 /255.0
            flo_path = os.path.join(flowdir,'{:06d}.png'.format(rawatmp))
            cv2.imwrite(flo_path, flo)
            #
            imgi = self.video.images[fi] #(3,h0,w0)
            imgi = imgi.permute(1,2,0).data.cpu().numpy() #(h0,w0,3)
            imgi = np.float32(imgi) / 255
            wtxpath = os.path.join(weightxdir,'{:06d}.png'.format(rawatmp))
            wtypath = os.path.join(weightydir,'{:06d}.png'.format(rawatmp))
            show_wtonimg(imgi,upwtix,wtxpath)
            show_wtonimg(imgi,upwtiy,wtypath)
            
        
        # 深度可视化
        disps = self.video.disps #(_,48,64)
        disps = disps[torch.unique(self.ii)].contiguous()
        fnum,_,_ = disps.shape
        #先上采样
        upmask = upmask.view(1,-1,576,ht,wd) #for vis
        disps = disps.view(1,-1,ht,wd)#(1,_,48,64)

        for i in range(fnum):
            rawatmp = int(self.video.tstamp[i].data.cpu().numpy()) #原始输入的id
            upmaski = upmask[:,i,:,:,:].view(1,-1,576,ht,wd)#(1,1,576,ht,wd)
            dispi = disps[:,i,:,:].view(1,-1,ht,wd)#(1,1,ht,wd)
            dispi = dispi.to("cuda:1") #depthdroidsm
            updispi = upsample_disp(dispi,upmaski) #(1,1,h0,w0)
            # dispi = updisps[i]
            updispi = updispi[0,0,:,:]
            smdispi = dispi[0,0,:,:]
            smdisparr = smdispi.data.cpu().numpy()
            disparr = updispi.data.cpu().numpy()
            #的到深度
            smdeptharr = 1./smdisparr
            deptharr = 1./disparr
            # 保存两个尺度 逆深度的值 #保存格式 png？
            smdepthfile = os.path.join(outkfdepthdir+'sm', '{:d}.png'.format(rawatmp))
            depthfile = os.path.join(outkfdepthdir, '{:d}.png'.format(rawatmp))
            cv2.imwrite(smdepthfile,smdeptharr)
            cv2.imwrite(depthfile,deptharr)
            disp_vis = depth_visualizer(disparr) #逆深度的可视化
            disp_path = os.path.join(depthdir,'{:d}.png'.format(rawatmp)) #06对于scannet 不用补0
            cv2.imwrite(disp_path, cv2.cvtColor(disp_vis, cv2.COLOR_RGB2BGR))


    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)