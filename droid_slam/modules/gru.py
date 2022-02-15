import torch
import torch.nn as nn
# -*- coding:utf8 -*-

class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128): #实际上 i_planes=128+128+64=320
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1) #所谓3x3 convgru
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1) # out (1,128,48,64)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0) # out (1,128,1,1)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
    #(1,x,48,64)
    def forward(self, net, *inputs): # 128, * 128,128,64
        inp = torch.cat(inputs, dim=1) #在通道上级联 (1,320,48,64)
        net_inp = torch.cat([net, inp], dim=1) #所有输入级联(1,448,48,64)

        b, c, h, w = net.shape #(1,128,48,64) #所以net是用来做 gru的hidden state吧？
        glo = torch.sigmoid(self.w(net)) * net #(1,128,48,64)*(1,128,48,64)=(1,128,48,64) pytorch中"*" 元素相乘！ 可认为这就是"hidden state"
        glo = glo.view(b, c, h*w).mean(-1).view(b, c, 1, 1)# (1,128,48x64=3072)->(1,128)->(1,128,1,1) #对应论文 对h平均得到global context 

        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))#(1,128,48,64)+(1,128,1,1)-> (1,128,48,64)
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))# 同上
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))#(1,448,48,64)->(1,128,48,64)+(1,128,1,1)=(1,128,48,64)

        net = (1-z) * net + z * q #(1,128,48,64)
        return net


