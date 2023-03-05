from copy import copy
import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')
import evaluation.transformation as tf

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse

if __name__ == '__main__':
    vallist = np.loadtxt('/mnt/lustre/share_data/scannet/public_datalist_185/ScanNet/Tasks/Benchmark/scannetv2_val.txt',dtype=str)
    datapath = '/mnt/lustre/share_data/scannet/public_datalist_185/scans'
    numscene = vallist.shape[0]

    for i in range(numscene):
        scenename = str(vallist[i])
        scenedir = os.path.join(datapath,scenename)
        print('eval on seq: ',scenedir)
        length = len(os.listdir(os.path.join(scenedir, 'color'))) #该scene总帧数
        print('len: \t', length)



