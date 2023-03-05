#!/bin/bash


KITTI_PATH=datasets/kitti/sequences#/mnt/lustre/donglongrui/kitti
#--disable_vis --stereo
python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=8 --disable_vis --stereo $@