#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
# srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -N 1 --ntasks-per-node=1 --job-name=demo_sfm_bench --kill-on-bad-exit=1 python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --disable_vis 2>&1|tee log/rundemo-$now.log &

KITTI_PATH=datasets/kitti #/mnt/lustre/donglongrui/kitti
srun --partition=3d_share --mpi=pmi2 --gres=gpu:2 -n 4 --ntasks-per-node=2 --job-name=test_kittim --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=-1 --disable_vis 2>&1|tee log/runkitti-$now.log &