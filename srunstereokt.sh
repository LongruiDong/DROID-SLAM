#!/usr/bin/env sh
  
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
# srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -N 1 --ntasks-per-node=1 --job-name=demo_sfm_bench --kill-on-bad-exit=1 python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --disable_vis 2>&1|tee log/rundemo-$now.log &

KITTI_PATH=datasets/kitti #/mnt/lustre/donglongrui/kitti
srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=-1 --disable_vis --stereo 2>&1|tee log/runkitti-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=1 --disable_vis --stereo 2>&1|tee log/runkitti1-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=2 --disable_vis --stereo 2>&1|tee log/runkitti2-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=3 --disable_vis --stereo 2>&1|tee log/runkitti3-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=4 --disable_vis --stereo 2>&1|tee log/runkitti4-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=5 --disable_vis --stereo 2>&1|tee log/runkitti5-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=6 --disable_vis --stereo 2>&1|tee log/runkitti6-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=7 --disable_vis --stereo 2>&1|tee log/runkitti7-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=8 --disable_vis --stereo 2>&1|tee log/runkitti8-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=9 --disable_vis --stereo 2>&1|tee log/runkitti9-$now.log &

srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 2 --ntasks-per-node=1 --job-name=test_kittis --kill-on-bad-exit=1 python evaluation_scripts/test_kitti.py --datapath=$KITTI_PATH --weights=droid.pth --id=10 --disable_vis --stereo 2>&1|tee log/runkitti10-$now.log &