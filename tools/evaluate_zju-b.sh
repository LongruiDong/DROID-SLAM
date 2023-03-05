#!/bin/bash


EUROC_PATH=datasets/zju

evalset=(
    B0
    B1
    B2
    B3
    B4
    B5
    B6
    B7
)

for seq in ${evalset[@]}; do
    # srun
    srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=test-$seq --kill-on-bad-exit=1 python evaluation_scripts/test_zju-b.py --datapath=$EUROC_PATH/$seq --gt=$EUROC_PATH/$seq/groundtruth/euroc_gt.csv --weights=droid.pth --disable_vis --plot_curve 2>&1|tee log/$seq.log &
done

