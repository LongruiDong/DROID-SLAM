#!/bin/bash


EUROC_PATH=datasets/euroc0

evalset=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    # MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

for seq in ${evalset[@]}; do
    # srun  --stereo
    srun --partition=xr_research --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=test-$seq --kill-on-bad-exit=1 python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --weights=droid.pth --disable_vis --plot_curve 2>&1|tee log/$seq-mono.log & # stereo-5
    # python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --weights=droid.pth --stereo $@
done

