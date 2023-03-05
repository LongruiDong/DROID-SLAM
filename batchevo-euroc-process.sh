#!/bin/bash

EUROC_PATH=datasets/euroc0
# datasets/euroc0/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
evalset=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)


# --plot_curve --align_origin
for seq in ${evalset[@]}; do
    printf "evo: %s" "$seq"
    python evaluation_scripts/prosess_zbx.py --datapath=$EUROC_PATH/$seq 
# 发现rpe中 -r full -r trans_part 趋势差不多

done
