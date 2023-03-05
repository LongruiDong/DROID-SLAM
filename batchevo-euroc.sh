#!/bin/bash

BENCHMARK_PATH=data/rawgt # eurocgt rawgt
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
    evo_ape euroc $BENCHMARK_PATH/$seq.txt result/euroc/$seq-srectimu2-mono.csv -v --t_max_diff 0.01 -as --plot_mode zy --save_plot evoplot/$seq-apesim.pdf --no_warnings
    
    # evo_ape euroc $BENCHMARK_PATH/$seq.txt result/euroc/$seq-srectimu2.csv -v --t_max_diff 0.01 -a --plot_mode zy --save_plot evoplot/$seq-apese.pdf --no_warnings
    
    # evo_ape euroc $BENCHMARK_PATH/$seq.txt result/euroc/$seq-srectimu2-mono.csv -v --t_max_diff 0.01 --plot_mode zy --save_plot evoplot/$seq-ape-na.pdf --no_warnings

    # evo_ape euroc $BENCHMARK_PATH/$seq.txt $EUROC_PATH/$seq/mav0/state_groundtruth_estimate0/data.csv -v --t_max_diff 0.01 --plot_mode xy --save_plot evoplot/$seq-ape-na-2gt.pdf --no_warnings
# 发现rpe中 -r full -r trans_part 趋势差不多 rectimu rectimu1

done
