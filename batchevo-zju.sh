#!/bin/bash

BENCHMARK_PATH=data/zjugt

evalset=(
    A0
    A1
    A2
    A3
    A4
    A5
    A6
    A7
    B0
    B1
    B2
    B3
    B4
    B5
    B6
    B7
)


# --plot_curve --align_origin
for seq in ${evalset[@]}; do
    printf "evo: %s" "$seq"
    evo_ape euroc $BENCHMARK_PATH/$seq.txt result/zju/$seq-simu.csv -v --t_max_diff 0.01 -as --plot_mode zy --save_plot evoplot/$seq-apesim.pdf --no_warnings
    
    evo_ape euroc $BENCHMARK_PATH/$seq.txt result/zju/$seq-simu.csv -v --t_max_diff 0.01 -a --plot_mode zy --save_plot evoplot/$seq-apese.pdf --no_warnings
    
    # evo_ape euroc $BENCHMARK_PATH/$seq.txt result/euroc/$seq-srectimu2-mono.csv -v --t_max_diff 0.01 --plot_mode zy --save_plot evoplot/$seq-ape-na.pdf --no_warnings

    # evo_ape euroc $BENCHMARK_PATH/$seq.txt $EUROC_PATH/$seq/mav0/state_groundtruth_estimate0/data.csv -v --t_max_diff 0.01 --plot_mode xy --save_plot evoplot/$seq-ape-na-2gt.pdf --no_warnings
# 发现rpe中 -r full -r trans_part 趋势差不多 rectimu rectimu1

done
