#!/bin/bash

BENCHMARK_PATH=datasets/tcsvt/gt_result2022

evalset=(
    atrium/A0
    atrium/A1
    atrium/A2
    atrium/A3
    atrium/A4
    corridor/C0
    corridor/C1
    corridor/C2
    corridor/C3
    exhibition-hall/E0
    exhibition-hall/E1
    exhibition-hall/E2
    indoor-office-room/I0
    indoor-office-room/I1
    indoor-office-room/I2
    outdoor-office-park/O0
    outdoor-office-park/O1
    # outdoor-office-park/O2
    stairs/S0
    stairs/S1
    stairs/S2
    # whole-floor/W0
    # whole-floor/W1
    # whole-floor/W2
    # whole-floor/W3
)

nameset=(
    atrium-A0
    atrium-A1
    atrium-A2
    atrium-A3
    atrium-A4
    corridor-C0
    corridor-C1
    corridor-C2
    corridor-C3
    exhibition-hall-E0
    exhibition-hall-E1
    exhibition-hall-E2
    indoor-office-room-I0
    indoor-office-room-I1
    indoor-office-room-I2
    outdoor-office-park-O0
    outdoor-office-park-O1
    # outdoor-office-park-O2
    stairs-S0
    stairs-S1
    stairs-S2
    # whole-floor-W0
    # whole-floor-W1
    # whole-floor-W2
    # whole-floor-W3
)

seqset=(
    A0
    A1
    A2
    A3
    A4
    C0
    C1
    C2
    C3
    E0
    E1
    E2
    I0
    I1
    I2
    O0
    O1
    # O2
    S0
    S1
    S2
    # W0
    # W1
    # W2
    # W3
)

# --plot_curve
for ((i=0,j=i+1; i<20; i++)); do
    printf "%s\t%s\t\n" "$i" "${evalset[$i]}" #"${bufferset[$i]}"
    evo_ape euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}.csv -v --t_max_diff 0.02 -as --align_origin --save_plot evoplot/${seqset[$i]}-mo-ape.pdf --no_warnings
    
    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}.csv -r trans_part --delta 30 --t_max_diff 0.02 -as --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-mo-rpe_tran.pdf --no_warnings

    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}.csv -r angle_deg --delta 30 --t_max_diff 0.02 -as --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-mo-rpe_rot.pdf --no_warnings


    evo_ape euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -v --t_max_diff 0.02 -as --align_origin --save_plot evoplot/${seqset[$i]}-st-ape.pdf --no_warnings
    
    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -r trans_part --delta 30 --t_max_diff 0.02 -as --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-st-rpe_tran.pdf --no_warnings

    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -r angle_deg --delta 30 --t_max_diff 0.02 -as --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-st-rpe_rot.pdf --no_warnings

    evo_ape euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -v --t_max_diff 0.02 -a --align_origin --save_plot evoplot/${seqset[$i]}-st-apese.pdf --no_warnings
    
    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -r trans_part --delta 30 --t_max_diff 0.02 -a --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-st-rpese_tran.pdf --no_warnings

    evo_rpe euroc $BENCHMARK_PATH/${evalset[$i]}/glass/gba_pose.csv result/${nameset[$i]}-st.csv -r angle_deg --delta 30 --t_max_diff 0.02 -a --align_origin --all_pairs -v --save_plot evoplot/${seqset[$i]}-st-rpese_rot.pdf --no_warnings
# 发现rpe中 -r full -r trans_part 趋势差不多

done
