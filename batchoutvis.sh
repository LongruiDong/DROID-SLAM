#!/bin/bash
BENCHMARK_PATH=datasets/tcsvt

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


for ((i=0,j=i+1; i<20; i++)); do
    printf "%s\t%s\t\n" "$i" "${evalset[$i]}" #"${bufferset[$i]}"
    # CUDA_VISIBLE_DEVICES=0,1 
    python outvideo_ar.py --seq_name=glassmo/${evalset[$i]} #> logarmono/seq-${nameset[$i]}1.log
    # nohup python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/glass --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=${bufferset[$i]} --plot_curve > log/seq-$i.log 2>&1 & 
done
