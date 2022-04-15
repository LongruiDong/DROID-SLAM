#!/bin/bash
mkdir -p logarmono
mkdir -p log
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

nameset=(
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

bufferset=(
    1000 #atrium/A0 1484 
    2000 #atrium/A1 3953 
    2000 # atrium/A2 3599
    1200 # atrium/A3 2638
    1000 # atrium/A4 2047
    1100 #corridor/C0 2195
    1200 #corridor/C1 2857
    1000 #corridor/C2 2152
    1200 #corridor/C3 2439
    900 #exhibition-hall/E0 1871
    1200 #exhibition-hall/E1 2845
    1000 #exhibition-hall/E2 2050
    1500 # indoor-office-room/I0 3340
    1600 #indoor-office-room/I1 3626
    1400 #indoor-office-room/I2 2750
    2500 #outdoor-office-park/O0 4907
    1900 #outdoor-office-park/O1 3829
    # 4400 #outdoor-office-park/O2 8117
    1000 #stairs/S0 1969
    1200 #stairs/S1 2518
    1100 #stairs/S2 2184
    # 4000 #whole-floor/W0 7312
    # 5000 #whole-floor/W1 9355
    # 7000 #whole-floor/W2 14630
    # 7000 #whole-floor/W3 14481
)

# for seq in ${evalset[@]}; do
#     python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/$seq/glass --gt=$BENCHMARK_PATH/gt_result2022/$seq/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=1000 $@
# done
# --plot_curve
for ((i=0,j=i+1; i<20; i++)); do
    printf "%s\t%s\t\n" "$i" "${evalset[$i]}" #"${bufferset[$i]}"
    # cudaid1=`expr $i % 2`
    # cudaid2=`expr $j % 2`
    # echo "$cudaid1, $cudaid2"
    # --stereo  --seq_name 用于保存depth
    # CUDA_VISIBLE_DEVICES=0,1 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/glass --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/glass/gba_pose.csv --weights=droid.pth --disable_vis --buffer=${bufferset[$i]} --plot_curve --seq_name glassst/${evalset[$i]} --stereo > log/seq-${nameset[$i]}1-bl.log
    CUDA_VISIBLE_DEVICES=0,1 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/glass --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/glass/gba_pose.csv --weights=droid.pth --disable_vis --buffer=${bufferset[$i]} --plot_curve --seq_name glassmo/${evalset[$i]} > logarmono/seq-${nameset[$i]}1.log
    # nohup python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/glass --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=${bufferset[$i]} --plot_curve > log/seq-$i.log 2>&1 & 
done

# CUDA_VISIBLE_DEVICES=0 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/outdoor-office-park/O0/glass --gt=$BENCHMARK_PATH/gt_result2022/outdoor-office-park/O0/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=2500 --plot_curve > log/seq-O0.log

# CUDA_VISIBLE_DEVICES=1 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/outdoor-office-park/O1/glass --gt=$BENCHMARK_PATH/gt_result2022/outdoor-office-park/O1/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=1900 --plot_curve > log/seq-O1.log

# CUDA_VISIBLE_DEVICES=2 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/outdoor-office-park/O2/glass --gt=$BENCHMARK_PATH/gt_result2022/outdoor-office-park/O2/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=4000 --plot_curve > log/seq-O2.log

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/corridor/C3/glass --gt=$BENCHMARK_PATH/gt_result2022/corridor/C3/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=1200 --plot_curve > log/seq-C3.log