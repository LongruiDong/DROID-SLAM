#!/bin/bash
mkdir -p log_mono_android_flip

BENCHMARK_PATH=datasets/tcsvt
CAMERA=android

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
    outdoor-office-park/O2
    stairs/S0
    stairs/S1
    stairs/S2
)

bufferset=(
    1000 #atrium/A0 1484 
    2000 #atrium/A1 3953 
    2000 # atrium/A2 3599 2000
    1200 # atrium/A3 2638 1900
    1000 # atrium/A4 2047 1300
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
    4000 #outdoor-office-park/O2 8117
    1000 #stairs/S0 1969
    1200 #stairs/S1 2518
    1100 #stairs/S2 2184
)

for ((i=0; i<21; i++)); do
    CUDA_VISIBLE_DEVICES=2,3 python evaluation_scripts/test_benchmark_mono_flip.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/$CAMERA --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/$CAMERA/gba_pose.csv --seq_name=flip/$CAMERA/${evalset[$i]} --disable_vis --buffer=${bufferset[$i]} > log_mono_android_flip/seq-$i.log $@
done