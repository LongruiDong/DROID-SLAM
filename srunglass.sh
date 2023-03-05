#!/bin/bash
mkdir -p logarmono
mkdir -p log

BENCHMARK_PATH=datasets/tcsvt_release_data

evalset=(
    # atrium/A0
    # atrium/A1
    # atrium/A2
    # atrium/A3
    # atrium/A4
    # corridor/C0
    # corridor/C1
    # corridor/C2
    # corridor/C3
    # exhibition-hall/E0
    # exhibition-hall/E1
    # exhibition-hall/E2
    # indoor-office-room/I0
    # indoor-office-room/I1
    # indoor-office-room/I2
    # outdoor-office-park/O0
    # outdoor-office-park/O1
    outdoor-office-park/O2
    # stairs/S0
    # stairs/S1
    # stairs/S2
    whole-floor/W0
    whole-floor/W1
    whole-floor/W2
    whole-floor/W3
)

nameset=(
    # A0
    # A1
    # A2
    # A3
    # A4
    # C0
    # C1
    # C2
    # C3
    # E0
    # E1
    # E2
    # I0
    # I1
    # I2
    # O0
    # O1
    O2
    # S0
    # S1
    # S2
    W0
    W1
    W2
    W3
)

bufferset=(
    # 1000 #atrium/A0 1484 
    # 2000 #atrium/A1 3953 
    # 2000 # atrium/A2 3599 2000
    # 1200 # atrium/A3 2638 1900
    # 1000 # atrium/A4 2047 1300
    # 1100 #corridor/C0 2195
    # 1200 #corridor/C1 2857
    # 1000 #corridor/C2 2152
    # 1200 #corridor/C3 2439
    # 900 #exhibition-hall/E0 1871
    # 1200 #exhibition-hall/E1 2845
    # 1000 #exhibition-hall/E2 2050
    # 1500 # indoor-office-room/I0 3340
    # 1600 #indoor-office-room/I1 3626
    # 1400 #indoor-office-room/I2 2750
    # 2500 #outdoor-office-park/O0 4907
    # 1900 #outdoor-office-park/O1 3829
    4000 #outdoor-office-park/O2 8117
    # 1000 #stairs/S0 1969
    # 1200 #stairs/S1 2518
    # 1100 #stairs/S2 2184
    4000 #whole-floor/W0 7312
    5000 #whole-floor/W1 9355
    7000 #whole-floor/W2 14630
    7000 #whole-floor/W3 14481
)

# for seq in ${evalset[@]}; do
#     srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=test_glass --kill-on-bad-exit=1 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/$seq/glass --gt=$BENCHMARK_PATH/gt_result2022/$seq/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --buffer=1000 2>&1|tee log/runkitti-$now.log &
# done

for ((i=0; i<5; i++)); do
    # printf "%s\t%s\t\n" "$i" "${evalset[$i]}" "${bufferset[$i]}"
    srun --partition=3d_share --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 --job-name=test_${evalset[$i]} --kill-on-bad-exit=1 python evaluation_scripts/test_glass.py --datapath=$BENCHMARK_PATH/${evalset[$i]}/glass --gt=$BENCHMARK_PATH/gt_result2022/${evalset[$i]}/glass/gba_pose.csv --weights=droid.pth --disable_vis --stereo --plot_curve --buffer=${bufferset[$i]} 2>&1|tee log/seq-${nameset[$i]}.log &
done

srun --partition=3d_share --mpi=pmi2 --gres=gpu:2 -n 1 --ntasks-per-node=1 --job-name=test_scannet --kill-on-bad-exit=1 python evaluation_scripts/validate_scannet.py --datapath=/mnt/lustre/share_data/scannet/public_datalist_185/scans --weights=droid.pth --disable_vis --plot_curve --sceneid=scene0000_00 2>&1|tee log/scannet.log &