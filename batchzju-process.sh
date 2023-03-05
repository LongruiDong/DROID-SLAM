#!/bin/bash

EUROC_PATH=datasets/zju

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
    python evaluation_scripts/prosess_zju.py --datapath=$EUROC_PATH/$seq 
done
