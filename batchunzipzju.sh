#!/bin/bash

BENCHMARK_PATH=datasets/zju

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




# --plot_curve
for ((i=0,j=i+1; i<16; i++)); do
    printf "unzip %s\t%s\t\n" "$i" "${evalset[$i]}" #"${bufferset[$i]}"
    unzip $BENCHMARK_PATH/${evalset[$i]}.zip  -d $BENCHMARK_PATH/${evalset[$i]}
done
