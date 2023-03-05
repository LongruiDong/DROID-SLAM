#!/bin/bash

BENCHMARK_PATH=datasets/euroc

evalset=(
    # MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    # MH_05_difficult
    # V1_01_easy
    # V1_02_medium
    # V1_03_difficult
    # V2_01_easy
    # V2_02_medium
    # V2_03_difficult
)


# --plot_curve
for ((i=0; i<3; i++)); do
    printf "unzip %s\t%s\t\n" "$i" "${evalset[$i]}" #"${bufferset[$i]}"
    unzip $BENCHMARK_PATH/${evalset[$i]}.zip  -d $BENCHMARK_PATH/${evalset[$i]}
done
