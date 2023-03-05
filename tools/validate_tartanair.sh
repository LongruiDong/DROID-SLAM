#!/bin/bash


TARTANAIR_PATH=datasets/TartanAir

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --id=21 --disable_vis $@

