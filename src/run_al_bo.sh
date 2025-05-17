#!/bin/bash
#$ -q long
#$ -pe smp 16
#$ -N AL_ei

python3 al_bo.py --acquisition ei --num_points_grid 500 1000 2000 5000 --query_size 1 > output.txt
