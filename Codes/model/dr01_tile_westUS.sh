#!/bin/bash

## running the dr01_tile_westUS.py script on CPU nodes

#SBATCH --partition=all
#SBATCH --ntasks=30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 #Request 1 GPU
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python dr01_tile_westUS.py

