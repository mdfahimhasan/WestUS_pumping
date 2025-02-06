#!/bin/bash

## running the tile_driver_westUS.py script on CPU nodes

#SBATCH --partition=smi_all
#SBATCH --ntasks=30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 #Request 1 GPU
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python tile_driver_westUS.py

