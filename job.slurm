#!/bin/bash

#SBATCH --job-name=obd_demo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=100:00:00
#SBATCH --output="log"

module purge

source ~/.bashrc

SRCDIR=$HOME/sd_bandits

conda activate sd_bandits_env

python ./scripting/obp_run.py --experiment-dir "./experiments/deezer"
