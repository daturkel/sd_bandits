#!/bin/bash

#SBATCH --job-name=obd3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GBgit
#SBATCH --time=100:00:00
#SBATCH --output="outputs/obd3.out"
#SBATCH --export=NONE

module purge
module load anaconda3/4.3.1

source activate sd_bandits_env

python ~/sd_bandits/scripting/obp_run.py --experiment-dir '/home/ci411/sd_bandits/experiments/obd_policy_3'