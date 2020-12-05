#!/bin/bash

#SBATCH --job-name=ex_jb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --output="outputs/example.out"
#SBATCH --export=NONE

module purge
module load anaconda3/4.3.1

source activate sd_bandits_env

cd ~/sd_bandits/scripting/

python opb_run.py --policy EpsilonGreedy --estimator ReplayMethod