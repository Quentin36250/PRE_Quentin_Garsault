#!/bin/bash
#
#SBATCH --job-name=RL_ADMM
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=1g

source activate directPolicyOptim
python main_keras.py $1