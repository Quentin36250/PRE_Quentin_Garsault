#!/bin/bash
#
#SBATCH --job-name=RL_ADMM_affichage
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:30:00
#SBATCH --mem=1g

source activate directPolicyOptim
python test.py $1 