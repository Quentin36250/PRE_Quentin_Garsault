LR=(0.001 0.0005 0.0001 0.00005 0.00001 0.000005 0.000001)
for lr in ${LR[@]}
do
    sbatch slurm_RL_ADMM.sh $lr
done
