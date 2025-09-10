#!/bin/bash
#
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --partition=special_mne_alicehu

#SBATCH --time=3-00:00:00



cd $SLURM_SUBMIT_DIR



echo $SLURM_JOB_NODELIST
echo $SLURM_SUBMIT_HOST
echo $PWD

env
python3 run.py &> result.out
