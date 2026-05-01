#!/bin/bash
#SBATCH --job-name=cfpb                     # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G                  # memory per cpu-core
#SBATCH --time=12:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --output=scripts/slurm-%j-%x.out
#SBATCH --error=scripts/slurm-%j-%x.err
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2024.6
conda activate rett

python sentiment_severity.py