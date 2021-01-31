#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=team08_tuning
#SBATCH --output=team08_tuning-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/10.1
# Run your python code
python3 tune.py