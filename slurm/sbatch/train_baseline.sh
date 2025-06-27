#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=128:00:00
#SBATCH --output=slurm/logs/train/%j.out
#SBATCH --mail-user=rwhsutherland1@sheffield.ac.uk
#SBATCH --mail-type=ALL

source activate echi_recipe

python3 run_train.py