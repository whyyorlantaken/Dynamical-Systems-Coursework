#!/bin/bash
#SBATCH --job-name=forking
#SBATCH --output=forking.out
#SBATCH --error=forking.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu

# Env
source ~/.bashrc
conda activate cptu

# Run
python bifurcation.py