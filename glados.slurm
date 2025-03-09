#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=1G
#SBATCH --partition=glados12
#SBATCH -o ./report/output.%a.out # STDOUT

####### Set environment variables ###############
source ../miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate ant_env

####### Run your script #########################
python src/main.py


