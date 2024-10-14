#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --partition=cpu2019
#SBATCH --array=0-838
#SBATCH -o ./report/output.%a.out # STDOUT

####### Set environment variables ###############
source ../miniforge3/bin/activate
eval "$(conda shell.bash hook)"
conda activate ant_env

####### Run your script #########################
#python src/process_data.py -0
python src/process_data.py $SLURM_ARRAY_TASK_ID


