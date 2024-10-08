 #!/bin/bash
   ####### Reserve computing resources #############
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=1
   #SBATCH --time=01:00:00
   #SBATCH --mem=1G
   #SBATCH --partition=cpu2019

   ####### Set environment variables ###############
   module load python/anaconda3-2018.12

   ####### Run your script #########################
   python src/process_data.py