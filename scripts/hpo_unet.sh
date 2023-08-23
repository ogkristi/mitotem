#!/bin/bash
#SBATCH --account=project_2008180
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of MPI tasks
#SBATCH --cpus-per-task=10          
#SBATCH --mem=64G
#SBATCH --time=00:15:00             # hh:mm:ss
#SBATCH --gres=gpu:v100:4           # Reserve 4 GPUs

module purge
module load pytorch

srun python /projappl/project_2008180/mitotem/src/models/hpo_unet.py --data_dir /scratch/project_2008180/dataset