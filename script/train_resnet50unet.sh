#!/bin/bash
#SBATCH --account=project_2008180
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of MPI tasks
#SBATCH --cpus-per-task=10          
#SBATCH --mem=20G
#SBATCH --time=03:00:00             # hh:mm:ss
#SBATCH --gres=gpu:v100:1           # Reserve 1 GPUs
PROJAPPL=/projappl/project_2008180

module purge
module load pytorch

srun -u python $PROJAPPL/mitotem/main.py train resnet50_unet