#!/bin/bash
#SBATCH --account=project_2008180
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of MPI tasks
#SBATCH --cpus-per-task=6          
#SBATCH --mem=20G
#SBATCH --time=03:00:00             # hh:mm:ss
#SBATCH --gres=gpu:v100:1           # Reserve 1 GPUs

module purge
module load pytorch

srun -u python /projappl/project_2008180/mitotem/src/models/train_unet.py --data_dir /scratch/project_2008180/dataset --run_dir /scratch/project_2008180/train_unet