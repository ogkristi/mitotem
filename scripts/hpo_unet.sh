#!/bin/bash
#SBATCH --account=project_2008180
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Number of MPI tasks
#SBATCH --cpus-per-task=40          
#SBATCH --mem=0                     # Reserve all memory on the node
#SBATCH --time=00:15:00             # hh:mm:ss
#SBATCH --gres=gpu:v100:4,nvme:5    # Reserve 5GB of memory (dataset is 2.8GB)

module purge
module load pytorch

cp /scratch/project_2008180/processed $LOCAL_SCRATCH
srun python3 ../src/models/hpo_unet.py 