#!/bin/bash
#SBATCH -J train_cluster_mdrp          # nombre del job
#SBATCH -p investigacion               # nombre de la particion 
#SBATCH --nodes=1                      # Number of nodes to allocate
#SBATCH --tasks-per-node=1             # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8              # Number of CPUs per task

module purge
module load miniconda/3.0
eval "$(conda shell.bash hook)"
conda activate tesis
python main.py 
conda deactivate
