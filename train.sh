#!/bin/bash
#SBATCH -J train_cluster_mdrp          # nombre del job
#SBATCH -p investigacion               # nombre de la particion 
#SBATCH --nodes=1                      # Number of nodes to allocate
#SBATCH --tasks-per-node=1             # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8              # Number of CPUs per task

module unload python/2.7.17 # carga el modulo de python version 2.7.17
module load python/3.9.2 
python train.py # siendo prueba_python.py el nombre del programa python
module unload python/3.9.2
