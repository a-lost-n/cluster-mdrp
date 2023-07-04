#!/bin/bash
SBATCH -J train_cluster_mdrp # nombre del job
SBATCH -p cluster_mdrp # nombre de la particion 
SBATCH -c 8  # numero de cpu cores a usar

module unload python/2.7.17 # carga el modulo de python version 2.7.17
module load python/3.9.2 
python train.py # siendo prueba_python.py el nombre del programa python
module unload python/3.9.2
