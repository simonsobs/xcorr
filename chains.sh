#! /bin/bash

#SBATCH -J chains
#SBATCH -N 4
#SBATCH -C haswell
#SBATCH -A m3404
#SBATCH -o logs/%x.o%j
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --mail-user=krolewski@berkeley.edu
#SBATCH --mail-type=ALL

export PATH=/global/homes/a/akrolew/miniconda3/bin:$PATH
srun -N 4 -n 8 python chains.py