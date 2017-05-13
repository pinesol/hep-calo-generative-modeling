#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=mean_stddev
#SBATCH --mail-type=END
#SBATCH --mail-user=akp258@nyu.edu
#SBATCH --output=mean_stddev_%j.out
#SBATCH --error=mean_stddev_%j.err

module purge
module load pillow/intel/4.0.0
module load h5py/intel/2.7.0rc2

cd /scratch/akp258/udon/scripts/

python mean_stddev.py
