#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=local
#SBATCH --mail-type=END
#SBATCH --mail-user=akp258@nyu.edu
#SBATCH --output=local_%j.out
#SBATCH --error=local_%j.err

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load pillow/intel/4.0.0
module load h5py/intel/2.7.0rc2
module load tensorflow/python2.7/20170218
module load scikit-image/intel/0.12.3
# NOTE: using keras 1.2 which I've loaded locally

cd /scratch/akp258/udon/scripts/spiral_gan/

python -u local.py local.cfg
