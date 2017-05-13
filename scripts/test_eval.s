#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=eval
#SBATCH --mail-type=END
#SBATCH --mail-user=charlie.guthrie@nyu.edu
#SBATCH --output=eval.out
#SBATCH --error=eval.err

module purge
module load h5py/intel/2.7.0rc2
module load tensorflow/python2.7/20170218
module load scikit-learn/intel/0.18.1

cd /home/cdg356/udon/scripts/

python eval.py ../exp/alex/spiral_gan_test_final_samples.pkl dl_test ../exp/test/ 
