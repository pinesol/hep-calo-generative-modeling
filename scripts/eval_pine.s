#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=eval
#SBATCH --mail-type=END
#SBATCH --mail-user=akp258@nyu.edu
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

module purge
module load h5py/intel/2.7.0rc2
module load tensorflow/python2.7/20170218
module load scikit-learn/intel/0.18.1

cd /scratch/akp258/udon/scripts/

#python eval.py dl_full ../exp/alex/spiral_gan_test_final_samples.pkl ../exp/test/ false
#python eval.py spiral_gan/spiral_gan_wasserpine_201704200531_samples.pkl dl_full spiral_gan/
python eval.py spiral_gan/spiral_gan_wasserpine_201704210736_samples.pkl dl_full spiral_gan/
