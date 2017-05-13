import sys


for slice_ix in range(0,25):
    to_write = '''
[inputs]
test = 1
splits = 90,10
batch_size = 32
patience = 5
slice = {}
logs_path = /Users/malkini/udon_logs/

'''.format(slice_ix)
    
    with open('/scratch/im965/udon/configs/classify_slice_{}.cfg'.format(slice_ix),'w') as f:
        f.write(to_write)
        
    
    to_write = '''
#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=20:00:00
#PBS -l mem=25GB
#PBS -N slice_{}
#PBS -j oe
#PBS -m ae

module load cuda/7.5.18
module load cudnn/7.5v5.0
source ~/.bashrc

cd /home/im965/

python -u /scratch/im965/udon/scripts/framework_classify.py /scratch/im965/udon/configs/classify_slice_{}.cfg'''.format(slice_ix,slice_ix)

    with open('/scratch/im965/udon/configs/ask.{}'.format(slice_ix),'w') as f:
        f.write(to_write)