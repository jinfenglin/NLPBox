#!/bin/csh
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N fv_gen   # Specify job name
#$ -t 1-60      # Specify number of tasks in array

module load tensorflow/0.12-python3
set root = "/afs/crc.nd.edu/user/j/jlin6/projects/NLPBox/SENET/src"
cd $root
set log = "../logs/feature_build_$SGE_TASK_ID.output"
pip3 install --user -r requirement.txt
python3 generate_feature_vectors.py $SGE_TASK_ID 60 > $log