#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N feature_v_classify        # Specify job name
#$ -t 1-1                     # Specify number of tasks in array

module load tensorflow/0.12-python3
set root = "/afs/crc.nd.edu/user/j/jlin6/projects/NLPBox/SENET/src"
cd $root

python3 classify_with_feature_vectors.py $SGE_TASK_ID 1