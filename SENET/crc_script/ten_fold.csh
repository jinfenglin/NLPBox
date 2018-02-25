#!/bin/csh
#$ -l gpu_card=1
#$ -q gpu     # Specify queue (use ‘debug’ for development)
#$ -N senet_classify        # Specify job name

module load tensorflow/0.12-python3
set root = "/afs/crc.nd.edu/user/j/jlin6/projects/NLPBox/SENET/src"
cd $root
pip3 install --user -r requirement.txt
python3 main.py ten_fold