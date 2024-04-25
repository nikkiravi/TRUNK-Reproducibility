#!/bin/sh -l
# FILENAME:  trunk_V100.sh

#SBATCH -A standby
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --constraint 'F|C|E'
#SBATCH --constraint V100

#SBATCH --time=04:00:00
#SBATCH --job-name trunk_cifar_vgg_v100

#SBATCH --output=/home/ravi30/logs/stdout/trunk_cifar_v100.out
#SBATCH --error=/home/ravi30/logs/stderr/trunk_cifar_v100.err

module load cuda
module load cudnn
module load anaconda/2020.11-py38
cd /home/ravi30/TRUNK_Tutorial_Paper/TRUNK
conda activate mnn
python main.py --train --dataset cifar10 --model_backbone vgg --grouping_volatility  --debug