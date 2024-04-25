#!/bin/sh -l
# FILENAME:  trunk_train.sh

#SBATCH -A standby
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --constraint 'I|K'
#SBATCH --constraint A100

#SBATCH --time=04:00:00
#SBATCH --job-name trunk_cifar_vgg

#SBATCH --output=/home/ravi30/logs/stdout/train_cifar.out
#SBATCH --error=/home/ravi30/logs/stderr/train_cifar.err

module load cuda
module load cudnn
module load anaconda/2020.11-py38
cd /home/ravi30/TRUNK_Tutorial_Paper/TRUNK
conda activate mnn
python main.py --train --dataset cifar10 --model_backbone vgg --grouping_volatility  --debug
python main.py --infer --dataset cifar10 --model_backbone vgg --grouping_volatility