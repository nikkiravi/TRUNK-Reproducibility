#!/bin/sh -l
# FILENAME:  trunk_infer.sh

#SBATCH -A debug
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --constraint 'I'
#SBATCH --constraint A100

#SBATCH --time=00:06:00
#SBATCH --job-name trunk_job_test

#SBATCH --output=/home/ravi30/logs/stdout/infer_cifar10.out
#SBATCH --error=/home/ravi30/logs/stderr/infer_cifar10.err

module load cuda
module load cudnn
module load anaconda/2020.11-py38
cd /home/ravi30/TRUNK_Tutorial_Paper/TRUNK
conda activate mnn
python main.py --infer --dataset cifar10 --model_backbone vgg --grouping_volatility