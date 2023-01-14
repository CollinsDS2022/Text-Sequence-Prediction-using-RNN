#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=gpu04,gpu06
#SBATCH --mem=10000
#SBATCH --mail-type=end
#SBATCH --mail-user=p.rajoria@tu-braunschweig.de

module load anaconda/3-5.0.1

source activate my_env

srun python -u hamlet_rnn.py
