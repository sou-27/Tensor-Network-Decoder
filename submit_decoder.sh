#!/bin/bash
#SBATCH --job-name=Decoder                       # Job name
#SBATCH --mail-type=BEGIN,END,FAIL                # Mail events
#SBATCH --mail-user=sou2781@connect.hku.hk       # Update your email address
#SBATCH --partition=amd                           # Specific Partition (intel/amd)
#SBATCH --qos=normal                             # Specific QOS (debug/normal/long)
#SBATCH --mem=150G                                 # Request total amount of RAM
#SBATCH --time=4-12:00:00                         # Wall time limit (days-hrs:min:sec)
#SBATCH --nodes=1                                 # Total number of compute node(s)
#SBATCH --cpus-per-task=64
#SBATCH --output=%x.out.%j                        # Standard output file
#SBATCH --error=%x.err.%j                         # Standard error file



module load anaconda/py3.11
# Initialize Conda for non-interactive shell sessions
eval "$(conda shell.bash hook)"

# Activate your environment
conda activate my_env

python -u run_sweep_noise.py
