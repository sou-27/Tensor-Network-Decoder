#!/bin/bash
#SBATCH --job-name=Decoder_25
#SBATCH --mail-type=BEGIN,END,FAIL                # Mail events
#SBATCH --mail-user=sou2781@connect.hku.hk       # Update your email address
#SBATCH --partition=amd
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G                     # NOT 150G; d=3 chi=6 needs a few MB per worker
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

mkdir -p logs results

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1             # driver progress lines only; workers stay quiet

module load anaconda/py3.11
eval "$(conda shell.bash hook)"
conda activate decoder_env

python run_sweep_parallel.py \
    --d 25 --chi 6 --noise_model depolarize \
    --noises 0.160 0.165 0.170 0.175 0.180 0.185 0.190 0.195 0.200 \
    --nfail 1000 --max_shots 2000000 \
    --workers ${SLURM_CPUS_PER_TASK}