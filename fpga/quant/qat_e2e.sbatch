#!/bin/bash
#SBATCH --job-name=qat_e2e
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ANON/logs/qat_e2e_%j.log
#SBATCH --error=/scratch/ANON/logs/qat_e2e_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
python qat_e2e.py --iters 1500 --lr 5e-5 --clamp 16 --out /scratch/ANON/flashocc_qat_e2e.pth 2>&1 | tail -80
echo QAT_E2E_SBATCH_DONE
