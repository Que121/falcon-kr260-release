#!/bin/bash
#SBATCH --job-name=ri8full
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/ANON/logs/realint8_%j.log
#SBATCH --error=/scratch/ANON/logs/realint8_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "######## FP32 (FULL 6019) ########"
python flashocc_eval_realint8.py --mode fp32 --cap 0 2>&1 | grep -E "====|mIoU|EVAL_DONE| [a-z]" | tail -25
echo "######## REAL DPU-INT8 (FULL 6019) ########"
python flashocc_eval_realint8.py --mode realint8 --clamp 16 --cap 0 2>&1 | grep -E "====|mIoU|EVAL_DONE|weight-FQ| [a-z]" | tail -30
echo REALINT8_DONE
