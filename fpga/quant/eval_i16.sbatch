#!/bin/bash
#SBATCH --job-name=evalbev256
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/ANON/logs/evalbev256_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
python eval_board_miou.py /scratch/ANON/board256_i16.npy 256 2>&1 | grep -E "evaluating|mIoU|EVAL_DONE|====" | tail -25
echo EVALBEV_DONE
