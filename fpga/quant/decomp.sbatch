#!/bin/bash
#SBATCH --job-name=decomp
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ANON/logs/decomp_%j.log
#SBATCH --error=/scratch/ANON/logs/decomp_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
G="====|mIoU|EVAL_DONE|decomp"
for M in none all img bev bev_rz; do
  echo "######## MODE = $M (cap 256) ########"
  python flashocc_eval_decomp.py --mode $M --cap 256 2>&1 | grep -E "$G" | tail -25
done
echo DECOMP_DONE
