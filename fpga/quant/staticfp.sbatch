#!/bin/bash
#SBATCH --job-name=staticfp
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ANON/logs/staticfp_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
G="====|mIoU:|mIoU of|staticfp|EVAL_DONE"
python flashocc_eval_staticfp.py --calib 32 --cap 256 2>&1 | grep -E "$G" | tail -20
echo STATICFP_DONE
