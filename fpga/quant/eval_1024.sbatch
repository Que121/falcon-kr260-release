#!/bin/bash
#SBATCH --job-name=eval1024
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/ANON/logs/eval1024_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
# board mIoU on the first 1024 val frames + in-pipeline FP32 baseline (from the dumped 'occ' on the SAME frames)
python eval_board_miou.py /scratch/ANON/board_argmax_1024.npy 1024 /scratch/ANON/buildB_io_full 2>&1 | grep -E "evaluating|mIoU|EVAL_DONE|====|bicycle|motorcycle|pedestrian" | tail -30
echo EVAL1024_DONE
