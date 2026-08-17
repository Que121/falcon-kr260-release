#!/bin/bash
#SBATCH --job-name=eval6019c
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ANON/logs/eval6019c_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""   # CPU-only; the occ mIoU eval is numpy histogram IoU, no GPU needed

# restore board predictions (transferred as .npy.gz)
if [ ! -f /scratch/ANON/board_argmax_full.npy ]; then
  echo "gunzip board_argmax_full.npy.gz ..."; gunzip -kf /scratch/ANON/board_argmax_full.npy.gz
fi

# BOARD mIoU on the FULL 6019 val frames + in-pipeline FP32 baseline (dumped 'occ' on the SAME frames).
python eval_board_miou.py /scratch/ANON/board_argmax_full.npy 6019 /scratch/ANON/buildB_io_full 2>&1 \
  | grep -E "evaluating|mIoU|EVAL_DONE|====|bicycle|motorcycle|pedestrian|others|barrier|car|truck" | tail -50
echo EVAL6019C_DONE
