#!/bin/bash
#SBATCH --job-name=eval6019
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/ANON/logs/eval6019_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

# 0) restore the board predictions (transferred as .npy.gz, ~136 MB -> 3.85 GB)
if [ ! -f /scratch/ANON/board_argmax_full.npy ]; then
  echo "gunzip board_argmax_full.npy.gz ..."
  gunzip -kf /scratch/ANON/board_argmax_full.npy.gz
fi

# 1) BOARD mIoU on the FULL 6019 val frames + in-pipeline FP32 baseline (dumped 'occ' on the SAME frames).
#    buildB_io_full must hold frame_0000..frame_6018.npz; if it only has 2048, FP32 col will be short -> rerun dump.
python eval_board_miou.py /scratch/ANON/board_argmax_full.npy 6019 /scratch/ANON/buildB_io_full 2>&1 \
  | grep -E "evaluating|mIoU|EVAL_DONE|====|bicycle|motorcycle|pedestrian|others|barrier" | tail -40
echo EVAL6019_DONE
