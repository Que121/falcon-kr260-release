#!/bin/bash
#SBATCH --job-name=ri8gpu
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/ANON/logs/ri8gpu_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
# The INT8-ALGORITHM reference on the FULL 6019 val frames, EXACT same settings that gave the 2048=31.5
# reference (idealized per-tensor-pow2 INT8 on all DPU convs; img backbone/neck/depth FP32; clamp 16 on
# BEV/occ). This is the apples-to-apples bar for "board >= INT8 algorithm" on full-val.
echo "######## REAL DPU-INT8 algorithm, clamp16, FULL 6019 ########"
python flashocc_eval_realint8.py --mode realint8 --clamp 16 --cap 0 2>&1 \
  | grep -E "mIoU|====|===>|weight-FQ|bicycle|motorcycle|pedestrian|EVAL_DONE" | tail -30
echo RI8GPU_DONE
