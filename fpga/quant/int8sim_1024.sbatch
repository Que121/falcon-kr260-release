#!/bin/bash
#SBATCH --job-name=int8sim1024
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/ANON/logs/int8sim1024_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
# INT8-quantized-algorithm reference (idealized per-tensor-pow2 INT8 on all DPU convs) on the SAME first 1024 val frames
echo "=== realint8 (INT8 algorithm) cap=1024 ==="
python flashocc_eval_realint8.py --mode realint8 --clamp 16 --cap 1024 2>&1 | grep -E "mIoU|====|bicycle|motorcycle|pedestrian" | tail -25
echo INT8SIM1024_DONE
