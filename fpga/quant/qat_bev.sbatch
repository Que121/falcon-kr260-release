#!/bin/bash
#SBATCH --job-name=qat_bev
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ANON/logs/qat_bev_%j.log
#SBATCH --error=/scratch/ANON/logs/qat_bev_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/qat_bev
python qat_bev_distill.py --data /scratch/ANON/flashocc_io_full --sd /scratch/ANON/bev_stage_sd.pth \
  --nframes 2500 --epochs 8 --bs 8 --out /scratch/ANON 2>&1 | tail -120
echo QAT_BEV_DONE
