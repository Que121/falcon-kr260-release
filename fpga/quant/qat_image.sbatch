#!/bin/bash
#SBATCH --job-name=qat_img
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ANON/logs/qat_img_%j.log
#SBATCH --error=/scratch/ANON/logs/qat_img_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/qat_bev
python qat_image_distill.py --data /scratch/ANON/buildB_io_full --sd /scratch/ANON/image_full_sd.pth \
  --nframes 3000 --epochs 14 --bs 4 --out /scratch/ANON/qatd_v2 2>&1 | tail -100
echo QAT_IMG_DONE
