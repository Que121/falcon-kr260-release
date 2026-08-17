#!/bin/bash
#SBATCH --job-name=bB_dump16
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/ANON/logs/bB_dump16_%j.log
#SBATCH --error=/scratch/ANON/logs/bB_dump16_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
python dump_buildB.py --n 16 --withfeat --out /scratch/ANON/buildB_io16 2>&1 | tail -50
echo BUILDB_DUMP16_DONE
