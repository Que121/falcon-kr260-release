#!/bin/bash
#SBATCH --job-name=bB_dumpfull
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/ANON/logs/bB_dumpfull_%j.log
#SBATCH --error=/scratch/ANON/logs/bB_dumpfull_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
python dump_buildB.py --n 6019 --out /scratch/ANON/buildB_io_full 2>&1 | tail -60
echo BUILDB_DUMPFULL_DONE
