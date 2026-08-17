#!/bin/bash
#SBATCH --job-name=b4_vru_recall
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ANON/logs/b4_vru_%j.log
#SBATCH --error=/scratch/ANON/logs/b4_vru_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "######## B4 on-board VRU recall (full 6019 val) ########"
python eval_board_vru_recall.py /scratch/ANON/board_argmax_full.npy 6019
