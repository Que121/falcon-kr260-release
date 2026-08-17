#!/bin/bash
#SBATCH --job-name=vitissat
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ANON/logs/vitissat_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
G="====|mIoU:|mIoU of|vitissat|EVAL_DONE"
echo "## A idealized (ceil/ceil/nofold) == realint8 sanity"
python flashocc_eval_vitissat.py --wq ceil  --aq ceil  --fold 0 --cap 256 2>&1 | grep -E "$G" | tail -20
echo "## B activation saturation only (ceil/round/nofold)"
python flashocc_eval_vitissat.py --wq ceil  --aq round --fold 0 --cap 256 2>&1 | grep -E "$G" | tail -20
echo "## C weight+act saturation (round/round/nofold)"
python flashocc_eval_vitissat.py --wq round --aq round --fold 0 --cap 256 2>&1 | grep -E "$G" | tail -20
echo "## D + BN-fold = most DPU-faithful (round/round/fold)"
python flashocc_eval_vitissat.py --wq round --aq round --fold 1 --cap 256 2>&1 | grep -E "$G" | tail -20
echo VITISSAT_DONE
