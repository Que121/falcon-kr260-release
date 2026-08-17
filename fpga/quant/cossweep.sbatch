#!/bin/bash
#SBATCH --job-name=cossweep
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ANON/logs/cossweep_%j.log
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd /home/ANON/ANONPROJ_occbev/FlashOCC
export PYTHONPATH="/home/ANON/ANONPROJ_occbev/FlashOCC:${PYTHONPATH:-}"
python /scratch/ANON/gather_cos_sweep.py --cap 96 2>&1 | grep -E "ranks|cos |SWEEP|Error|Traceback"
echo COSSWEEP_SBATCH_DONE
