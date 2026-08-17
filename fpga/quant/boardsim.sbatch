#!/bin/bash
#SBATCH --job-name=boardsim
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ANON/logs/boardsim_%j.log
#SBATCH --error=/scratch/ANON/logs/boardsim_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):$PYTHONPATH
G="====|mIoU|EVAL_DONE|vt_out|boardsim"
for SEAM in none board featdepth board_optfp vt8pc vt16; do
  echo "######## SEAM = $SEAM (cap 256) ########"
  python flashocc_eval_boardsim.py --seam $SEAM --cap 256 2>&1 | grep -E "$G" | tail -25
done
echo BOARDSIM_DONE
