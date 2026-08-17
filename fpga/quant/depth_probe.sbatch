#!/bin/bash
#SBATCH --job-name=depthprobe
#SBATCH --ntasks=1
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ANON/logs/depthprobe_%j.log
set -e
source ~/miniconda3/etc/profile.d/conda.sh
module load CUDA/12.1.1
conda activate fbocc
cd ~/ANONPROJ_occbev/FlashOCC
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
CAP=64
python flashocc_depth_probe.py --mode fp32 --cap $CAP --out /scratch/ANON/depth_fp32.npz 2>&1 | grep -E "probe|PROBE_DONE"
python flashocc_depth_probe.py --mode int8 --cap $CAP --out /scratch/ANON/depth_int8.npz 2>&1 | grep -E "probe|PROBE_DONE"
python - <<'PY'
import numpy as np
a=np.load('/scratch/ANON/depth_fp32.npz'); b=np.load('/scratch/ANON/depth_int8.npz')
fa, ia = a['argmax'], b['argmax']
def cos(x,y): x=x.ravel().astype(np.float64); y=y.ravel().astype(np.float64); return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
print("==== IDEALIZED image-INT8 vs FP32 (image path only) — the TARGET the board image_split should hit ====")
print("  depth-argmax exact   : %.4f (board image_split ~0.71-0.75)" % float((fa==ia).mean()))
print("  depth-softmax cos    : %.5f (board image_split ~0.949)" % cos(a['depth'].astype(np.float32), b['depth'].astype(np.float32)))
print("  feat (context) cos   : %.5f (board image_split ~0.940)" % cos(a['feat'].astype(np.float32), b['feat'].astype(np.float32)))
print("  mean peak conf  FP32=%.3f  INT8=%.3f" % (float(a['maxp'].mean()), float(b['maxp'].mean())))
print("DEPTHCMP_DONE")
PY
echo DEPTHPROBE_DONE
