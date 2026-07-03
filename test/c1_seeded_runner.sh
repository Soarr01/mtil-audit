#!/bin/bash
# C1 fixed-step() closed-loop at the reproduction protocol: 3 x 50 rollouts -> pool -> Wilson CI.
# $1 = GPU id. Uses evaluate_step_fs.py (FIXED step(), O(T)) on the transfer_cube ckpt (version_6).
G=${1:-4}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
CKD=$M/train/lightning_logs/version_6/checkpoints
cd $M/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true; conda activate ssm_vla 2>/dev/null || true
source $M/eval_render.sh 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 2; do
  echo "##### C1 fixed-step run seed=$s (50 rollouts) #####"
  SEED=$s python -u evaluate_step_fs.py 50 "$CKD" 2>&1 | grep -aE "SUCCESS RATE|Reward >= 4|Reward >= 0|MODE="
done
echo "##### C1 SEEDED DONE #####"
