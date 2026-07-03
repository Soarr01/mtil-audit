#!/bin/bash
# Re-run k16 windowed-trained 3x50 seeds 1,2 (seed=0 already=92%) to complete the pooled Wilson CI.
# Runs AFTER ssm_vla was restored (causal_conv1d 1.6.1) — the original chain's seed1,2 died on the env corruption.
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd "$M/test"
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssm_vla 2>/dev/null || true
source "$M/eval_render.sh" 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKD="$M/train/lightning_logs/version_16/checkpoints"
for s in 1 2; do
  echo "##### k16 seed=$s (50 rollouts) #####"
  SEED=$s python -u evaluate_fs_windowed.py 50 16 "$CKD" 2>&1 | grep -aiE "SUCCESS|Rollout [0-9]+: highest|k *= *16:"
done
echo "##### K16 SEED12 DONE #####"
