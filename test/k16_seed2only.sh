#!/bin/bash
# Re-run ONLY k16 seed=2 (seed=2 crashed instantly in the paired run; seed0=92%, seed1=86% already in hand).
# FULL output (no grep filter) so any crash is visible.
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd "$M/test"
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssm_vla 2>/dev/null || true
source "$M/eval_render.sh" 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKD="$M/train/lightning_logs/version_16/checkpoints"
echo "##### k16 seed=2 ALONE (50 rollouts) $(date +%H:%M) #####"
SEED=2 python -u evaluate_fs_windowed.py 50 16 "$CKD"
echo "##### K16 SEED2 ALONE DONE rc=$? $(date +%H:%M) #####"
