#!/bin/bash
# Experiment #1: seeded re-eval (3 seeds) of transfer_cube + insertion for variance/Wilson-CI.
# transfer_cube -> GPU0 (ckpt version_6, val 0.0007); insertion -> GPU4 (ckpt version_12, val 0.0017).
# Each task's 3 seeds run SEQUENTIALLY; the two tasks run in PARALLEL.
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null; conda activate ssm_vla
source /data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/eval_render.sh
cd /data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/test
PP=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
V6=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints
V12=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_12/checkpoints
N=${1:-50}

( for s in 0 1 2; do
    echo "[transfer seed$s] start $(date '+%H:%M:%S')"
    SEED=$s CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PP PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -u evaluate_fs.py $N "$V6" > eval_transfer_seed$s.log 2>&1
    echo "[transfer seed$s] DONE $(date '+%H:%M:%S') -> $(grep -aoE 'SUCCESS RATE: [0-9.]+%' eval_transfer_seed$s.log | tail -1)"
  done ) &
PT=$!

( for s in 0 1 2; do
    echo "[insert seed$s] start $(date '+%H:%M:%S')"
    SEED=$s CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PP PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -u evaluate_fs_insert.py $N "$V12" > eval_insert_seed$s.log 2>&1
    echo "[insert seed$s] DONE $(date '+%H:%M:%S') -> $(grep -aoE 'SUCCESS RATE: [0-9.]+%' eval_insert_seed$s.log | tail -1)"
  done ) &
PI=$!

wait $PT $PI
echo "=== ALL 6 SEEDED EVALS DONE $(date '+%H:%M:%S') ==="
python /data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/test/wilson_ci.py
