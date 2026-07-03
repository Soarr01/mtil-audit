#!/bin/bash
# Run the cue-swap on the bf16 aux run (version_5) as soon as a GPU in {0,4} has >=8GB free.
# (Epoch is already past 25; the gate now is GPU availability, not epoch.)
B=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
CKD=$B/train/lightning_logs_mem_full_aux/lightning_logs/version_5/checkpoints
NEED=8000
G=""
# GPU4 ONLY (GPU0 hosts the training; its momentary free windows during validation vanish and
# OOM the cue-swap). Require a SUSTAINED >=8GB free on GPU4 (two checks 30s apart) before launching.
while [ -z "$G" ]; do
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 4 2>/dev/null | tr -d ' ')
  if [ -n "$f1" ] && [ "$f1" -ge "$NEED" ] 2>/dev/null; then
    sleep 30
    f2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 4 2>/dev/null | tr -d ' ')
    if [ -n "$f2" ] && [ "$f2" -ge "$NEED" ] 2>/dev/null; then G=4; echo "[gpu] GPU4 sustained ${f1}/${f2}MiB free -> use it"; fi
  fi
  [ -z "$G" ] && sleep 90
done
HI=$(ls $CKD/*.ckpt 2>/dev/null | grep -oE 'epoch=[0-9]+' | sed 's/epoch=//' | sort -g | tail -1)
echo "[cue-swap] GPU$G, best-val ckpt of version_5 (latest epoch trained=$HI)"
cd $B/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true; conda activate ssm_vla 2>/dev/null || true
source $B/eval_render.sh 2>/dev/null
PYTHONPATH=$B CUDA_VISIBLE_DEVICES=$G WINDOW_K=0 SEED=1234 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u cueswap_diag.py 8 "$CKD" 2>&1 | grep -aE "ckpt:|pose[0-9]|follow rate|FLIP rate|out of memory"
echo "[cue-swap DONE]"
