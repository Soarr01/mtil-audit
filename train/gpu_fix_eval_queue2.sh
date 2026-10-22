#!/bin/bash
# RA-L revision evals, round 2 (Codex review v2, 2026-10-05). One stream on GPU $1.
#  (1) noens_step_full_*: full history, TE OFF, through the FIXED RECURRENT controller (evaluate_step_fs.py
#      NO_ENSEMBLE=1) -> removes the step-vs-scan path difference from the TE-off comparison.
#  (2) RMBench (sequential, needs >=11 GB free; SAPIEN crashed when VRAM was tight):
#      pbb_full_s42_hostB   : full-history seed-42 checkpoint on THIS host (missing same-checkpoint cross-host check)
#      pbb_win16_s42_rep2   : windowed seed-42 checkpoint, eval seed 0 again on this host (same-host repeatability)
#      pbb_win16_s{42,1}_e150fv : 150-epoch fixed-validation windowed runs, eval seeds 0 and 1 (after training DONE)
# transfer_cube: 3x50 seeded protocol. Outputs: train/c3_runs/fix_evals/<name>/seed<s>.log; RMBench as usual.
# Usage: bash gpu_fix_eval_queue2.sh <gpu> [maxpar]   env NO_RMB=1: skip RMBench; NO_TRANSFER=1: skip transfer_cube jobs
# (2026-10-05: RMBench OOM-looped when sharing GPU0 with a transfer eval -> run the two kinds on different GPUs)
G=$1; MAXPAR=${2:-1}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
R=$M/train/c3_runs; F=$R/fix_evals; mkdir -p $F
source /opt/miniconda/etc/profile.d/conda.sh; conda activate ssm_vla
source $M/train/runtime_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "##### $(date '+%F %T') [GPU$G fix2] $* #####"; }
free_mb() { nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
alive() { [ -f $1/pid ] && kill -0 $(cat $1/pid) 2>/dev/null; }
mine() { local n=0; for l in $F/*/seed*.lock; do [ -f $l/gpu ] && [ "$(cat $l/gpu)" = $G ] && alive $l && n=$((n+1)); done; echo $n; }
unlock() { rm -f "$F/$1/seed$2.lock/pid" "$F/$1/seed$2.lock/gpu"; rmdir "$F/$1/seed$2.lock" 2>/dev/null; }
done_log() { grep -qE "SUCCESS RATE|=== k=" $1 2>/dev/null; }
rmb_done() { [ -f $1 ] && tr '\r' '\n' < $1 | grep -aq "Success rate: .*/100.*=>"; }
V6=$M/train/lightning_logs/version_6/checkpoints/mamba-par-epoch=116-val_epoch_loss=0.0007.ckpt
ck() { ls $R/$1/checkpoints/*-epoch=$2-*.ckpt | head -1; }
TASKS=(
  "noens_step_full_s42|$V6"
  "noens_step_full_s1|$(ck full_s1 147)"
  "noens_step_full_s2|$(ck full_s2 111)"
)
# out_tag|train_tag|mode|eval_seed|port
RMB=(
  "pbb_full_s42_hostB|rmbench_put_back_block_full_s42|full|0|6034"
  "pbb_win16_s42_rep2|rmbench_put_back_block_win_k16_w16_s42|win|0|6035"
  "pbb_win16_s42_e150fv|rmbench_put_back_block_win_k16_w16_s42_e150_fv|win|0|6036"
  "pbb_win16_s1_e150fv|rmbench_put_back_block_win_k16_w16_s1_e150_fv|win|0|6037"
  "pbb_win16_s42_e150fv|rmbench_put_back_block_win_k16_w16_s42_e150_fv|win|1|6036"
  "pbb_win16_s1_e150fv|rmbench_put_back_block_win_k16_w16_s1_e150_fv|win|1|6037"
  "pbb_full_s42_hostB|rmbench_put_back_block_full_s42|full|1|6034"
)
run() { # (background) run <name> <ckpt> <seed>
  local N=$1 CK=$2 s=$3; local D=$F/$N; local L=$D/seed$s.log
  echo $BASHPID > $D/seed$s.lock/pid
  cd $M/test; source $M/eval_render.sh
  export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEED=$s
  NO_ENSEMBLE=1 RESULTS_DIR=$D/eval_step python -u evaluate_step_fs.py 50 $CK > $L 2>&1
  log "END $N seed=$s: $(grep -aE 'SUCCESS RATE|=== k=' $L | tail -1)"
  unlock $N $s
}
rmb_loop() { # (background) sequential RMBench evals; waits for training DONE and for free VRAM
  for J in "${RMB[@]}"; do
    IFS='|' read -r OUT TT MODE S PORT <<< "$J"
    rmb_done $R/rmbench_eval/$OUT/eval_seed$S.log && continue
    until [ -f $R/$TT/DONE ]; do sleep 300; done
    until [ "$(free_mb)" -ge 11000 ]; do sleep 60; done
    log "RMB START $OUT seed=$S ($TT)"
    bash $M/test/rmbench_eval.sh $R/$TT/checkpoints $MODE 16 $S $G $PORT $OUT put_back_block 100 > /dev/null 2>&1
    log "RMB END $OUT seed=$S: $(tr '\r' '\n' < $R/rmbench_eval/$OUT/eval_seed$S.log 2>/dev/null | grep -a 'Success rate' | tail -1)"
  done
  log "RMB LOOP DONE"
}
log "START $(hostname) maxpar=$MAXPAR"
[ "${NO_RMB:-0}" = 1 ] || { rmb_loop & sleep 300; }
while [ "${NO_TRANSFER:-0}" != 1 ]; do
  pending=0
  for T in "${TASKS[@]}"; do
    IFS='|' read -r N CK <<< "$T"
    for s in 0 1 2; do
      done_log $F/$N/seed$s.log && continue
      pending=1
      mkdir -p $F/$N
      alive $F/$N/seed$s.lock && continue
      [ "$(mine)" -lt $MAXPAR ] && [ "$(free_mb)" -ge 8000 ] || continue
      unlock $N $s; mkdir $F/$N/seed$s.lock 2>/dev/null || continue
      echo $$ > $F/$N/seed$s.lock/pid; echo $G > $F/$N/seed$s.lock/gpu
      log "START $N seed=$s ($(basename $CK), free $(free_mb) MiB)"; run $N $CK $s & sleep 90
    done
  done
  [ $pending -eq 0 ] && [ "$(mine)" -eq 0 ] && break
  sleep 60
done
log "TRANSFER EVALS DONE (RMB loop may still be running)"
wait
log "ALL DONE"
