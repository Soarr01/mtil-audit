#!/bin/bash
# RA-L revision evals (Codex review 2026-10-04), one stream per GPU; streams share lock dirs so they never duplicate.
#  (1) noens_*: temporal ensembling OFF (NO_ENSEMBLE=1) -> k=1 is a truly memoryless controller; full-history control
#      run under the SAME protocol (evaluate_fs_windowed.py with k=99999 = full forward_sequence).
#  (2) best_*: standard protocol re-evaluated at the TRUE lowest-val-loss checkpoint (filename rounding had picked a tie).
#  (3) RMBench k16 s42 eval seed 0 re-run with the per-episode diagnostic log (tag pbb_win16_s42_diag) — GPU0 stream only.
# transfer_cube: 3x50 seeded protocol (SEED 0/1/2). Outputs: train/c3_runs/fix_evals/<name>/seed<s>.log
# Usage: bash gpu_fix_eval_queue.sh <gpu> [maxpar]
G=$1; MAXPAR=${2:-3}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
R=$M/train/c3_runs; F=$R/fix_evals; mkdir -p $F
source /opt/miniconda/etc/profile.d/conda.sh; conda activate ssm_vla
source $M/train/runtime_env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "##### $(date '+%F %T') [GPU$G fix] $* #####"; }
free_mb() { nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
alive() { [ -f $1/pid ] && kill -0 $(cat $1/pid) 2>/dev/null; }
mine() { local n=0; for l in $F/*/seed*.lock; do [ -f $l/gpu ] && [ "$(cat $l/gpu)" = $G ] && alive $l && n=$((n+1)); done; echo $n; }
unlock() { rm -f "$F/$1/seed$2.lock/pid" "$F/$1/seed$2.lock/gpu"; rmdir "$F/$1/seed$2.lock" 2>/dev/null; }   # lock dir holds only pid+gpu
done_log() { grep -qE "SUCCESS RATE|=== k=" $1 2>/dev/null; }
V6=$M/train/lightning_logs/version_6/checkpoints/mamba-par-epoch=116-val_epoch_loss=0.0007.ckpt
V16=$M/train/lightning_logs/version_16/checkpoints/mamba-win16-epoch=34-val_epoch_loss=0.0014.ckpt
ck() { ls $R/$1/checkpoints/*-epoch=$2-*.ckpt | head -1; }
# name|kind|ckpt|K    kind: win (evaluate_fs_windowed), winNE (same, NO_ENSEMBLE=1), step (evaluate_step_fs), enc:<vit> (enc_wrap step)
TASKS=(
  "noens_k1_s42|winNE|$(ck win_k1_w256_s42 43)|1"
  "noens_k1_s1|winNE|$(ck win_k1_w256_s1 44)|1"
  "noens_k1_s2|winNE|$(ck win_k1_w256_s2 37)|1"
  "noens_full_s42|winNE|$V6|99999"
  "noens_full_s1|winNE|$(ck full_s1 147)|99999"
  "noens_full_s2|winNE|$(ck full_s2 111)|99999"
  "best_full_s42|step|$V6|0"
  "best_full_s1|step|$(ck full_s1 147)|0"
  "best_full_s2|step|$(ck full_s2 111)|0"
  "best_k16_s42|win|$V16|16"
  "best_k4_s1|win|$(ck win_k4_w64_s1 43)|4"
  "best_k4_s42|win|$(ck win_k4_w64_s42 40)|4"
  "best_vitb14|enc:vitb14|$(ck vitb14_full_s42 116)|0"
  "best_vits14|enc:vits14|$(ck vits14_full_s42 125)|0"
)
run() { # (background) run <name> <kind> <ckpt> <K> <seed>
  local N=$1 KIND=$2 CK=$3 K=$4 s=$5; local D=$F/$N; local L=$D/seed$s.log
  echo $BASHPID > $D/seed$s.lock/pid
  cd $M/test; source $M/eval_render.sh
  export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEED=$s
  case $KIND in
    win)   python -u evaluate_fs_windowed.py 50 $K $CK > $L 2>&1 ;;
    winNE) NO_ENSEMBLE=1 python -u evaluate_fs_windowed.py 50 $K $CK > $L 2>&1 ;;
    step)  RESULTS_DIR=$D/eval_step python -u evaluate_step_fs.py 50 $CK > $L 2>&1 ;;
    enc:*) RESULTS_DIR=$D/eval_step ENC=${KIND#enc:} python -u enc_wrap.py evaluate_step_fs.py 50 $CK > $L 2>&1 ;;
  esac
  log "END $N seed=$s: $(grep -aE 'SUCCESS RATE|=== k=' $L | tail -1)"
  unlock $N $s
}
log "START $(hostname) maxpar=$MAXPAR"
# RMBench diagnostic re-run (GPU0 stream only), runs alongside the transfer_cube jobs
if [ $G = 0 ] && ! grep -aq "Success rate: .*/100.*=>" <(tr '\r' '\n' < $R/rmbench_eval/pbb_win16_s42_diag/eval_seed0.log 2>/dev/null); then
  log "RMB START pbb_win16_s42_diag seed=0"
  ( bash $M/test/rmbench_eval.sh $R/rmbench_put_back_block_win_k16_w16_s42/checkpoints win 16 0 $G 6033 pbb_win16_s42_diag put_back_block 100 > /dev/null 2>&1
    log "RMB END pbb_win16_s42_diag: $(tr '\r' '\n' < $R/rmbench_eval/pbb_win16_s42_diag/eval_seed0.log | grep -a 'Success rate' | tail -1)" ) &
  sleep 240
fi
while true; do
  pending=0
  for T in "${TASKS[@]}"; do
    IFS='|' read -r N KIND CK K <<< "$T"
    for s in 0 1 2; do
      done_log $F/$N/seed$s.log && continue
      pending=1
      mkdir -p $F/$N
      alive $F/$N/seed$s.lock && continue
      [ "$(mine)" -lt $MAXPAR ] && [ "$(free_mb)" -ge 8000 ] || continue
      unlock $N $s; mkdir $F/$N/seed$s.lock 2>/dev/null || continue     # atomic claim across streams
      echo $$ > $F/$N/seed$s.lock/pid; echo $G > $F/$N/seed$s.lock/gpu
      log "START $N seed=$s ($KIND K=$K $(basename $CK), free $(free_mb) MiB)"; run $N $KIND $CK $K $s & sleep 90
    done
  done
  [ $pending -eq 0 ] && [ "$(mine)" -eq 0 ] && break
  sleep 60
done
log "ALL DONE"
