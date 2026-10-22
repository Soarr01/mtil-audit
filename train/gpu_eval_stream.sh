#!/bin/bash
# Evaluation stream on GPU $1 (default 5). NEVER touches other GPUs: CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=$G
# for CUDA, MUJOCO_EGL_DEVICE_ID=$G for the MuJoCo EGL context (dm_control otherwise grabs EGL device 0 = GPU0).
# RMBench (SAPIEN/Vulkan) evals are SKIPPED when NO_RMB=1: the NVIDIA Vulkan driver opens every GPU at init (+16 MB on GPU0).
# Repeatedly scans the task list and launches any eval whose checkpoint is trained, up to MAXPAR concurrent jobs and
# only if enough VRAM is free. transfer_cube: 3x50 seeded protocol, one job per seed; RMBench: 100-episode protocol.
# Per-job lock dirs (<dir>/…lock/pid) prevent duplicates across restarts; dead locks are ignored. Idempotent.
G=${1:-5}
MAXPAR=${MAXPAR:-2}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
R=$M/train/c3_runs
source /opt/miniconda/etc/profile.d/conda.sh; conda activate ssm_vla
source $M/train/runtime_env.sh      # scratch/caches -> /dev/shm (root overlay full)
log() { echo "##### $(date '+%F %T') [GPU$G eval] $* #####"; }
free_mb() { nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
alive() { [ -f $1/pid ] && kill -0 $(cat $1/pid) 2>/dev/null; }
alive_locks() { local n=0; for l in $R/*/eval_seed*.lock $R/rmbench_eval/*/seed*.lock; do alive $l && n=$((n+1)); done; echo $n; }
seed_done() { grep -qE "SUCCESS RATE|=== k=" $1 2>/dev/null; }
rmb_done() { [ -f $1 ] && tr '\r' '\n' < $1 | grep -aq "Success rate: .*/100.*=>"; }
trained() { [ -f $R/$1/DONE ] || grep -q "C3 RUN DONE" $R/$1/train.log 2>/dev/null; }

run_c3() { # (background) run_c3 <tag> <seed>
  local TAG=$1 s=$2; local D=$R/$TAG; local L=$D/eval_seed$s.log
  echo $BASHPID > $D/eval_seed$s.lock/pid
  cd $M/test; source $M/eval_render.sh; export PYTHONPATH=$M CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  case $TAG in
    full_*) SEED=$s RESULTS_DIR=$D/eval_step python -u evaluate_step_fs.py 50 $D/checkpoints > $L 2>&1 ;;
    vit*)   SEED=$s RESULTS_DIR=$D/eval_step ENC=${TAG%%_*} python -u enc_wrap.py evaluate_step_fs.py 50 $D/checkpoints > $L 2>&1 ;;
    win_k*) K=$(echo $TAG | sed -E 's/win_k([0-9]+)_.*/\1/'); SEED=$s python -u evaluate_fs_windowed.py 50 $K $D/checkpoints > $L 2>&1 ;;
  esac
  log "EVAL END $TAG seed=$s: $(grep -aE 'SUCCESS RATE|=== k=' $L | tail -1)"
  rm -rf $D/eval_seed$s.lock
}
run_rmb() { # (background) run_rmb <train_tag> <mode> <seed> <out_tag> <port>
  local O=$R/rmbench_eval/$4; mkdir -p $O
  echo $BASHPID > $O/seed$3.lock/pid
  bash $M/test/rmbench_eval.sh $R/$1/checkpoints $2 16 $3 $G $5 $4 put_back_block 100 > /dev/null 2>&1
  log "RMB EVAL END $4 seed=$3: $(tr '\r' '\n' < $O/eval_seed$3.log | grep -a 'Success rate' | tail -1)"
  rm -rf $O/seed$3.lock
}
# kind:train_tag[:mode:out_tag:port]      (priority order)
TASKS=(
  "c3:full_s2" "c3:win_k16_w16_s2"
  "rmb:rmbench_put_back_block_full_s1:full:pbb_full_s1:6031"
  "rmb:rmbench_put_back_block_win_k16_w16_s1:win:pbb_win16_s1:6032"
  "c3:win_k64_w4_s42_e150" "c3:win_k1_w256_s1" "c3:win_k64_w16_s42_e150" "c3:win_k16_w4_s42" "c3:win_k4_w64_s1" "c3:win_k1_w256_s2" "c3:win_k4_w64_s2"
  "c3:vits14_full_s42" "c3:vitb14_full_s42"
)
slot() { [ ! -f $M/train/TRAIN_RAMP_G$G ] && [ "$(alive_locks)" -lt $MAXPAR ] && [ "$(free_mb)" -ge $1 ]; }
log "START on $(hostname) ($(nvidia-smi -i $G --query-gpu=name --format=csv,noheader))"
while true; do
  pending=0
  for T in "${TASKS[@]}"; do
    IFS=':' read -r KIND TAG MODE OUT PORT <<< "$T"
    if [ $KIND = c3 ]; then
      for s in 0 1 2; do
        seed_done $R/$TAG/eval_seed$s.log && continue
        pending=1
        trained $TAG || continue
        alive $R/$TAG/eval_seed$s.lock && continue
        slot 9000 || continue
        rm -rf $R/$TAG/eval_seed$s.lock; mkdir -p $R/$TAG/eval_seed$s.lock
        log "EVAL START $TAG seed=$s (free $(free_mb) MiB)"; run_c3 $TAG $s & sleep 120
      done
      n=0; for s in 0 1 2; do seed_done $R/$TAG/eval_seed$s.log && n=$((n+1)); done
      [ $n -eq 3 ] && [ ! -f $R/$TAG/EVAL_DONE ] && touch $R/$TAG/EVAL_DONE && log "EVAL DONE $TAG"
    elif [ "${NO_RMB:-0}" = 1 ]; then continue
    else
      rmb_done $R/rmbench_eval/$OUT/eval_seed0.log && continue
      pending=1
      trained $TAG || continue
      alive $R/rmbench_eval/$OUT/seed0.lock && continue
      slot 10500 || continue
      mkdir -p $R/rmbench_eval/$OUT; rm -rf $R/rmbench_eval/$OUT/seed0.lock; mkdir -p $R/rmbench_eval/$OUT/seed0.lock
      log "RMB EVAL START $OUT seed=0 (free $(free_mb) MiB)"; run_rmb $TAG $MODE 0 $OUT $PORT & sleep 180
    fi
  done
  [ $pending -eq 0 ] && [ "$(alive_locks)" -eq 0 ] && break
  sleep 60
done
log "ALL EVALS FINISHED"
