#!/bin/bash
# Training stream on GPU $1 (default 5). Picks, in priority order, the first unfinished job whose memory need fits
# the currently free VRAM; if none fits, waits and retries. Idempotent / resumable (train_c3.py resumes from last.ckpt).
# Can share its GPU with gpu_eval_stream.sh: while a training ramps up (first 15 min) it holds TRAIN_RAMP_G<gpu>, during
# which the eval stream starts nothing new. Never touches other GPUs (CUDA_DEVICE_ORDER=PCI_BUS_ID; no rendering here).
G=${1:-5}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
R=$M/train/c3_runs
FEAT=/data/nas07_new/PersonalData/phuocthien/mtil_data
source /opt/miniconda/etc/profile.d/conda.sh; conda activate ssm_vla
source $M/train/runtime_env.sh      # scratch/caches -> /dev/shm (root overlay full)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PRELOAD=1
log() { echo "##### $(date '+%F %T') [GPU$G train] $* #####"; }
free_mb() { nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
done_run() { [ -f $R/$1/DONE ] && return 0; grep -q "C3 RUN DONE" $R/$1/train.log 2>/dev/null && touch $R/$1/DONE; }
shm() { [ -f /dev/shm/c3feat_$1/COMPLETE ] && return
        log "build shm cache $1"; ( cd $M/train && python -u build_shm_cache.py $2 /dev/shm/c3feat_$1 | tail -1 ); }

JOBF=$M/train/train_jobs.txt      # job list, re-read every loop iteration (edit live; '#' lines ignored)
load_jobs() { mapfile -t JOBS < <(grep -v '^#' $JOBF | grep -v '^ *$'); }
load_jobs
cleanup_shm() {   # drop a RAM cache once no unfinished job needs it
  for c in rmbench_put_back_block vitl14 vits14 vitb14; do
    local need=0
    for J in "${JOBS[@]}"; do IFS='|' read -r t n s f a <<< "$J"; [ "$s" = "$c" ] && ! done_run $t && need=1; done
    [ $need -eq 0 ] && [ -d /dev/shm/c3feat_$c ] && rm -rf /dev/shm/c3feat_$c && log "removed shm cache $c"
  done
}
log "START on $(hostname) ($(nvidia-smi -i $G --query-gpu=name --format=csv,noheader))"
while true; do
  load_jobs
  # never start a job while any train_c3 is alive (e.g. an orphan from a restarted stream); it finishes on its own
  if pgrep -f "python -u train_c3.py" > /dev/null; then sleep 120; continue; fi
  remaining=0; started=0
  for J in "${JOBS[@]}"; do
    IFS='|' read -r TAG NEED CACHE SRC ARGS <<< "$J"
    done_run $TAG && continue
    remaining=1
    [ "$(free_mb)" -ge "$NEED" ] || continue
    shm $CACHE $SRC
    [ "$(free_mb)" -ge "$NEED" ] || continue
    mkdir -p $R/$TAG; log "TRAIN START $TAG (free $(free_mb) MiB)"
    touch $M/train/TRAIN_RAMP_G$G; ( sleep 900; rm -f $M/train/TRAIN_RAMP_G$G ) &
    ( cd $M/train && python -u train_c3.py $ARGS >> $R/$TAG/train.log 2>&1 )
    if grep -q "C3 RUN DONE" $R/$TAG/train.log; then touch $R/$TAG/DONE; log "TRAIN DONE $TAG"; else log "TRAIN FAILED $TAG"; fi
    cleanup_shm; started=1; break
  done
  [ $remaining -eq 0 ] && break
  [ $started -eq 0 ] && sleep 120
done
cleanup_shm
log "ALL TRAINING FINISHED"
