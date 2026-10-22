#!/bin/bash
# RMBench put_back_block evaluation of ONE condition of the 2nd training seed (s1) on one GPU: eval seeds 0 then 1
# (100 expert-checked episodes each, diagnostic log on). Waits for an already-running eval of the same condition
# (e.g. started by rmbench_s1_evals.sh) and skips evaluations that already have 100 episodes.
# Usage: bash rmbench_cond_evals.sh <gpu> full|win
G=$1; COND=$2
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
R=$M/train/c3_runs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$COND" = full ]; then RUN=rmbench_put_back_block_full_s1; OUT=pbb_full_s1; PORT=6031
else RUN=rmbench_put_back_block_win_k16_w16_s1; OUT=pbb_win16_s1; PORT=6032; fi
done_eval() { [ -f $1 ] && tr '\r' '\n' < $1 | grep -aq "Success rate: .*/100.*=>"; }
for S in 0 1; do
  while pgrep -f "rmbench_eval.sh .*$RUN/checkpoints .* $S [0-9]+ [0-9]+ $OUT put_back_block" > /dev/null; do sleep 60; done
  done_eval $R/rmbench_eval/$OUT/eval_seed$S.log && { echo "skip $OUT seed $S"; continue; }
  echo "##### $(date '+%F %T') RMB EVAL $OUT seed=$S (GPU$G) #####"
  bash $M/test/rmbench_eval.sh $R/$RUN/checkpoints $COND 16 $S $G $PORT $OUT put_back_block 100
done
echo "##### RMBENCH $OUT EVALS DONE #####"
