#!/bin/bash
# RMBench closed-loop eval of an MTIL checkpoint: policy server (ssm_vla env) + RMBench simulator (rmbench env).
# Usage: bash rmbench_eval.sh <ckpt_dir> <mode full|win> <k> <seed> <gpu> <port> <tag> [task=put_back_block] [n=100]
# Seeds: RMBench protocol, st_seed = 100000*(1+seed), expert-checked seeds, n test episodes (default 100).
CK=$1; MODE=$2; K=$3; SEED=$4; G=$5; PORT=$6; TAG=$7; TASK=${8:-put_back_block}; N=${9:-100}
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
RB=/data/nas07_new/PersonalData/phuocthien/ssm_vla/Chronos/RMBench
OUT=$M/train/c3_runs/rmbench_eval/$TAG; mkdir -p $OUT
# 1) policy server
( cd $M/test && PYTHONPATH=$M CUDA_VISIBLE_DEVICES=$G /opt/miniconda/envs/ssm_vla/bin/python -u rmbench_policy_server.py \
    --task $TASK --mode $MODE --k $K --ckpt_dir $CK --port $PORT > $OUT/server_seed$SEED.log 2>&1 ) &
for i in $(seq 1 120); do grep -q listening $OUT/server_seed$SEED.log 2>/dev/null && break; sleep 5; done
grep -q listening $OUT/server_seed$SEED.log || { echo "server failed"; tail -5 $OUT/server_seed$SEED.log; exit 1; }
# 2) simulator + adapter
cd $RB
MTIL_DIAG_LOG=$OUT/diag_seed$SEED.jsonl VK_ICD_FILENAMES=/data/nas07_new/PersonalData/phuocthien/ssm_vla/rmbench_vk/nvidia_icd.json CUDA_VISIBLE_DEVICES=$G RMB_TEST_NUM=$N \
PYTHONWARNINGS=ignore::UserWarning /opt/miniconda/envs/rmbench/bin/python -u script/eval_policy.py --config policy/MTIL/deploy_policy.yml \
  --overrides --task_name $TASK --task_config demo_clean_rgb --ckpt_setting $TAG --seed $SEED --policy_name MTIL --server_port $PORT \
  > $OUT/eval_seed$SEED.log 2>&1
echo "eval rc=$?"
grep -a "Success rate" $OUT/eval_seed$SEED.log | tail -1
# 3) stop server
pkill -f "rmbench_policy_server.py --task $TASK --mode $MODE --k $K --ckpt_dir $CK --port $PORT"
