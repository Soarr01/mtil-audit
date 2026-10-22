#!/bin/bash
# Usage (from RMBench root, rmbench env, server already running): bash policy/MTIL/eval.sh <task> <ckpt_setting> <seed> <gpu_id> <port>
task_name=${1:-put_back_block}; ckpt_setting=${2:-full}; seed=${3:-0}; gpu_id=${4:-4}; port=${5:-6010}
export CUDA_VISIBLE_DEVICES=${gpu_id}
PYTHONWARNINGS=ignore::UserWarning python script/eval_policy.py --config policy/MTIL/deploy_policy.yml \
  --overrides --task_name ${task_name} --task_config demo_clean_rgb --ckpt_setting ${ckpt_setting} \
  --seed ${seed} --policy_name MTIL --server_port ${port}
