#!/bin/bash
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd $M/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null||true; conda activate ssm_vla 2>/dev/null
source $M/eval_render.sh 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RESULTS_DIR=/tmp/prist_smoke_out
CKD=$M/train/lightning_logs/version_6/checkpoints
echo "SMOKE seed=0 N=10 $(date +%H:%M)"
SEED=0 PRISTINE=1 stdbuf -oL -eL python -u evaluate_step_fs.py 10 "$CKD" 2>&1
echo "SMOKE rc=$? $(date +%H:%M)"
