#!/bin/bash
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd $M/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null||true; conda activate ssm_vla 2>/dev/null
source $M/eval_render.sh 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RESULTS_DIR=$M/artifact_c1_step_audit/results
CKD=$M/train/lightning_logs/version_6/checkpoints
for s in 0 1; do
  echo "===== PRISTINE seed=$s start $(date +%H:%M) ====="
  SEED=$s PRISTINE=1 python -u evaluate_step_fs.py 50 "$CKD" 2>&1
  echo "===== seed=$s rc=$? end $(date +%H:%M) ====="
done
echo "===== PRISTINE S01 V2 DONE ====="
