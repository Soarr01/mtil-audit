#!/bin/bash
# W2 + C2 re-measure on an idle GPU: (1) C2 vision-latency 3x for median+spread; (2) k16 windowed-trained 3x50 seeded.
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd $M/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true; conda activate ssm_vla 2>/dev/null || true
source $M/eval_render.sh 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "##### C2 quiet-host re-measure (3 runs) #####"
for r in 1 2 3; do
  echo "--- C2 run $r ---"
  python -u profile_figures.py 2>&1 | grep -aE "t_vision|total speedup|FIGURES \+ JSON"
done
echo "##### k16 windowed-trained 3x50 seeded (transfer_cube, PRISTINE env) #####"
CKD=$M/train/lightning_logs/version_16/checkpoints
for s in 0 1 2; do
  echo "--- k16 seed=$s ---"
  SEED=$s python -u evaluate_fs_windowed.py 50 16 "$CKD" 2>&1 | grep -aiE "SUCCESS|k *= *16|rate|reward >= 4|highest"
done
echo "##### REVERIFY DONE #####"
