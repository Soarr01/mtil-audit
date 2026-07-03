#!/bin/bash
M=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
cd $M/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null||true; conda activate ssm_vla 2>/dev/null
export PYTHONPATH=$M CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ep in episode_1 episode_2 episode_19; do
  echo "######## EPISODE=$ep ########"
  EPISODE=$ep python -u audit_upstream_step.py 2>&1 | grep -aE "reference forward_sequence vs GT|PRISTINE/published|FIXED \(ours\)|episode:"
done
echo "######## AUDIT 3EP DONE ########"
