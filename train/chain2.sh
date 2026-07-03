#!/bin/bash
B=/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL
CKD=$B/train/lightning_logs_mem_full/lightning_logs/version_2/checkpoints
cd $B/test
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssm_vla 2>/dev/null || true
source $B/eval_render.sh 2>/dev/null
export PYTHONPATH=$B CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "##### 1) PROFILING (figures, #22) #####"
python -u profile_figures.py 2>&1 | grep -aE "ckpt:|t_vision|fs_blocks|total speedup|FIGURES|peak mem" | tail -8
echo "##### 2) DE-RISK epoch-120 (k=0 vs k=16, 25 rollouts, same ckpt) #####"
echo "--- k=0 (FULL, sees cue) ---"
WINDOW_K=0 SEED=1234 python -u evaluate_fs_mem.py 25 "$CKD" 2>&1 | grep -aE "ckpt:|window k|SUCCESS|bin-choice|avg return"
echo "--- k=16 (windowed-eval, cue hidden) ---"
WINDOW_K=16 SEED=1234 python -u evaluate_fs_mem.py 25 "$CKD" 2>&1 | grep -aE "ckpt:|window k|SUCCESS|bin-choice|avg return"
echo "##### CHAIN2 DONE #####"
