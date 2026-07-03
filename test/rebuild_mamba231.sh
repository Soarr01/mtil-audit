#!/bin/bash
# Rebuild mamba_ssm 2.3.1 from source into ssm_vla (the leaked longhorn editable uninstalled the real 2.3.1).
# NON-editable, absolute pip, gcc-12 host compiler, ssm_vla's CUDA 12.1 (matches torch 2.5.1+cu121).
# NO PATH prepend (that caused the original leak). NO -e (editable caused the shadowing).
set +e
source /opt/miniconda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssm_vla 2>/dev/null || true
PFX=/opt/miniconda/envs/ssm_vla
PY=$PFX/bin/python
echo "env: $CONDA_PREFIX ; py: $PY ; $($PY -c 'import torch;print("torch",torch.__version__)')"

# 0. remove the ABI-broken torch2.6 wheel install
$PY -m pip uninstall -y mamba_ssm > /tmp/mamba_src_build.log 2>&1

# 1. clone v2.3.1 (fresh)
cd /data/nas07_new/PersonalData/phuocthien/ssm_vla
[ -d mamba-231src ] || git clone --depth 1 -b v2.3.1 https://github.com/state-spaces/mamba.git mamba-231src >> /tmp/mamba_src_build.log 2>&1
echo "clone: $(ls mamba-231src/setup.py 2>/dev/null && echo ok)"

# 2. build NON-editable with gcc-12 + ssm_vla CUDA 12.1
cd mamba-231src
export CC=/home/phuocthien4869/.conda/envs/robossm_audit/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/phuocthien4869/.conda/envs/robossm_audit/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CUDA_HOME=$PFX CUDACXX=$PFX/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.6" MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 CUDA_VISIBLE_DEVICES=4
echo "nvcc: $($CUDACXX --version | grep -oE 'release [0-9.]+') ; gcc: $($CC -dumpversion)"
$PY -m pip install . --no-deps --no-build-isolation >> /tmp/mamba_src_build.log 2>&1
echo "BUILD rc=$?"
tail -3 /tmp/mamba_src_build.log

# 3. verify
echo "##### VERIFY #####"
cd /tmp
CUDA_VISIBLE_DEVICES=4 $PY -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__); from mamba_ssm import Mamba2; import torch; m=Mamba2(d_model=256,d_state=64,headdim=64).cuda().half(); y=m(torch.randn(1,32,256,device='cuda',dtype=torch.float16)); print('Mamba2 fwd OK', tuple(y.shape), torch.isfinite(y).all().item())" 2>&1 | tail -4
echo "##### MAMBA231 REBUILD DONE #####"
