# Reproduction environment (pinned)

The C1 reproduction runs in one conda env. Most packages are captured machine-readably in
`requirements-lock.txt` (pip) and `environment.yml` (conda). **Two CUDA extensions must be built
from source** at the exact commits below — they are *not* installable from a version string
(no PyPI wheel matches torch 2.5.1+cu121; the published wheels start at torch 2.6 and fail on 2.5.1
with a `c10::Warning` undefined-symbol ABI error). The `@ file://` lines in `requirements-lock.txt`
are these two; replace them with the build recipe below.

## Core
- Python **3.10**, conda env (`ssm_vla`)
- **torch 2.5.1+cu121**, torchvision 0.20.1+cu121, triton 3.1.0
- CUDA toolkit **12.1** (nvcc); numpy 2.2.6, einops 0.8.2, h5py 3.16, opencv 4.13
- mujoco **2.3.7**, dm_control **1.0.14**, pytorch-lightning 2.6.5, matplotlib 3.10
- Headless render: EGL (`MUJOCO_GL=egl`); see `eval_render.sh` in the repo.

## From-source CUDA extensions (the fragile part — pin by commit)
| package | repo | tag | commit |
|---|---|---|---|
| `mamba_ssm` **2.3.1** | github.com/state-spaces/mamba | v2.3.1 | `c5afbdf3bda1a09d68f65181ae3a43ec71079820` |
| `causal_conv1d` **1.6.1** | github.com/Dao-AILab/causal-conv1d | v1.6.1 | `c51519fb10d92ffa257c1982e1831d3243454b2f` |

Build recipe (both packages; run inside the activated env):
```bash
# Host compiler: CUDA-12.1 nvcc REJECTS gcc>12. If system gcc is 13+, use a gcc-12 toolchain
# (e.g. conda: `conda install -c conda-forge gxx_linux-64=12 gcc_linux-64=12`) and point CC/CXX at it.
export CC=<gcc-12>  CXX=<g++-12>  CUDAHOSTCXX=$CXX
export CUDA_HOME=$CONDA_PREFIX  CUDACXX=$CONDA_PREFIX/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.6"   # set to YOUR GPU arch — see table below
git clone --depth 1 -b v1.6.1 https://github.com/Dao-AILab/causal-conv1d  # then v2.3.1 state-spaces/mamba
cd causal-conv1d && CAUSAL_CONV1D_FORCE_BUILD=TRUE $CONDA_PREFIX/bin/python -m pip install . --no-build-isolation
cd ../mamba       && MAMBA_FORCE_BUILD=TRUE        $CONDA_PREFIX/bin/python -m pip install . --no-build-isolation
```
Install **non-editable** (`pip install .`, not `-e .`) and via the env's absolute `python -m pip`:
an editable install registers a PEP660 path-finder that can silently shadow/uninstall the real
package, and a foreign `pip` on `PATH` installs into the wrong env (we corrupted this env this way once).
Verify: `python -c "from mamba_ssm import Mamba2; import torch; m=Mamba2(d_model=256,d_state=64,headdim=64).cuda().half(); m(torch.randn(1,32,256,device='cuda').half()); print('OK')"`.

## GPU-architecture caveats
| GPU (example) | arch | `TORCH_CUDA_ARCH_LIST` | precision note |
|---|---|---|---|
| RTX 2080 Ti | Turing **sm_75** | `7.5` | the chunked scan (`mamba_chunk_scan_combined`) **crashes in fp32** (`IndexError: map::at`); wrap forward in `torch.autocast(dtype=float16)` |
| RTX 3080 / A100 | Ampere **sm_86/80** | `8.6` / `8.0` | fp32 runs; we use fp16 anyway to match training (`precision='16-mixed'`) |

The C1 numeric kill-test and closed-loop driver are fp32 on Ampere (matching the diag5 kill-test).

## Provenance
- Pristine upstream MTIL clone HEAD: **`9bea607`** (`Update README.md`), byte-identical to the published repo.
- **SWH archives source, not the compiled CUDA env** — hence the commit pins + build recipe above are the reconstruction path, not the binaries.

## Audited external repos (cross-repository audit, §6.2) — commit pins
| repo | commit | how audited |
|---|---|---|
| `youngjuY/RoboSSM` | `07df6df` (cloned to `RoboSSM_audit/`) | `Longhorn` mixer extracted verbatim; `robossm_killtest.py` Part A reproduces the `AttributeError` |
| `ALRhub/MaIL` | (record commit at archival) | source inspection: `bc_agent.py predict()` re-stacks a window, no recurrent step |
| `lmzpai/roboMamba` | (record commit at archival) | source inspection: action head is one parallel forward; recurrence serves text only |
