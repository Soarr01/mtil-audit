# Artifact — Reproducibility & correctness audit of MTIL (full-history Mamba VLA)

Reproduces the paper's central findings on the **unmodified published MTIL code** (IEEE RA-L 2025;
`github.com/yulinzhouZYL/MTIL`, arXiv 2505.12410):

- **C1** — the recurrent O(T) inference `step()` shipped in the published code does **not** compute the
  same function as the sequence-parallel training path `forward_sequence` (residual-collapse bug), so
  closed-loop rollouts driven by it fail completely; a ~15-line residual-wiring fix recovers an exact
  O(T) controller (0.7% → 96.7%).
- **§6.2 cross-repo** — among the four released SSM robot policies we examined, none both *uses* and
  *verifies* a recurrent rollout path. RoboSSM's recurrent decode is **unimplemented dead code** that
  raises `AttributeError` on first use (demonstrated here, no GPU).
- **C2 / C3** — the corrected O(T) win is invisible behind the vision frontend; full history is not necessary for high success
  (a k=16-trained policy solves transfer_cube at 90%). Reproduction scripts + result dumps
  included.

## Claim and evidence (C1)

| Level | Pristine / published `step()` | Fixed `step()` (ours) |
|---|---|---|
| Source | residual-collapse bug (`hidden=residual=hidden_out`), byte-identical to github HEAD `9bea607` | canonical two-stream `Block.forward` |
| **Numeric** (per-frame max\|Δ\| vs `forward_sequence`) | **≈ 15** (fp32 **and** fp16) | **≈ 8×10⁻⁴** (< the model's 0.012 GT-error floor) |
| **Closed-loop** (transfer_cube, seeded 3×50) | **0.7%** (1/150; one fluke, 3 sim-crashes) | **96.7%** (145/150, Wilson [92.4, 98.6]) |

(Numeric magnitude is episode-dependent, ≈15–60 across trajectories; always O(10) ≫ the GT-error floor.)

## Reproduce — three tiers by dependency

### TIER 0 — no GPU, no checkpoint, < 1 min (anyone can run these now)
```bash
# (a) SEE the C1 bug in the published source:
git clone https://github.com/yulinzhouZYL/MTIL          # HEAD 9bea607 (see PRISTINE_CLONE_HEAD.txt)
sed -n '/def step/,/return/p' MTIL/train/mamba_policy_par.py   # note 'hidden=residual=hidden_out' (residual collapse)

# (b) RoboSSM recurrent path is dead code (§6.2), CPU-only, random weights:
python robossm_killtest.py        # PART A -> AttributeError: 'Longhorn' has no '_get_states_from_cache'
```

### TIER 1 — C1 reproduction, ~10 min on one GPU (needs the published checkpoint)
Set `ARTIFACT_ROOT` to the downloaded bundle (see **Data** below; contains `checkpoint.ckpt`,
`scaler_transfer.pth`, `mamba_policy_par_pristine.py`, one `episode.hdf5` + its `top.npy`).
**Run from `<MTIL_root>/test`** — the scripts and their imports (`mamba_policy_par`, `sim_env`,
`visualize_episodes`) live there; the copies in this artifact dir are reference/diff only.
```bash
cd <MTIL_root>/test
export ARTIFACT_ROOT=/path/to/bundle  PYTHONPATH=<MTIL_root>  CUDA_VISIBLE_DEVICES=<free>
# Numeric kill-test (no rendering): pristine ≈15 (BROKEN), fixed ≈8e-4 — fp32 and fp16
python audit_upstream_step.py
# Closed-loop (seeded): pristine 0.7% vs fixed ~96–100%; writes results_step_*_seed*.json
source <MTIL_root>/eval_render.sh
SEED=0 PRISTINE=1 python evaluate_step_fs.py 50    # published step()  -> 0.7% (1/150)
SEED=0            python evaluate_step_fs.py 50    # fixed step()      -> ~96%
```
The certified number is the seeded **3×50 = 96.7% [92.4, 98.6]** (seeds 0/1/2 → 96/98/96%); the small
N=10 invocation is a smoke-test. Our own per-rollout dumps are in `results/` for diffing.

### TIER 2 — C2 / C3 supporting results (run from `<MTIL_root>/test`)
```bash
cd <MTIL_root>/test
PYTHONPATH=<MTIL_root> CUDA_VISIBLE_DEVICES=<free> python profile_oT.py   # §7 latency/speedup/peak-mem vs T
# -> figs/profile_oT.json; cf results/c2_profile_oT.json (vision ~189ms, speedup ~1.014x flat)
# §8.1 k=16 windowed-TRAINED: needs the version_16 (windowed-trained) checkpoint — NOT in the base bundle.
# Train it (train/train_par_windowed.py) or diff against the committed per-seed counts:
SEED=0 python evaluate_fs_windowed.py 50 16 <version_16_ckpt_dir>
# -> ~90%; cf results/c3_k16_windowed_results.json (per-seed 46/43/46 = 135/150 = 90.0%, Wilson [84.2,93.8])
```

## Files
- `audit_upstream_step.py` — **numeric kill-test**: pristine vs fixed `MambaPolicy` from one checkpoint, per-frame divergence vs `forward_sequence` (fp32+fp16). Honors `ARTIFACT_ROOT`/`SEED`.
- `evaluate_step_fs.py` — **closed-loop** rollout; `PRISTINE=1` drives the published `step()`; `SEED`/`ARTIFACT_ROOT` configurable; writes per-rollout JSON.
- `robossm_killtest.py` — **§6.2 second-repo probe**: instantiates RoboSSM's deployed `Longhorn` mixer (verbatim, random weights) and shows its recurrent decode raises `AttributeError` (Part A, CPU). Part B (parallel≡ref) is optional and self-skips without the longhorn CUDA build.
- `cueswap_diag.py` — the §8.3 delayed-cue counterfactual probe.
- `step_fix.diff` — the exact fix (`git diff` of `train/`+`test/` `mamba_policy_par.py`; the `MambaPolicy.step()` residual hunk is C1).
- `c2c3/` — `profile_oT.py` (C2), `evaluate_fs_windowed.py` + `train_par_windowed.py` (C3 k=16).
- `results/` — `c2_profile_oT.json`, `c3_k16_windowed_results.json` (per-seed counts), and the C1 `results_step_*.json` dumps you generate.
- `repro/` — `requirements-lock.txt`, `environment.yml`, **`ENVIRONMENT.md`** (from-source build recipe for mamba-ssm 2.3.1 / causal-conv1d 1.6.1 + GPU-arch caveats + commit pins).
- `PRISTINE_CLONE_HEAD.txt` — provenance of the clean upstream clone (`9bea607`).

## Data / checkpoint bundle
TIER 1 needs a trained checkpoint + scaler + one feature episode. These are published as a
self-contained bundle (**≈4.6 GB**: weights-only `checkpoint.ckpt` 1.62 GB, `scaler_transfer.pth`, one
`episode.hdf5` 0.37 GB + its `top.npy` 2.64 GB, the pristine `mamba_policy_par_pristine.py`) —
**DOI: [10.5281/zenodo.21154106](https://doi.org/10.5281/zenodo.21154106)**. Point `ARTIFACT_ROOT` at the unpacked bundle. No 123 GB
feature cache is required. (For the closed-loop test alone, only the checkpoint + scaler are needed, ~1.6 GB;
the two episode files are for the offline numeric kill-test. The C3 k=16 checkpoint is *not* in the base
bundle — TIER 2 trains it or diffs against `results/c3_k16_windowed_results.json`.)

## Archival
The checkpoint bundle is published at **DOI [10.5281/zenodo.21154106](https://doi.org/10.5281/zenodo.21154106)** (Zenodo; includes SHA256SUMS). This directory + the pristine clone are archived via **Software Heritage** (`save code now`); SWHID: **`<to be filled after SWH ingest>`**. Cite alongside arXiv 2505.12410.
