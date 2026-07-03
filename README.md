# When the Scan and Step Paths Diverge — audit code & reproduction artifact

Code for the reproducibility-and-correctness audit of **MTIL** (a full-history Mamba-2
vision-language-action policy; IEEE RA-L 2025, arXiv:2505.12410).

**Start here → [`artifact_c1_step_audit/README.md`](artifact_c1_step_audit/README.md)** — the
TIER 0/1/2 reproduction guide (TIER 0 needs no GPU and runs in under a minute).

## What this repository shows

- **C1** — the recurrent `step()` controller shipped in the published MTIL code diverges from its
  own sequence-parallel training path (residual-collapse bug): closed-loop success 0.7% (1/150,
  seeded). A ~15-line fix restores a faithful O(T) controller: 96.7% (145/150, Wilson [92.4, 98.6]).
- **Cross-repo** — among four released SSM robot policies examined, none both deploys *and*
  verifies a recurrent rollout path (`artifact_c1_step_audit/robossm_killtest.py` demonstrates
  RoboSSM's recurrent branch is unimplemented dead code, CPU-only).
- **C2** — the corrected O(T) advantage is real at the block level (~1.4×) but invisible
  end-to-end (~1.014×, flat in T) behind the per-frame DINOv2 frontend.
- **C3** — full history is not necessary for high success on the headline benchmark
  (k=16-trained policy: 90.0%, Wilson [84.2, 93.8]).

## Layout

| Path | Contents |
|---|---|
| `artifact_c1_step_audit/` | reproduction guide, kill-test + closed-loop drivers, result dumps, pinned environment (`repro/`) |
| `test/` | evaluation & diagnostic scripts (run from here; includes the **fixed** `mamba_policy_par.py`) |
| `train/` | training / feature-extraction scripts (includes the fixed `mamba_policy_par.py` twin — both copies must match) |
| `figs/` | the C2 efficiency-frontier figure + data |
| `eval_render.sh` | headless EGL rendering environment for MuJoCo |

## Data

The ~4.6 GB checkpoint bundle (weights, scaler, one episode + cached features, SHA256SUMS) is on
Zenodo: **[10.5281/zenodo.21154106](https://doi.org/10.5281/zenodo.21154106)**. Point
`ARTIFACT_ROOT` at the unpacked bundle (see the artifact README).

## Provenance

The audited upstream is `github.com/yulinzhouZYL/MTIL` at HEAD `9bea607` (byte-identical clone;
see `artifact_c1_step_audit/PRISTINE_CLONE_HEAD.txt`). Environment pins and audited-repo commit
hashes are in `artifact_c1_step_audit/repro/ENVIRONMENT.md`.
