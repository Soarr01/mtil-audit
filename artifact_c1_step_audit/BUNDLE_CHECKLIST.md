# Publish & archive checklist (for the cert deliverable) — run by the author

The reproduction artifact is **executable only with a trained checkpoint + data**, which can't live in a
git/SWH source archive. This checklist publishes a self-contained bundle to a DOI host and archives the
code on Software Heritage, then wires the identifiers into the paper. Everything that can be prepared
offline is **already staged in `bundle/`**; the steps below are the upload/trigger actions that need
*your* account.

## What is already staged (`artifact_c1_step_audit/bundle/`)
| file | what | size |
|---|---|---|
| `checkpoint.ckpt` | transfer_cube weights-only (optimizer stripped 4.4 GB → ~1.6 GB) | ~1.6 GB |
| `scaler_transfer.pth` | per-joint normalizer | ~19 KB |
| `mamba_policy_par_pristine.py` | the published (broken) policy, for the side-by-side numeric test | small |
| `episode.hdf5` | one transfer_cube episode (episode_19) for the numeric kill-test | ~0.37 GB |
| `top.npy` | that episode's cached DINOv2 features `[T,1024,35,46]` | ~2.6 GB |

Total ≈ **4.6 GB**. (For the closed-loop test alone, only `checkpoint.ckpt` + `scaler_transfer.pth` are
needed — vision is computed live; the two episode files are only for the offline numeric kill-test.)
Optional for C3 TIER-2: add the k=16 `version_16` checkpoint (strip the same way → `checkpoint_k16.ckpt`).

## 1. DOI host (Zenodo or HuggingFace) — gives a citable DOI
```bash
cd artifact_c1_step_audit && zip -r mtil_repro_bundle.zip bundle/      # or upload the dir directly
```
- **Zenodo**: New upload → drag `mtil_repro_bundle.zip` → set title "MTIL C1 reproduction bundle
  (checkpoint + scaler + one episode)", license, link to arXiv 2505.12410 → **Publish** → copy the DOI
  (`10.5281/zenodo.XXXXXXX`).
- (HuggingFace Datasets/Models is an alternative; the DOI/URL plays the same role.)

## 2. Software Heritage — archives the *source* (code, not binaries)
For **both** the pristine upstream clone and this artifact directory:
- Go to `https://archive.softwareheritage.org/save/` → "Save code now" → paste each git origin URL
  (the upstream MTIL repo; and your repo/dir containing `artifact_c1_step_audit/`) → submit.
- Once "ingested", open the snapshot and copy the **SWHID** (`swh:1:snp:...` or `swh:1:dir:...`).
- Record the artifact's own commit hash next to `PRISTINE_CLONE_HEAD.txt` (currently `9bea607` for upstream).

## 3. Wire identifiers into the paper + artifact (then the conditional wording can go)
- `artifact_c1_step_audit/README.md` → replace the two `<to be filled ...>` placeholders (Data DOI; SWHID).
- `paper_tmlr_full.md` → Appendix A and §10/abstract: replace "We recommend archiving …" with the
  resolved DOI + SWHID; this closes the cert's "archived, reconstructable, runnable artifact" requirement.

## 4. Sanity-check the published bundle on a clean path (proves third-party runnability)
```bash
unzip mtil_repro_bundle.zip -d /tmp/check && export ARTIFACT_ROOT=/tmp/check/bundle
cd MTIL/test && PYTHONPATH=<MTIL_root> CUDA_VISIBLE_DEVICES=<free> python audit_upstream_step.py   # pristine ≈15, fixed ≈8e-4
source <MTIL_root>/eval_render.sh && SEED=0 PRISTINE=1 python evaluate_step_fs.py 10               # 0%
                                     SEED=0            python evaluate_step_fs.py 10               # ~100%
```
If these pass reading only from `$ARTIFACT_ROOT`, the cert's runnable-artifact criterion is met.
