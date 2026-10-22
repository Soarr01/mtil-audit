# Run records for "Auditing Full-History Robot Policies: An MTIL Case Study"

Every number in the paper's context, RMBench, encoder and timing results can be recomputed from the
files below. No checkpoints or datasets are stored here (checkpoints from the earlier audit are on
Zenodo, see the top-level README). Success is closed-loop success: reward 4 on ALOHA cube transfer,
stage 2 on RMBench Put Back Block.

## Layout

| Path | Contents |
|---|---|
| `runs/<run>/history.json` | per-epoch training/validation losses; checkpoint selection = lowest **unrounded** `val_epoch_loss` |
| `runs/<run>/checkpoint_files.txt` | the top-5 checkpoint files kept by training (file names carry a 4-decimal rounded loss) |
| `runs/<run>/train_summary.log` | run header (task, mode, seed, k, W, epochs) and per-epoch lines |
| `runs/<run>/eval_seed{0,1,2}.log` | original cube-transfer evaluations (3 x 50 rollouts) |
| `transfer_cube_evals/best_*/` | re-evaluations at the true minimum-validation-loss checkpoint (8 runs whose earlier selection used a rounded-loss tie) |
| `transfer_cube_evals/noens_*/` | temporal ensembling OFF (`NO_ENSEMBLE=1`): `noens_k1_*` and `noens_full_*` via the parallel scan, `noens_step_full_*` via the fixed recurrent controller |
| `rmbench_evals/<eval>/diag_seed*.jsonl` | per-episode diagnostics: `orig_mat`, `nearest_mat` (final), `max_stage`, `success` |
| `rmbench_evals/<eval>/eval_seed*.log.gz` | RMBench evaluation logs (100 expert-checked episodes per evaluation seed) |
| `timing/timing_{step,fs}_{amp,fp32}_seed0.json` | per-tick stopwatch records (5 rollouts x 400 ticks each) |
| `profile/profile_horizon_gpu0_3080.json` | component latencies vs history length (medians of 20 trials) |
| `checker/scan_step_equiv_*.json` | scan-versus-step checker outputs (per-frame max abs difference) |
| `c3_summary.json` | per-training-seed cube-transfer summary (`test/c3_summarize.py`) |

## Run names

- `full_s{1,2}`, `win_k{1,4,16,64}_w{W}_s{seed}[_e150]`: cube-transfer runs from `train/train_c3.py`
  (seed-42 full history and k=16 come from the original audit: `lightning_logs/version_6`, `version_16`).
- `vits14_full_s42`, `vitb14_full_s42`: full history with a frozen ViT-S/14 or ViT-B/14 encoder.
- `rmbench_put_back_block_{full,win_k16_w16}_s{42,1}`: RMBench runs; `*_e150_fv` = 150 epochs with a fixed
  validation window bank (`--fixed_val`; these two runs used the earlier routine that restored only the
  CPU RNG state after validation — see `FixedValWindows` in `train/train_c3.py`).

## RMBench evaluations and hosts

| Evaluation dir | Checkpoint | Host |
|---|---|---|
| `pbb_full_s42`, `pbb_win16_s42` | seed 42, 150 / 45 epochs | A |
| `pbb_full_s42_hostB` | seed-42 full history (same checkpoint as `pbb_full_s42`) | B |
| `pbb_win16_s42_diag`, `pbb_win16_s42_rep2` | seed-42 45-epoch windowed, evaluation seed 0, runs 1 and 2 | B |
| `pbb_full_s1`, `pbb_win16_s1` | seed 1, 150 / 45 epochs | B |
| `pbb_win16_s{42,1}_e150fv` | 150-epoch fixed-validation windowed controls | B |

`pbb_win16_s42/eval_seed0` (host A) predates the diagnostic log, so it has no `diag_seed0.jsonl`.
Directories with `CRASHED_RUN_EXCLUDED.txt` had an evaluation that crashed (out of GPU memory) and was rerun;
the crashed attempt is not used anywhere.

## Mat indices

`0 = left, 1 = right, 2 = front, 3 = back`. Stage 1 = button pressed with the block at the centre;
stage 2 = block back on its source mat with the gripper open (success).
