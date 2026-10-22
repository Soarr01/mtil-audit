"""
c3_summarize.py — aggregate the resubmission C3 runs (training-seed replication + context-length curve).
Per condition (full / k=1 / k=4 / k=16 / k=64): per TRAINING seed the 3x50 eval-seed success, then
  * pooled Wilson-95% over all rollouts of all training seeds (eval-variability view, as in the paper),
  * mean ± sd across training seeds and a t-interval over training seeds (training-variability view: the
    comparison between training REGIMES the reviewer asked for),
  * two-proportion z (pooled) and Welch t (training-seed level) of each windowed condition vs full history.
Original paper checkpoints (seed 42) are included from their committed logs:
  full  = train/lightning_logs/version_6, evaluate_step_fs.py seeds 0/1/2 -> 48/49/48 of 50 (train/c1_seeded2.log)
  k=16  = train/lightning_logs/version_16, evaluate_fs_windowed.py seeds 0/1/2 -> 46/43/46 of 50 (paper §8.1: 92/86/92%)
Usage: python c3_summarize.py            -> prints a table, writes mtil_data/c3/c3_summary.json
"""
import os, re, glob, json, math
import numpy as np
from scipy import stats

R = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/c3_runs'
OUT = '/data/nas07_new/PersonalData/phuocthien/mtil_data/c3'; os.makedirs(OUT, exist_ok=True)
N = 50
ORIG = {('full', 42): [48, 49, 48], ('16', 42): [46, 43, 46]}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def parse(d):
    succ = []
    for s in (0, 1, 2):
        L = os.path.join(d, f'eval_seed{s}.log')
        if not os.path.exists(L): return None
        t = open(L, errors='ignore').read()
        m = re.search(r'SUCCESS RATE: ([\d.]+)% over (\d+)', t) or re.search(r'=== k=\d+: SUCCESS ([\d.]+)% over (\d+)', t)
        if not m: return None
        succ.append(round(float(m.group(1)) * int(m.group(2)) / 100))
    return succ


cond = {}   # cond -> {train_seed: [succ per eval seed]}
for (c, ts), v in ORIG.items(): cond.setdefault(c, {})[ts] = v
for d in sorted(glob.glob(os.path.join(R, '*'))):
    tag = os.path.basename(d)
    if tag.startswith('smoke'): continue
    m = re.match(r'(?:(vit[sb]14)_)?(?:full_s(\d+)|win_k(\d+)_w(\d+)_s(\d+))(?:_e(\d+))?$', tag)
    if not m: continue
    enc, fs, k, w, ws, ep = m.groups()
    # condition key: k for the standard windowed recipe (k*W = 256, 45 epochs); otherwise k/W/epochs made explicit
    std_w = {1: 256, 4: 64, 16: 16, 64: 4}
    if fs: c = 'full'
    elif int(w) == std_w.get(int(k)) and ep is None: c = k
    else: c = f"{k}(W={w}{', ' + ep + 'ep' if ep else ''})"
    if enc: c = f"{enc}:{c}"
    ts = int(fs or ws); v = parse(d)
    if v: cond.setdefault(c, {})[ts] = v

rows = []; summ = {}
full_pool = sum(sum(v) for v in cond.get('full', {}).values()); full_n = N * 3 * len(cond.get('full', {}))
full_ts = [100 * sum(v) / (3 * N) for v in cond.get('full', {}).values()]
order = sorted(cond, key=lambda c: (':' in c, c != 'full', '(' in c, int(re.match(r'\D*(\d+)', c.split(':')[-1]).group(1)) if re.match(r'\D*(\d+)', c.split(':')[-1]) else 0, c))
for c in order:
    seeds = cond[c]; k = sum(sum(v) for v in seeds.values()); n = N * 3 * len(seeds)
    per = {ts: 100 * sum(v) / (3 * N) for ts, v in sorted(seeds.items())}
    arr = np.array(list(per.values())); lo, hi = wilson(k, n)
    rec = {'condition': c, 'training_seeds': sorted(seeds), 'per_training_seed_pct': per,
           'per_eval_seed_counts': {str(ts): v for ts, v in seeds.items()}, 'pooled_k': k, 'pooled_n': n,
           'pooled_pct': 100 * k / n, 'wilson95': [lo, hi], 'mean_over_training_seeds': float(arr.mean()),
           'sd_over_training_seeds': float(arr.std(ddof=1)) if len(arr) > 1 else None}
    if len(arr) > 1:
        tq = stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr))
        rec['t95_over_training_seeds'] = [float(arr.mean() - tq), float(arr.mean() + tq)]
    if c != 'full' and full_n:
        p1, p2 = k / n, full_pool / full_n; pp = (k + full_pool) / (n + full_n)
        z = (p2 - p1) / math.sqrt(pp * (1 - pp) * (1 / n + 1 / full_n)) if 0 < pp < 1 else float('nan')
        rec['vs_full_pooled_z'] = z; rec['vs_full_pooled_p'] = float(2 * (1 - stats.norm.cdf(abs(z))))
        if len(arr) > 1 and len(full_ts) > 1:
            tt = stats.ttest_ind(full_ts, arr, equal_var=False); rec['vs_full_welch_t'] = float(tt.statistic); rec['vs_full_welch_p'] = float(tt.pvalue)
    summ[c] = rec
    print(f"{c:>12} | train seeds {sorted(seeds)} | per-seed {', '.join(f'{p:.1f}' for p in per.values())} | "
          f"pooled {100*k/n:.1f}% ({k}/{n}) Wilson [{lo:.1f},{hi:.1f}]"
          + (f" | mean±sd {arr.mean():.1f}±{arr.std(ddof=1):.1f}" if len(arr) > 1 else '')
          + (f" | vs full z={rec['vs_full_pooled_z']:.2f} p={rec['vs_full_pooled_p']:.3f}" if 'vs_full_pooled_z' in rec else '')
          + (f" welch p={rec['vs_full_welch_p']:.3f}" if 'vs_full_welch_p' in rec else ''))
json.dump(summ, open(os.path.join(OUT, 'c3_summary.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'c3_summary.json'))
