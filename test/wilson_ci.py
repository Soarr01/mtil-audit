"""Aggregate the seeded re-eval logs into per-seed + pooled success rates with 95% Wilson CIs.
Counts 'Success=True' lines per log (each = one rollout that reached env_max_reward)."""
import glob, math, os, re

Z = 1.96  # 95%

def wilson(k, n):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + Z*Z/n
    center = (p + Z*Z/(2*n)) / den
    half = Z*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n)) / den
    return p, max(0, center-half), min(1, center+half)

def count_log(path):
    txt = open(path, errors='ignore').read()
    succ = len(re.findall(r'Success=True', txt))
    n = len(re.findall(r'Rollout \d+:', txt))
    return succ, n

D = os.path.dirname(os.path.abspath(__file__))
for task, pat in [('transfer_cube', 'eval_transfer_seed*.log'), ('insertion', 'eval_insert_seed*.log')]:
    logs = sorted(glob.glob(os.path.join(D, pat)))
    if not logs:
        print(f'\n[{task}] no logs'); continue
    print(f'\n=== {task} ===')
    tot_k = tot_n = 0
    srs = []
    for lg in logs:
        k, n = count_log(lg)
        if n == 0: continue
        p, lo, hi = wilson(k, n)
        srs.append(p); tot_k += k; tot_n += n
        seed = re.search(r'seed(\d+)', lg).group(1)
        print(f'  seed{seed}: {k}/{n} = {100*p:.1f}%  (Wilson95 [{100*lo:.1f}, {100*hi:.1f}])')
    if tot_n:
        p, lo, hi = wilson(tot_k, tot_n)
        mean = sum(srs)/len(srs)
        std = (sum((x-mean)**2 for x in srs)/len(srs))**0.5
        print(f'  POOLED: {tot_k}/{tot_n} = {100*p:.1f}%  Wilson95 [{100*lo:.1f}, {100*hi:.1f}]')
        print(f'  across-seed mean±std: {100*mean:.1f}% ± {100*std:.1f}%')
