"""
make_fig_horizon_ral.py — RA-L version of make_fig_horizon.py (larger fonts, component-accurate labels). Figure for the resubmission C2 extension (profile_horizon.py, quiet GPU0 RTX 3080).
(a) per-tick Mamba-block latency vs history length t: parallel re-encode fs_blocks(t) vs recurrent step (flat),
    with the three vision-frontend latencies as horizontal references;
(b) per-tick end-to-end speedup (t_vis + fs(t)) / (t_vis + t_step) vs t for ViT-S/B/L, with the benchmark
    horizon T=400 marked.
Also overlays the direct rollout stopwatch point (time_rollout.py, ViT-L, T<=400) if available.
"""
import json, os, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

J = '/data/nas07_new/PersonalData/phuocthien/mtil_data/figs/profile_horizon_gpu0_3080.json'
OUT = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/latex_ral/figs/fig_horizon_crossover'
TIM = '/data/nas07_new/PersonalData/phuocthien/mtil_data/timing'
d = json.load(open(J))
ts = [int(t) for t, v in d['fs_blocks'].items() if isinstance(v, dict)]
fs = np.array([d['fs_blocks'][str(t)]['median_ms'] for t in ts]); stp = d['step_block']['median_ms']
vis = {e: v['median_ms'] for e, v in d['vision'].items()}
lab = {'vits14': 'ViT-S/14', 'vitb14': 'ViT-B/14', 'vitl14': 'ViT-L/14 (MTIL)'}
plt.rcParams.update({'font.size': 8})
col = {'vits14': 'tab:green', 'vitb14': 'tab:orange', 'vitl14': 'tab:blue'}

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
a = ax[0]
a.plot(ts, fs, 'o-', color='k', ms=4, label='parallel scan over $t$ cached frames')
a.axhline(stp, color='k', ls='--', label=f'recurrent step ({stp:.1f} ms)')
for e in ['vitl14', 'vitb14', 'vits14']:
    a.axhline(vis[e], color=col[e], ls=':', lw=1.5, label=f'DINOv2+adapter {lab[e]} ({vis[e]:.1f} ms)')
a.axvline(400, color='gray', lw=0.8)
a.set_xscale('log', base=2); a.set_xlabel('history length t (frames)'); a.set_ylabel('latency per tick (ms)')
a.set_title('(a) sequence model (4 Mamba blocks + output proj.)', fontsize=8); a.legend(fontsize=6.5, loc='upper left'); a.grid(alpha=0.3)

b = ax[1]
for e in ['vits14', 'vitb14', 'vitl14']:
    sp = (vis[e] + fs) / (vis[e] + stp)
    b.plot(ts, sp, 'o-', color=col[e], label=lab[e])
tim = sorted(glob.glob(os.path.join(TIM, 'timing_*_amp_seed0.json')))
if len(tim) == 2:
    s = {json.load(open(f))['summary']['mode']: json.load(open(f))['summary'] for f in tim}
    if 'step' in s and 'fs' in s:
        r = s['fs']['tick_ms_mean'] / s['step']['tick_ms_mean']
        b.plot([400], [r], marker='*', ms=14, color='tab:red', ls='none', label=f'rollout stopwatch, mean tick ratio ({r:.3f}$\\times$)')
b.axhline(1, color='gray', lw=0.8); b.axvline(400, color='gray', lw=0.8)
b.set_xscale('log', base=2); b.set_xlabel('history length t (frames)'); b.set_ylabel('speedup of recurrent step')
b.set_title('(b) composed encoder+sequence-model speedup', fontsize=8); b.legend(fontsize=6.5); b.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=150)
print('wrote', OUT + '.pdf/.png')
