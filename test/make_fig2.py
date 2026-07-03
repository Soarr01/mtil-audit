"""
make_fig2.py — Figure 2: the C2 efficiency frontier (per-step latency / speedup / peak-mem vs horizon T),
from the authoritative profile_oT.json (20 trials/point, quiet host). Renders the paper's first
quantitative figure. matplotlib Agg, no GPU.
"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

J = json.load(open('/data/nas07_new/PersonalData/phuocthien/mtil_data/figs/profile_oT.json'))
T = np.array(J['grid'])
t_vis = J['t_vision']; t_step = J['t_step_block']
fs = np.array(J['fs_blocks_ms'])                 # O(T^2) re-encode block cost (flat ~9.8ms)
o2 = np.array(J['oT2_total_ms'])                 # vision + fs_blocks  (O(T^2) per-step total)
o1 = np.array(J['oT_total_ms'])                  # vision + step       (O(T)   per-step total)
mem = np.array(J['peak_gb'])
speedup = o2 / o1

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

# Panel A: per-step latency vs T (log-y) — vision dwarfs both SSM modes, all flat in T
ax[0].plot(T, np.full_like(T, t_vis, dtype=float), 'o-', color='#c0392b', label=f'vision frontend (DINOv2), {t_vis:.0f} ms')
ax[0].plot(T, o2, 's-', color='#2980b9', label='O(T²) re-encode  (vision+blocks)')
ax[0].plot(T, o1, '^-', color='#27ae60', label='O(T) step  (vision+step)')
ax[0].plot(T, fs, 'd--', color='#8e44ad', label=f'O(T²) blocks only (~{fs.mean():.1f} ms)')
ax[0].plot(T, np.full_like(T, t_step, dtype=float), 'v--', color='#16a085', label=f'O(T) step block only (~{t_step:.1f} ms)')
ax[0].set_yscale('log'); ax[0].set_xlabel('horizon T (frames)'); ax[0].set_ylabel('per-step latency (ms, log)')
ax[0].set_title('(a) latency — vision dominates, flat in T'); ax[0].legend(fontsize=6.5, loc='center right')
ax[0].grid(True, which='both', alpha=0.25)

# Panel B: end-to-end speedup O(T^2)->O(T) vs T (flat ~1.014x)
ax[1].plot(T, speedup, 'o-', color='#2c3e50')
ax[1].axhline(1.0, color='gray', ls=':', lw=1)
ax[1].set_ylim(0.98, 1.10); ax[1].set_xlabel('horizon T (frames)'); ax[1].set_ylabel('end-to-end speedup ×')
ax[1].set_title(f'(b) O(T²)→O(T) speedup ≈ {speedup.mean():.3f}× — invisible')
ax[1].annotate(f'block-level {fs.mean()/t_step:.1f}× hidden behind vision',
               xy=(T[-1], speedup[-1]), xytext=(T[1], 1.06), fontsize=7,
               arrowprops=dict(arrowstyle='->', color='gray'))
ax[1].grid(True, alpha=0.25)

# Panel C: peak memory vs T (flat ~7.53-7.55 GB)
ax[2].plot(T, mem, 'o-', color='#d35400')
ax[2].set_ylim(7.0, 8.0); ax[2].set_xlabel('horizon T (frames)'); ax[2].set_ylabel('peak GPU memory (GB)')
ax[2].set_title(f'(c) peak memory flat ({mem.min():.2f}→{mem.max():.2f} GB)')
ax[2].grid(True, alpha=0.25)

plt.tight_layout()
OUT = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/figs'
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig2_efficiency_frontier.png', dpi=150, bbox_inches='tight')
fig.savefig(f'{OUT}/fig2_efficiency_frontier.pdf', bbox_inches='tight')
print('wrote', f'{OUT}/fig2_efficiency_frontier.png/.pdf')
print(f'speedup mean {speedup.mean():.4f}x  range [{speedup.min():.4f}, {speedup.max():.4f}]  block {fs.mean()/t_step:.2f}x')
