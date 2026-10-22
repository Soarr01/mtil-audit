"""
make_fig_context.py — RA-L figure: cube-transfer success per TRAINING seed, context length x execution protocol.
Each point = one trained checkpoint evaluated on 3x50 rollouts (true-min val-loss checkpoint); horizontal line = mean over seeds.
TE = ACT temporal ensembling (default execution); no TE = execute the first action of the current chunk only.
Numbers are the success counts out of 150 from train/c3_runs/fix_evals/* and test/c3_summarize.py (seeds 42/1/2).
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/latex_ral/figs/fig_context'
TE = {'1': [143, 146, 143], '4': [136, 141, 130], '16': [130, 136, 146], '64': [149], 'full': [148, 148, 149]}
NOTE = {'1': [74, 77, 85], 'full': [122, 136, 139]}          # no TE; full history through the fixed recurrent controller
labels = ['1', '4', '16', '64', 'full']
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(3.45, 2.1))
for i, k in enumerate(labels):
    for d, (src, col, mk, face) in zip((-0.14, 0.14), ((TE, 'tab:blue', 'o', 'tab:blue'), (NOTE, 'tab:red', 's', 'none'))):
        if k not in src:
            continue
        v = np.array(src[k]) / 1.5
        ax.hlines(v.mean(), i + d - 0.11, i + d + 0.11, color=col, lw=1.6, zorder=2)   # across-seed mean
        jit = np.linspace(-0.05, 0.05, len(v)) if len(v) > 1 else [0]
        ax.scatter(i + d + np.array(jit), v, s=14, marker=mk, facecolors=face, edgecolors=col, lw=0.9, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([{'64': '$k$=64\n(1 seed)', 'full': 'full'}.get(k, f'$k$={k}') for k in labels], fontsize=7)
ax.set_ylabel('success (%)', fontsize=7); ax.set_ylim(40, 101); ax.tick_params(labelsize=7)
ax.grid(axis='y', alpha=0.3)
ax.scatter([], [], marker='o', color='tab:blue', s=14, label='with TE')
ax.plot([], [], color='gray', lw=1.6, label='seed mean')
ax.scatter([], [], marker='s', facecolors='none', edgecolors='tab:red', s=14, label='without TE')
ax.legend(fontsize=6.5, loc='lower left', bbox_to_anchor=(0.25, 0.02), frameon=True)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
print('wrote', OUT)
