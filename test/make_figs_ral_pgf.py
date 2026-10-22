"""
make_figs_ral_pgf.py — RA-L figures 2 and 3 as pgfplots code, styled with the same rules as Fig. 1
(diagram-design editorial system: ink/muted palette, ONE accent for the focal series, faint gridlines,
direct labels instead of floating legends, zero-based axes where magnitude matters).
Writes latex_ral/figs/fig_context.tex and latex_ral/figs/fig_horizon.tex (input by main.tex).
Data: success counts from train/c3_runs (see test/c3_summarize.py, fix_evals) and
mtil_data/figs/profile_horizon_gpu0_3080.json + mtil_data/timing/timing_*_amp_seed0.json.
"""
import json, os
import numpy as np

# paths: MTIL_ROOT (repo root, default = parent of test/) and MTIL_DATA (data root, default = <repo>/../../mtil_data)
ROOT = os.environ.get('MTIL_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get('MTIL_DATA', os.path.normpath(os.path.join(ROOT, '..', '..', 'mtil_data')))
OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'latex_ral', 'figs'))
PROF = os.path.join(DATA, 'figs', 'profile_horizon_gpu0_3080.json')
TIM = os.path.join(DATA, 'timing')

# ------------------------------------------------------------------ Fig. 2: context x execution dumbbell
# successes out of 150 per training seed (seeds 42/1/2); TE = temporal ensembling
TE = {'full': [148, 148, 149], '64': [149], '16': [130, 136, 146], '4': [136, 141, 130], '1': [143, 146, 143]}
NOTE = {'full': [122, 136, 139], '1': [74, 77, 85]}   # no TE; full history via the fixed recurrent controller
rows = [('full', 'full history'), ('64', '$k{=}64$ (1 seed)'), ('16', '$k{=}16$'), ('4', '$k{=}4$'), ('1', '$k{=}1$')]
pct = lambda v: [x / 1.5 for x in v]


def stack(vals, near=1.0, step=0.11):
    """keep each seed's exact x; stack coincident/near-coincident seeds vertically so all are visible"""
    out, placed = [], []
    for v in sorted(vals):
        k = sum(1 for u in placed if abs(u - v) < near)
        placed.append(v); out.append((v, k * step))
    return out

L = []
L.append(r"""\begin{tikzpicture}[execute at begin picture=\sansmath]
\begin{axis}[width=0.86\columnwidth, height=4.6cm, scale only axis=false,
  xmin=0, xmax=100, ymin=0.45, ymax=5.55, axis line style={draw=none},
  xtick={0,20,40,60,80,100}, ytick={1,2,3,4,5},
  yticklabels={""" + ','.join('{' + lab + '}' for _, lab in reversed(rows)) + r"""},
  tick style={draw=none}, xmajorgrids, grid style={ddink!10, line width=0.4pt},
  ticklabel style={font=\sffamily\scriptsize, text=ddmuted},
  yticklabel style={font=\sffamily\scriptsize, text=ddink},
  xlabel={success over 150 rollouts per training seed (\%)}, xlabel style={font=\sffamily\scriptsize, text=ddmuted, yshift=3pt},
  clip=false]
\draw[ddink!35, line width=0.5pt] (axis cs:0,0.45) -- (axis cs:0,5.55);""")
for i, (key, _) in enumerate(reversed(rows)):
    y = i + 1
    te = pct(TE[key]); mte = float(np.mean(te))
    if key in NOTE:
        ne = pct(NOTE[key]); mne = float(np.mean(ne))
        L.append(rf"\draw[ddink!55, line width=0.6pt] (axis cs:{mne:.2f},{y}) -- (axis cs:{mte:.2f},{y});")
        L.append(rf"\node[font=\sffamily\scriptsize\bfseries, text=ddaccent, anchor=east] at (axis cs:{mne - 15:.2f},{y}) {{\textminus{mte - mne:.1f}}};")   # gap label left of the pair, clear of seed dots
        for v, dy in stack(ne):
            L.append(rf"\fill[ddaccent!75] (axis cs:{v:.2f},{y - 0.22 - dy:.3f}) circle (1.0pt);")
        L.append(rf"\draw[ddink, line width=0.5pt, fill=ddaccent] (axis cs:{mne:.2f},{y}) circle (2.1pt);")
        L.append(rf"\node[font=\sffamily\scriptsize, text=ddmuted, left=3.5pt] at (axis cs:{mne:.2f},{y}) {{{mne:.1f}}};")
    for v, dy in stack(te):
        L.append(rf"\fill[ddmuted!80] (axis cs:{v:.2f},{y + 0.22 + dy:.3f}) circle (1.0pt);")
    L.append(rf"\draw[ddmuted, line width=0.9pt, fill=white] (axis cs:{mte:.2f},{y}) circle (2.1pt);")
    L.append(rf"\node[font=\sffamily\scriptsize, text=ddmuted, right=4.5pt] at (axis cs:{mte:.2f},{y}) {{{mte:.1f}}};")
L.append(r"""\end{axis}
% legend: horizontal strip under the plot
\begin{scope}[shift={(0,-1.1)}, font=\sffamily\scriptsize, text=ddmuted]
\draw[ddmuted, line width=0.9pt, fill=white] (0.15,0) circle (2.1pt); \node[anchor=west] at (0.28,0) {with TE (mean)};
\draw[ddink, line width=0.5pt, fill=ddaccent] (2.3,0) circle (2.1pt); \node[anchor=west] at (2.43,0) {without TE (mean)};
\fill[ddmuted!80] (4.75,0.06) circle (1.0pt); \fill[ddaccent!75] (4.85,-0.06) circle (1.0pt); \node[anchor=west] at (4.95,0) {seeds};
\end{scope}
\end{tikzpicture}""")
open(os.path.join(OUT, 'fig_context.tex'), 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ Fig. 3: horizon profile
d = json.load(open(PROF))
ts = sorted(int(t) for t, v in d['fs_blocks'].items() if isinstance(v, dict))
fs = {t: d['fs_blocks'][str(t)]['median_ms'] for t in ts}
R = d['step_block']['median_ms']
V = {e: v['median_ms'] for e, v in d['vision'].items()}
s = {json.load(open(os.path.join(TIM, f)))['summary']['mode']: json.load(open(os.path.join(TIM, f)))['summary']
     for f in ('timing_step_amp_seed0.json', 'timing_fs_amp_seed0.json')}
star = s['fs']['tick_ms_mean'] / s['step']['tick_ms_mean']
coords = lambda f: ' '.join(f'({t},{f(t):.4f})' for t in ts)
enc = [('vitl14', 'ViT-L/14 (MTIL)', 'ddaccent', '1.1pt'), ('vitb14', 'ViT-B/14', 'ddink!80', '0.7pt'), ('vits14', 'ViT-S/14', 'ddink!62', '0.7pt')]

DDAX = r"""width=0.43\textwidth, height=4.1cm, xmode=log, log basis x=2, xmin=1, xmax=16384,
  xtick={1,8,64,512,4096,16384}, xticklabels={1,8,64,512,4k,16k}, axis line style={draw=none}, tick style={draw=none},
  ymajorgrids, grid style={ddink!10, line width=0.4pt}, ticklabel style={font=\sffamily\scriptsize, text=ddmuted},
  xlabel={history length $t$ (frames)}, label style={font=\sffamily\scriptsize, text=ddmuted}, clip=false"""
H = [r"""\begin{tikzpicture}[execute at begin picture=\sansmath]"""]
# (a) sequence-model cost
H.append(r"""\begin{axis}["""+DDAX+r""", name=a, ymin=0, ymax=200, ytick={0,50,100,150,200}, ylabel={ms per tick}, ylabel style={yshift=-4pt}]
\draw[ddink!35, line width=0.5pt] (axis cs:1,0) -- (axis cs:16384,0);
\draw[ddink!25, line width=0.5pt, dashed] (axis cs:400,0) -- (axis cs:400,200);
\node[font=\sffamily\scriptsize, text=ddmuted, fill=white, inner sep=1pt, anchor=south] at (axis cs:400,203) {benchmark $T{=}400$};""")
for e, lab, col, lw in enc:
    H.append(rf"\draw[ddmuted!70, line width=0.5pt, dash pattern=on 1.5pt off 1.5pt] (axis cs:1,{V[e]:.2f}) -- (axis cs:16384,{V[e]:.2f});")
    H.append(rf"\node[font=\sffamily\scriptsize, text=ddmuted, anchor=west] at (axis cs:16384,{V[e] + (5 if e == 'vits14' else 0):.2f}) {{{lab.split(' ')[0]} {V[e]:.1f}}};")
H.append(rf"\addplot[ddink, line width=0.7pt, dashed, domain=1:16384, samples=2] {{{R:.3f}}};")
H.append(rf"\node[font=\sffamily\scriptsize, text=ddink, anchor=west] at (axis cs:16384,{R - 4:.2f}) {{recurrent {R:.1f}}};")
H.append(rf"\addplot[ddaccent, line width=1.1pt, mark=*, mark size=1.3pt] coordinates {{{coords(lambda t: fs[t])}}};")
H.append(rf"\node[font=\sffamily\scriptsize, text=ddaccent, anchor=west] at (axis cs:16384,{fs[16384]:.2f}) {{parallel scan {fs[16384]:.0f}}};")
H.append(r"""\node[font=\sffamily\scriptsize\bfseries, text=ddsoft, anchor=south west] at (rel axis cs:-0.12,1.13) {(a) SEQUENCE-MODEL COST VS.\ HISTORY};
\end{axis}""")
# (b) composed speedup
H.append(r"""\begin{axis}["""+DDAX+r""", at={(a.east)}, anchor=west, xshift=2.4cm, ymin=0.9, ymax=7, ytick={1,2,3,4,5,6,7}, yticklabels={1$\times$,2$\times$,3$\times$,4$\times$,5$\times$,6$\times$,7$\times$}]
\draw[ddink!35, line width=0.5pt] (axis cs:1,1) -- (axis cs:16384,1);
\draw[ddink!25, line width=0.5pt, dashed] (axis cs:400,0.9) -- (axis cs:400,7);""")
for e, lab, col, lw in reversed(enc):
    sp = lambda t, e=e: (V[e] + fs[t]) / (V[e] + R)
    mark = ', mark=*, mark size=1.3pt' if e == 'vitl14' else ''
    H.append(rf"\addplot[{col}, line width={lw}{mark}] coordinates {{{coords(sp)}}};")
    H.append(rf"\node[font=\sffamily\scriptsize, text={'ddaccent' if e == 'vitl14' else 'ddink'}, anchor=west] at (axis cs:16384,{sp(16384):.3f}) {{{lab} {sp(16384):.1f}$\times$}};")
H.append(rf"\addplot[only marks, mark=diamond*, mark size=2.4pt, mark options={{draw=ddink, fill=white, line width=0.6pt}}] coordinates {{(400,{star:.4f})}};")
H.append(rf"\node[font=\sffamily\scriptsize, text=ddink, fill=white, inner sep=1pt, anchor=south east, align=right] at (axis cs:330,1.6) {{rollout mean (AMP)\\$T{{=}}400$: {star:.3f}$\times$}};")
H.append(r"""\node[font=\sffamily\scriptsize\bfseries, text=ddsoft, anchor=south west] at (rel axis cs:-0.12,1.13) {(b) COMPOSED SPEEDUP OF RECURRENCE};
\end{axis}
\end{tikzpicture}""")
open(os.path.join(OUT, 'fig_horizon.tex'), 'w').write('\n'.join(H) + '\n')
print('wrote fig_context.tex, fig_horizon.tex; stopwatch ratio %.4f' % star)
