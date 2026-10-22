"""
profile_horizon.py — resubmission, C2 extension: WHERE does the O(T) recurrent path start to matter?
§7 measured T<=400 (the benchmark's horizon), where the re-encode block cost is flat (~launch-bound).
Here the same two block paths as profile_oT.py are timed out to long horizons, plus the per-frame vision
frontend for three DINOv2 sizes (ViT-S/B/L, each + its adapter), so the crossover can be read off directly:
  per tick at history t:   re-encode = t_vis + fs_blocks(t)      recurrent = t_vis + t_step
  per episode of length T: sum_t re-encode(t)  vs  T * recurrent
Weights are random (latency does not depend on values). fp16 autocast for adapter+blocks, DINOv2 fp32 (§7 setup).
Usage: PYTHONPATH=<MTIL> python profile_horizon.py      -> mtil_data/figs/profile_horizon.json
"""
import os, json, time, numpy as np, torch
import mamba_policy_par as MP
import mamba_policy_enc as ME

GRID = [1, 64, 256, 400, 512, 1024, 2048, 4096, 8192, 16384]
TRIALS = int(os.environ.get('TRIALS', '20')); WARM = 5
cfg = MP.MambaConfig(); cfg.camera_names = ['top']; cfg.embed_dim = 2048; cfg.lowdim_dim = 14
cfg.d_model = 2048; cfg.action_dim = 14; cfg.sum_camera_feats = False; cfg.num_blocks = 4
kw = dict(camera_names=cfg.camera_names, embed_dim=cfg.embed_dim, lowdim_dim=cfg.lowdim_dim, d_model=cfg.d_model,
          action_dim=cfg.action_dim, sum_camera_feats=cfg.sum_camera_feats, num_blocks=cfg.num_blocks, mamba_cfg=cfg)
torch.manual_seed(0)
policy = MP.MambaPolicy(**kw).cuda().eval()


def timed(fn, trials=TRIALS, warmup=WARM):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
    ts = np.array(ts); return float(np.median(ts)), float(ts.mean()), float(ts.std(ddof=1))


@torch.no_grad()
def fs_blocks(x, t):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        xr = x[:, :t]; residual = None
        for blk in policy.blocks: xr, residual = blk(xr, residual)
        return policy.out_proj(xr[:, -1])


@torch.no_grad()
def step_block(x, st):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        hidden = x[:, 0]; residual = None
        for i, blk in enumerate(policy.blocks):
            conv_st, ssm_st = st[i]
            residual = hidden if residual is None else residual + hidden
            hln = blk.norm(residual.to(blk.norm.weight.dtype)).half()
            y, _, _ = blk.mixer.step(hln.unsqueeze(1), conv_st, ssm_st); y = y.squeeze(1)
            residual = y + residual
            hidden = blk.mlp(blk.norm2(residual.to(blk.norm2.weight.dtype)).half())
        return policy.out_proj(hidden)


out = {'gpu': torch.cuda.get_device_name(0), 'trials': TRIALS, 'grid': GRID, 'vision': {}, 'fs_blocks': {}, 'peak_gb': {}}
# vision frontends (1 frame, 480x640): DINOv2 fp32 + adapter fp16
img = torch.rand(1, 3, 480, 640, device='cuda')
for enc in ['vits14', 'vitb14', 'vitl14']:
    p = policy if enc == 'vitl14' else ME.make_policy(MP, enc, **kw).cuda().eval()
    @torch.no_grad()
    def vis(p=p):
        d = p.shared_backbone(img)
        with torch.autocast(device_type='cuda', dtype=torch.float16): return p.process_vision_chunk(d)
    med, mean, sd = timed(vis, trials=max(10, TRIALS // 2), warmup=3)
    out['vision'][enc] = {'median_ms': med, 'mean_ms': mean, 'sd_ms': sd}
    print(f'vision {enc}: median {med:.2f} ms (mean {mean:.2f} ± {sd:.2f})', flush=True)
    if p is not policy: del p; torch.cuda.empty_cache()

x = torch.randn(1, max(GRID), cfg.d_model, device='cuda', dtype=torch.float16)
st = [tuple(blk.mixer.allocate_inference_cache(1, max_seqlen=1, dtype=torch.float16)) for blk in policy.blocks]
med, mean, sd = timed(lambda: step_block(x, st))
out['step_block'] = {'median_ms': med, 'mean_ms': mean, 'sd_ms': sd}
print(f'step block (recurrent, constant): median {med:.2f} ms', flush=True)
for t in GRID:
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m_, mean, sd = timed(lambda: fs_blocks(x, t), trials=TRIALS if t <= 4096 else 10)
        out['fs_blocks'][t] = {'median_ms': m_, 'mean_ms': mean, 'sd_ms': sd}
        out['peak_gb'][t] = torch.cuda.max_memory_allocated() / 2**30
        print(f'fs_blocks t={t:6d}: median {m_:8.2f} ms  peak {out["peak_gb"][t]:.2f} GB', flush=True)
    except torch.OutOfMemoryError:
        out['fs_blocks'][t] = 'OOM'; print(f'fs_blocks t={t}: OOM', flush=True); torch.cuda.empty_cache()

# derived: per-tick speedup at history t, and whole-episode speedup for horizon T (linear interpolation of fs(t))
ts = [t for t in GRID if isinstance(out['fs_blocks'][t], dict)]
fsm = np.array([out['fs_blocks'][t]['median_ms'] for t in ts]); stp = out['step_block']['median_ms']
der = {}
for enc, v in out['vision'].items():
    tv = v['median_ms']
    tick = {t: (tv + f) / (tv + stp) for t, f in zip(ts, fsm)}
    ep = {}
    for T in ts:
        tt = np.arange(1, T + 1); f_all = np.interp(tt, ts, fsm)
        ep[T] = float((T * tv + f_all.sum()) / (T * (tv + stp)))
    der[enc] = {'per_tick_speedup': tick, 'episode_speedup': ep}
    print(f'{enc}: per-tick speedup ' + ', '.join(f't={t}:{s:.2f}x' for t, s in tick.items()), flush=True)
out['derived'] = der
p = os.environ.get('OUT_JSON', '/data/nas07_new/PersonalData/phuocthien/mtil_data/figs/profile_horizon.json')
json.dump(out, open(p, 'w'), indent=1); print('wrote', p, flush=True)
