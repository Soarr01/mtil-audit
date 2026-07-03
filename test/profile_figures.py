"""
profile_figures.py — #22: generate the latency / peak-memory vs horizon-T FIGURES for the
efficiency chapter (Ch. 6). Same measurements as profile_oT.py, plus matplotlib figures + a JSON dump.

Two inference modes per rollout step:
  O(T^2) re-encode (the only correct path before the step() fix): vision(1 frame) + fs_blocks(history t)
  O(T)   fixed step():                                            vision(1 frame) + step_block(constant)
Headline: the per-frame DINOv2 vision frontend dominates; the Mamba block cost is small AND flat in t,
so the O(T)->O(T^2) "win" is ~1.02x (invisible) at manipulation horizons. Figures make this visual.

NOTE: timing precision degrades under CPU/GPU contention; for final publication numbers re-run in a
quiet window (load<8, GPU otherwise idle). The QUALITATIVE shape (vision >> Mamba, Mamba flat) is robust.
"""
import os, glob, re, time, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mamba_policy_par as MP

CKPT_DIR='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_14/checkpoints'
FEAT='/data/nas07_new/PersonalData/phuocthien/mtil_data/insertion_feat/train/episode_0/top.npy'
OUTDIR='/data/nas07_new/PersonalData/phuocthien/mtil_data/figs'; os.makedirs(OUTDIR, exist_ok=True)
GRID=[1,25,50,100,200,300,400]
TRIALS=20; WARMUP=5

config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
ckpt=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)

feat=np.load(FEAT, mmap_mode='r'); Tmax=min(max(GRID), feat.shape[0])
feats=torch.from_numpy(np.array(feat[:Tmax],dtype=np.float32)).cuda()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    cam=policy.process_vision_chunk(feats).float()
obs=torch.randn(1,Tmax,14,device='cuda')

@torch.no_grad()
def build_fused(dtype=torch.float16):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dtype==torch.float16)):
        cff=cam.to(dtype).unsqueeze(0).reshape(Tmax,-1); lf=obs.to(dtype).reshape(Tmax,-1)
        proj=policy.in_proj(cff)
        fused=policy.cross_attn(query=proj.unsqueeze(0),key=lf.unsqueeze(0),value=lf.unsqueeze(0)).squeeze(0)
    return fused.view(1,Tmax,policy.d_model)

def timed(fn, trials=TRIALS, warmup=WARMUP):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(trials): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/trials*1000.0

@torch.no_grad()
def fs_blocks(x, t):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        xr=x[:,:t]; residual=None
        for blk in policy.blocks: xr,residual=blk(xr,residual)
        return policy.out_proj(xr)

@torch.no_grad()
def step_block(x):
    st=[list(blk.mixer.allocate_inference_cache(1,max_seqlen=1,dtype=torch.float16)) for blk in policy.blocks]
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        hidden=x[:,0]; residual=None
        for i,blk in enumerate(policy.blocks):
            conv_st,ssm_st=st[i]
            residual=hidden if residual is None else residual+hidden
            hln=blk.norm(residual.to(blk.norm.weight.dtype)).half()
            y,conv_st,ssm_st=blk.mixer.step(hln.unsqueeze(1),conv_st,ssm_st); y=y.squeeze(1)
            residual=y+residual
            hidden=blk.mlp(blk.norm2(residual.to(blk.norm2.weight.dtype)).half())
        return policy.out_proj(hidden)

@torch.no_grad()
def vision_frontend():
    img=torch.rand(1,3,480,640,device='cuda')
    d=policy.shared_backbone(img)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return policy.process_vision_chunk(d)

x16=build_fused()
t_vis=timed(vision_frontend, trials=10, warmup=3)
t_step=timed(lambda: step_block(x16))
fs=[]; peak=[]
for t in GRID:
    torch.cuda.reset_peak_memory_stats()
    fs.append(timed(lambda: fs_blocks(x16,t)))
    peak.append(torch.cuda.max_memory_allocated()/1e9)
GRID=np.array(GRID); fs=np.array(fs); peak=np.array(peak)
o2=t_vis+fs; o1=np.full_like(fs, t_vis+t_step)   # O(T) total is constant in T -> broadcast to an array
data=dict(ckpt=os.path.basename(ckpt), grid=GRID.tolist(), t_vision=t_vis, t_step_block=t_step,
          fs_blocks_ms=fs.tolist(), peak_gb=peak.tolist(), oT2_total_ms=o2.tolist(), oT_total_ms=o1.tolist(),
          speedup_total=(o2/o1).tolist(), speedup_block=(fs/max(t_step,1e-6)).tolist())
with open(os.path.join(OUTDIR,'profile_oT.json'),'w') as f: json.dump(data,f,indent=2)
print('ckpt:',data['ckpt'],'| t_vision=%.2fms t_step=%.2fms'%(t_vis,t_step))
print('fs_blocks:',[round(x,2) for x in fs],'| peak GB:',[round(x,2) for x in peak])
print('total speedup O(T^2)->O(T):',[round(x,3) for x in (o2/o1)])

# ---- Figure 1: per-step latency vs horizon T ----
plt.figure(figsize=(7,4.5))
plt.axhline(t_vis, color='crimson', ls='--', lw=1.5, label=f'vision frontend (DINOv2+adapter) = {t_vis:.0f} ms  [paid every step]')
plt.plot(GRID, fs, 'o-', color='steelblue', label='Mamba blocks, O(T²) re-encode (history t)')
plt.axhline(t_step, color='seagreen', ls='-', lw=1.5, label=f'Mamba block, O(T) fixed step = {t_step:.1f} ms (flat)')
plt.plot(GRID, o2, 's-', color='navy', alpha=0.6, label='TOTAL per step, O(T²) re-encode')
plt.plot(GRID, o1, '^-', color='darkgreen', alpha=0.6, label='TOTAL per step, O(T) step()')
plt.xlabel('history length  T  (frames)'); plt.ylabel('per-step latency (ms)')
plt.title('Per-step inference latency vs horizon — vision frontend dominates')
plt.legend(fontsize=7.5, loc='center right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'fig_latency_vs_T.png'), dpi=150); plt.close()

# ---- Figure 2: total speedup O(T^2)->O(T) vs T (the "invisible win") ----
plt.figure(figsize=(7,4))
plt.plot(GRID, o2/o1, 'o-', color='purple')
plt.axhline(1.0, color='gray', ls=':')
for xx,yy in zip(GRID, o2/o1): plt.annotate(f'{yy:.2f}×', (xx,yy), textcoords='offset points', xytext=(0,6), fontsize=7, ha='center')
plt.xlabel('history length  T  (frames)'); plt.ylabel('end-to-end per-step speedup  (O(T²)→O(T))')
plt.title('The O(T) win is invisible behind the vision frontend (~1.0× across T)')
plt.grid(alpha=0.3); plt.ylim(0.95, max(1.1, float((o2/o1).max())*1.05)); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'fig_speedup_vs_T.png'), dpi=150); plt.close()

# ---- Figure 3: peak memory of the O(T^2) re-encode vs T ----
plt.figure(figsize=(7,4))
plt.plot(GRID, peak, 'o-', color='darkorange')
plt.xlabel('history length  T  (frames)'); plt.ylabel('peak GPU memory (GB)')
plt.title('Peak memory of the O(T²) re-encode path vs horizon')
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'fig_peakmem_vs_T.png'), dpi=150); plt.close()

print('FIGURES + JSON written to', OUTDIR)
print('  fig_latency_vs_T.png  fig_speedup_vs_T.png  fig_peakmem_vs_T.png  profile_oT.json')
