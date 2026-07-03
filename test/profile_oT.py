"""
profile_oT.py — DE-RISK: is the O(T) inference win VISIBLE, or does the per-frame DINOv2
vision frontend dominate per-step latency? (Review Risk #4.)

Per rollout step the two inference modes cost:
  O(T^2) re-encode (current correct eval): t_vision(1 frame) + t_fs_blocks(history length t)
  O(T)   fixed step():                      t_vision(1 frame) + t_step_block(1 frame, constant)
The Mamba-block saving = t_fs_blocks(t) - t_step_block. MEASURED (profile_oT.json): both terms are
kernel-launch-overhead-bound and ~constant over T<=400 (~9.8 vs ~7 ms), so the saving is a flat
~2.8 ms; the O(t) compute term is below measurement noise and would only grow at T>>400. It matters
end-to-end only if large vs t_vision (~189 ms) — it is not (speedup ~1.014x, flat in T).
This script measures all three on a clean GPU.
"""
import os, glob, re, time, numpy as np, torch
import mamba_policy_par as MP
from train.scaler_M import Scaler

CKPT_DIR='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_14/checkpoints'
FEAT='/data/nas07_new/PersonalData/phuocthien/mtil_data/insertion_feat/train/episode_0/top.npy'
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

feat=np.load(FEAT, mmap_mode='r')
Tmax=min(max(GRID), feat.shape[0])
feats=torch.from_numpy(np.array(feat[:Tmax],dtype=np.float32)).cuda()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    cam=policy.process_vision_chunk(feats).float()        # [Tmax, embed_dim]
obs=torch.randn(1,Tmax,14,device='cuda')                  # lowdim placeholder (timing only)

# build fused x [1,Tmax,d_model] once (matches forward_sequence fusion, single cam)
@torch.no_grad()
def build_fused(dtype):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dtype==torch.float16)):
        cf=cam.to(dtype); cff=cf.unsqueeze(0).reshape(Tmax,-1)
        lf=obs.to(dtype).reshape(Tmax,-1)
        proj=policy.in_proj(cff)
        fused=policy.cross_attn(query=proj.unsqueeze(0),key=lf.unsqueeze(0),value=lf.unsqueeze(0)).squeeze(0)
    return fused.view(1,Tmax,policy.d_model)

def timed(fn, trials=TRIALS, warmup=WARMUP):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(trials): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/trials*1000.0   # ms

@torch.no_grad()
def fs_blocks(x, t):  # parallel block stack + out_proj on t frames (= re-encode per-step block cost: O(t) in history, O(T^2) per episode)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        xr=x[:,:t]; residual=None
        for blk in policy.blocks: xr,residual=blk(xr,residual)
        return policy.out_proj(xr)

@torch.no_grad()
def step_block(x):  # ONE fixed recurrent step (constant) = O(T) per-step block cost
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
def vision_frontend():  # per-frame DINOv2 (fp32) + spatial_adapter (fp16) — 1 frame
    img=torch.rand(1,3,480,640,device='cuda')
    d=policy.shared_backbone(img)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return policy.process_vision_chunk(d)

x16=build_fused(torch.float16)
print(f'ckpt: {os.path.basename(ckpt)} | Tmax={Tmax} | trials={TRIALS}')
print('\n=== per-step latency (ms) ===')
t_vis=timed(vision_frontend, trials=10, warmup=3)
t_step=timed(lambda: step_block(x16))
print(f'  vision frontend (DINOv2+adapter, 1 frame) : {t_vis:8.3f} ms   [paid EVERY step by BOTH modes]')
print(f'  O(T) fixed step  block cost (constant)     : {t_step:8.3f} ms')
print(f'\n  history t | fs_blocks(t) ms |  O(T^2) step total | O(T) step total | block speedup | total speedup')
peak={}
for t in GRID:
    torch.cuda.reset_peak_memory_stats()
    tf=timed(lambda: fs_blocks(x16,t))
    peak[t]=torch.cuda.max_memory_allocated()/1e9
    o2=t_vis+tf; o1=t_vis+t_step
    print(f'  {t:9d} | {tf:14.3f} | {o2:18.3f} | {o1:15.3f} | {tf/max(t_step,1e-6):11.2f}x | {o2/max(o1,1e-6):11.2f}x')
print(f'\n  peak mem fs_blocks: t=1 {peak.get(1,0):.3f} GB -> t={Tmax} {peak.get(Tmax,0):.3f} GB')
print('\n=== VERDICT ===')
tf400=timed(lambda: fs_blocks(x16,Tmax))
print(f'  At T={Tmax}: Mamba-block work = {tf400:.2f}ms vs vision frontend = {t_vis:.2f}ms  ->  '
      f'block/vision = {tf400/max(t_vis,1e-6):.2f}')
print(f'  => O(T) efficiency win is {"VISIBLE (blocks dominate)" if tf400>t_vis else "MOSTLY HIDDEN by vision frontend"} at this horizon.')
print(f'  => Full per-step speedup O(T^2)->O(T) at T={Tmax}: {(t_vis+tf400)/(t_vis+t_step):.2f}x')
