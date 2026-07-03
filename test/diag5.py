"""
diag5.py — KILL-TEST: chung minh step() hong = bug dau day residual, va ban-sua khop forward_sequence.

3 duong tren CUNG mot chuoi feature da fuse x[1,T,d_model]:
  (A) ref_par   : block loop SONG SONG (= dung block stack cua forward_sequence) -> tham chieu
  (B) broken    : replay step() ban HIEN TAI (mamba_policy_par.py:686-721) frame-by-frame
  (C) fixed     : replay step() ban SUA (Block.forward 2-luong chuan) frame-by-frame
Ky vong: maxabsdiff(ref, broken) LON dan; maxabsdiff(ref, fixed) ~0.
Chay fp32 (Ampere RTX3080 OK); neu kernel loi -> fallback fp16.
"""
import os, glob, h5py, numpy as np, torch
import mamba_policy_par as MP
from train.scaler_M import Scaler

CKPT_DIR = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_14/checkpoints'
SCALER   = '/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_insertion.pth'
H5GLOB   = '/data/nas07_new/PersonalData/phuocthien/mtil_data/insertion/train/*.hdf5'
FEATROOT = '/data/nas07_new/PersonalData/phuocthien/mtil_data/insertion_feat/train'
NFRAMES  = 40   # so frame replay (Python per-frame, giu nho)

lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
import re
ckpt=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ckpt)
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
fp={k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')}
miss,unexp=policy.load_state_dict(fp,strict=False)
print('miss(non-bb):',[k for k in miss if 'shared_backbone' not in k],'| unexp:',unexp)
scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load(SCALER); scaler.cuda()

h5path=sorted(glob.glob(H5GLOB))[0]; stem=os.path.splitext(os.path.basename(h5path))[0]
with h5py.File(h5path,'r') as f: qpos=f['/observations/qpos'][()]
feat=np.load(os.path.join(FEATROOT,stem,'top.npy'))
print('episode:',stem,'| qpos',qpos.shape,'| feat',feat.shape)

def t(a): return torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
def norm_q(qn):
    d1=scaler.normalize({'agl_1':t(qn[0:1]),'agl_2':t(qn[1:2]),'agl_3':t(qn[2:3]),'agl_4':t(qn[3:4]),'agl_5':t(qn[4:5]),'agl_6':t(qn[5:6]),'gripper_pos':t(qn[6:7])})
    a1=torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']],dim=1)
    d2=scaler.normalize({'agl2_1':t(qn[7:8]),'agl2_2':t(qn[8:9]),'agl2_3':t(qn[9:10]),'agl2_4':t(qn[10:11]),'agl2_5':t(qn[11:12]),'agl2_6':t(qn[12:13]),'gripper_pos2':t(qn[13:14])})
    a2=torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']],dim=1)
    return torch.cat([a1,a2],dim=1).float()

@torch.no_grad()
def build_fused(T, dtype):
    feats=torch.from_numpy(feat[:T].astype(np.float32)).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(dtype==torch.float16)):
        cam=policy.process_vision_chunk(feats)            # [T, embed_dim]
    cam=cam.to(dtype); obs=torch.cat([norm_q(qpos[i]) for i in range(T)],dim=0).unsqueeze(0).to(dtype)  # [1,T,14]
    B,L=1,T
    cam_feats_flat=cam.unsqueeze(0).reshape(B*L,-1)
    lowdim_flat=obs.reshape(B*L,-1)
    cam_feats_proj=policy.in_proj(cam_feats_flat)
    fused=policy.cross_attn(query=cam_feats_proj.unsqueeze(0), key=lowdim_flat.unsqueeze(0), value=lowdim_flat.unsqueeze(0)).squeeze(0)
    return fused.view(B,L,policy.d_model), cam, obs

@torch.no_grad()
def ref_parallel(x):
    xr=x; residual=None
    for blk in policy.blocks: xr,residual=blk(xr,residual)
    return policy.out_proj(xr)            # [1,T,flat]

def alloc(dtype):
    return [list(blk.mixer.allocate_inference_cache(1, max_seqlen=1, dtype=dtype)) for blk in policy.blocks]

@torch.no_grad()
def replay(x, T, fixed: bool, dtype):
    st=alloc(dtype); preds=[]
    for tt in range(T):
        hidden=x[:,tt]; residual=None
        for i,blk in enumerate(policy.blocks):
            conv_st,ssm_st=st[i]
            residual = hidden if residual is None else residual+hidden
            hln=blk.norm(residual.to(blk.norm.weight.dtype)).to(dtype)
            y,conv_st,ssm_st=blk.mixer.step(hln.unsqueeze(1),conv_st,ssm_st); y=y.squeeze(1)
            st[i]=[conv_st,ssm_st]
            if fixed:
                # canonical Block.forward (2-luong): hidden=mlp(norm2(residual_new)), residual giu rieng
                residual = y + residual
                hidden = blk.mlp(blk.norm2(residual.to(blk.norm2.weight.dtype)).to(dtype))
            else:
                # BAN HONG hien tai (mamba_policy_par.py:709-718)
                hidden_out = y + residual
                hidden_out = blk.mlp(blk.norm2(hidden_out.to(blk.norm2.weight.dtype)).to(dtype)) + hidden_out
                hidden=hidden_out; residual=hidden_out
        preds.append(policy.out_proj(hidden))
    return torch.stack(preds,dim=1)       # [1,T,flat]

def md(a,b): return float((a-b).abs().max())

def run(dtype):
    tag='fp32' if dtype==torch.float32 else 'fp16'
    T=min(NFRAMES, feat.shape[0])
    x,cam,obs=build_fused(T,dtype)
    ref=ref_parallel(x)
    fs=policy.forward_sequence(obs, {'top': cam.unsqueeze(0)}).reshape(1,T,-1)   # sanity: real path
    brk=replay(x,T,fixed=False,dtype=dtype)
    fix=replay(x,T,fixed=True,dtype=dtype)
    print(f'\n========== {tag} | T={T} ==========')
    print(f'sanity  maxabsdiff(ref_parallel, forward_sequence) = {md(ref,fs):.3e}  (xac nhan x dung)')
    frames=[0,1,2,5,10,20,T-1]; frames=[f for f in frames if f<T]
    print(' frame |  ref-vs-BROKEN  |  ref-vs-FIXED')
    for f in frames:
        print(f'  {f:4d} |  {md(ref[:,f],brk[:,f]):12.4e} |  {md(ref[:,f],fix[:,f]):12.4e}')
    print(f' OVERALL broken maxabsdiff = {md(ref,brk):.4e} | fixed maxabsdiff = {md(ref,fix):.4e}')
    print(f' => VERDICT: fixed {"MATCHES" if md(ref,fix)<(5e-3 if tag=="fp16" else 1e-2) else "DIFFERS"} forward_sequence; '
          f'broken {"DIVERGES" if md(ref,brk)>1e-1 else "ok?"}')

try:
    run(torch.float32)
except Exception as e:
    print('fp32 failed:',repr(e)); print('--- fallback fp16 ---'); run(torch.float16)
