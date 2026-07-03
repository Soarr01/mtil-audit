"""
diag_libero_openloop.py — localize the 0% rollout. Feed a VAL demo's OWN cached features+qpos
through forward_sequence, denorm the predicted action chunks, compare to GT actions (real units).
This removes proprio-convention / env / live-vision issues -> isolates MODEL quality.
Small diff => model fine, the 0% is an eval-harness bug. Large diff => model underfit.
"""
import os, glob, re, numpy as np, torch
import mamba_policy_par as MP
from train.scaler_M import Scaler

FEAT='/data/nas07_new/PersonalData/phuocthien/mtil_data/libero_mem_feat/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo'
CKPT_DIR='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs_libero/KITCHEN_SCENE1_1_pic_full/lightning_logs/version_0/checkpoints'
SCALER='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_libero_KITCHEN_SCENE1_1_pic.pth'

cfg=MP.MambaConfig(); cfg.camera_names=['agentview']; cfg.embed_dim=2048; cfg.d_model=2048
cfg.action_dim=7; cfg.lowdim_dim=8; cfg.img_size=(640,480); cfg.num_blocks=4; cfg.sum_camera_feats=False
policy=MP.MambaPolicy(camera_names=['agentview'],embed_dim=2048,lowdim_dim=8,d_model=2048,action_dim=7,
   sum_camera_feats=False,num_blocks=4,future_steps=16,img_size=(640,480),mamba_cfg=cfg).cuda().eval()
ck=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',os.path.basename(ck))
sd=torch.load(ck,map_location='cpu')['state_dict']
policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
scaler=Scaler(lowdim_dict={**{f'obs_{i}':1 for i in range(8)},**{f'act_{i}':(16,1) for i in range(7)}}); scaler.load(SCALER); scaler.cuda()

def t_(a): return torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
def norm_q(p8):
    d=scaler.normalize({f'obs_{i}':t_(p8[i:i+1]) for i in range(8)})
    return torch.cat([d[f'obs_{i}'].unsqueeze(1) for i in range(8)],dim=1).float()
def denorm(a):
    d=scaler.denormalize({f'act_{i}':a[...,i:i+1] for i in range(7)})
    return torch.cat([d[f'act_{i}'] for i in range(7)],dim=-1)

# val demo (last 10 = val split). pick demo_96
demos=sorted([d for d in os.listdir(FEAT) if d.startswith('demo_')], key=lambda s:int(s.split('_')[1]))
dm=demos[-4]   # a val demo
rp=os.path.join(FEAT,dm)
feat=np.load(os.path.join(rp,'agentview.npy')); qpos=np.load(os.path.join(rp,'qpos.npy')); act=np.load(os.path.join(rp,'action.npy'))
T=feat.shape[0]; print(f'demo {dm}: T={T}, action range [{act.min():.3f},{act.max():.3f}]')
feats=torch.from_numpy(feat.astype(np.float32)).cuda()
with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
    cam=policy.process_vision_chunk(feats)
obs=torch.cat([norm_q(qpos[i]) for i in range(T)],dim=0).unsqueeze(0)
with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
    seq=policy.forward_sequence(obs, {'agentview':cam.unsqueeze(0)})   # [1,T,16,7]
pred=denorm(seq.float())[0]   # [T,16,7] real units

# immediate action: pred[t,0] vs GT act[t]
imm_err=[]; perdim=np.zeros(7)
for t in range(T):
    p=pred[t,0].cpu().numpy(); g=act[t]
    imm_err.append(float(np.abs(p-g).max())); perdim+=np.abs(p-g)
perdim/=T
print(f'\n=== OPEN-LOOP immediate-action error (real units) over {T} frames ===')
print(f' mean maxabsdiff: {np.mean(imm_err):.4f} | max: {np.max(imm_err):.4f}')
print(f' per-dim mean|err| (7-DoF): {np.round(perdim,4)}')
print(f' GT action std per-dim: {np.round(act.std(0),4)}')
print(f'\n sample frames (pred[t,0] vs GT):')
for t in [0,30,60,T//2,T-20]:
    if t<T: print(f'  t={t:3d} pred {np.round(pred[t,0].cpu().numpy(),3)} | GT {np.round(act[t],3)}')
