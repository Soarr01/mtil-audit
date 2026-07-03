"""
history_ablation.py — DOES FULL HISTORY HELP? (open-loop, transfer_cube version_6, no training.)
For each context window k, run forward_sequence on the last-k frames ending at frame t, take the
frame-t 16-step action chunk, compare to ground-truth action[t:t+16]. Average over frames + a few
trajectories. If error is FLAT from small k -> the model does not use long history -> full-history
gives no accuracy benefit (undercuts the 'encode full history' premise). If error DROPS as k grows
-> history helps. NOTE: open-loop MSE is a first signal; closed-loop rollout is the gold standard.
"""
import os, glob, re, h5py, numpy as np, torch
import mamba_policy_par as MP
from train.scaler_M import Scaler

CKPT_DIR='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints'
SCALER='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
H5DIR='/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train'
FEATDIR='/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat/train'
KS=[1,2,4,8,16,32,64,99999]   # 99999 = full history
N_EP=3; STRIDE=8; NQ=16

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
ckpt=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ckpt)
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load(SCALER); scaler.cuda()

def t_(a): return torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
def norm_q(qn):
    d1=scaler.normalize({'agl_1':t_(qn[0:1]),'agl_2':t_(qn[1:2]),'agl_3':t_(qn[2:3]),'agl_4':t_(qn[3:4]),'agl_5':t_(qn[4:5]),'agl_6':t_(qn[5:6]),'gripper_pos':t_(qn[6:7])})
    a1=torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']],dim=1)
    d2=scaler.normalize({'agl2_1':t_(qn[7:8]),'agl2_2':t_(qn[8:9]),'agl2_3':t_(qn[9:10]),'agl2_4':t_(qn[10:11]),'agl2_5':t_(qn[11:12]),'agl2_6':t_(qn[12:13]),'gripper_pos2':t_(qn[13:14])})
    a2=torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']],dim=1)
    return torch.cat([a1,a2],dim=1).float()
def denorm(a):
    a1={'agl_1_act':a[...,0:1],'agl_2_act':a[...,1:2],'agl_3_act':a[...,2:3],'agl_4_act':a[...,3:4],'agl_5_act':a[...,4:5],'agl_6_act':a[...,5:6],'gripper_act':a[...,6:7]}
    a2={'agl2_1_act':a[...,7:8],'agl2_2_act':a[...,8:9],'agl2_3_act':a[...,9:10],'agl2_4_act':a[...,10:11],'agl2_5_act':a[...,11:12],'agl2_6_act':a[...,12:13],'gripper_act2':a[...,13:14]}
    d1=scaler.denormalize(a1); d2=scaler.denormalize(a2)
    return torch.cat([d1['agl_1_act'],d1['agl_2_act'],d1['agl_3_act'],d1['agl_4_act'],d1['agl_5_act'],d1['agl_6_act'],d1['gripper_act'],
                      d2['agl2_1_act'],d2['agl2_2_act'],d2['agl2_3_act'],d2['agl2_4_act'],d2['agl2_5_act'],d2['agl2_6_act'],d2['gripper_act2']],dim=-1)

@torch.no_grad()
def pred_frame(cam, obs, t, k):
    lo=max(0, t-k+1)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        seq=policy.forward_sequence(obs[:, lo:t+1], {'top': cam[lo:t+1].unsqueeze(0)})
    return denorm(seq[:, -1].float())[0]   # [16,14] chunk predicted for frame t

h5s=sorted(glob.glob(os.path.join(H5DIR,'*.hdf5')))[:N_EP]
agg={k:[] for k in KS}      # mean maxabsdiff per (ep) for each k
for h5path in h5s:
    stem=os.path.splitext(os.path.basename(h5path))[0]
    with h5py.File(h5path,'r') as f: qpos=f['/observations/qpos'][()]; action=f['/action'][()]
    feat=np.load(os.path.join(FEATDIR,stem,'top.npy'))
    T=feat.shape[0]
    feats=torch.from_numpy(feat.astype(np.float32)).cuda()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        cam=policy.process_vision_chunk(feats).float()
    obs=torch.cat([norm_q(qpos[i]) for i in range(T)],dim=0).unsqueeze(0)
    frames=list(range(0, T-NQ, STRIDE))
    for k in KS:
        errs=[]
        for t in frames:
            p=pred_frame(cam,obs,t,k).cpu().numpy()          # [16,14]
            gt=action[t:t+NQ]                                # [16,14]
            errs.append(float(np.abs(p-gt).max()))
        agg[k].append(np.mean(errs))
    print(f'  {stem}: T={T}, {len(frames)} frames sampled')

print(f'\n=== HISTORY ABLATION (transfer_cube v6, {N_EP} eps, open-loop maxabsdiff vs GT chunk) ===')
full=np.mean(agg[99999])
print(f' {"window k":>10} | {"mean maxabsdiff":>16} | {"vs full":>10}')
for k in KS:
    m=np.mean(agg[k]); lbl='FULL' if k==99999 else str(k)
    print(f' {lbl:>10} | {m:16.4f} | {m/max(full,1e-9):9.2f}x')
print('\n => If error is FLAT from small k, the model does NOT exploit long history (full-history premise weak).')
print('    If error DROPS as k grows, history helps. (Closed-loop rollout is the gold-standard follow-up.)')
