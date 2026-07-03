"""
diag6.py — localize why evaluate_step_fs gave 0%. Compare the REAL policy.step() (live images,
maintaining per-block state) frame-by-frame vs forward_sequence (the 96% reference) on episode_0.
Tests fp16 and fp32. If step matches forward_sequence per-frame -> bug is in the rollout script;
if it diverges -> step() still has a defect.
"""
import os, glob, re, h5py, numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler

CKPT_DIR='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints'
SCALER='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
H5=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/*.hdf5'))[0]
_stem=os.path.splitext(os.path.basename(H5))[0]
FEAT=f'/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat/train/{_stem}/top.npy'
N=24

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
def prep(im): return torch.from_numpy(rearrange(im,'h w c -> c h w')/255.0).float().cuda().unsqueeze(0)

with h5py.File(H5,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]; imgs=f['/observations/images/top'][()]
feat=np.load(FEAT)
cam_feats=torch.from_numpy(feat[:N].astype(np.float32)).cuda()
obs=torch.cat([norm_q(qpos[i]) for i in range(N)],dim=0).unsqueeze(0)

# reference: forward_sequence fp16 on cached feats (the 96% path)
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    camv=policy.process_vision_chunk(cam_feats)
    ref=policy.forward_sequence(obs, {'top': camv.unsqueeze(0)})    # [1,N,16,14]
ref_dn=denorm(ref.float())

def run_step(precision):
    fp16=(precision=='fp16')
    if fp16:
        hidden=[list(blk.mixer.allocate_inference_cache(1,max_seqlen=1,dtype=torch.float16)) for blk in policy.blocks]
    else:
        hidden=policy.init_hidden_states(1)
    out=[]
    for tt in range(N):
        q=norm_q(qpos[tt]); img=prep(imgs[tt])
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
            chunk,hidden=policy.step(q, {'top': img}, hidden)
        out.append(denorm(chunk.float())[0])    # [16,14]
    return torch.stack(out,0)    # [N,16,14]

print('ckpt:',os.path.basename(ckpt),'| N=',N)
for prec in ['fp32','fp16']:
    try:
        s=run_step(prec)
        diffs=[float((s[t]-ref_dn[0,t]).abs().max()) for t in range(N)]
        gt=[float((s[t].cpu().numpy()-action[t:t+16]).max()) for t in range(N)]
        print(f'\n[{prec}] step() vs forward_sequence (per-frame maxabsdiff):')
        print('  frames 0..11 :', [round(d,3) for d in diffs[:12]])
        print('  frames 12..  :', [round(d,3) for d in diffs[12:]])
        print(f'  MEAN diff = {np.mean(diffs):.4f} | MAX = {np.max(diffs):.4f}')
        print(f'  (ref vs GT mean = {np.mean([float((ref_dn[0,t].cpu().numpy()-action[t:t+16]).max()) for t in range(N)]):.4f})')
    except Exception as e:
        print(f'\n[{prec}] FAILED: {repr(e)}')
