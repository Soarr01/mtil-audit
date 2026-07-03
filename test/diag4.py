import os, glob, re, h5py, numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler

lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig()
config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
_vd=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_*/checkpoints'),key=os.path.getmtime)
ckpt=min(glob.glob(os.path.join(_vd[-1],'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt',ckpt)
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
fp={k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')}
policy.load_state_dict(fp,strict=False)
scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load('/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'); scaler.cuda()

h5path=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/*.hdf5'))[0]
stem=os.path.splitext(os.path.basename(h5path))[0]
with h5py.File(h5path,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]
feat=np.load(f'/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat/train/{stem}/top.npy')

def t(a): return torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
def norm_q(qn):
    d1=scaler.normalize({'agl_1':t(qn[0:1]),'agl_2':t(qn[1:2]),'agl_3':t(qn[2:3]),'agl_4':t(qn[3:4]),'agl_5':t(qn[4:5]),'agl_6':t(qn[5:6]),'gripper_pos':t(qn[6:7])})
    a1=torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']],dim=1)
    d2=scaler.normalize({'agl2_1':t(qn[7:8]),'agl2_2':t(qn[8:9]),'agl2_3':t(qn[9:10]),'agl2_4':t(qn[10:11]),'agl2_5':t(qn[11:12]),'agl2_6':t(qn[12:13]),'gripper_pos2':t(qn[13:14])})
    a2=torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']],dim=1)
    return torch.cat([a1,a2],dim=1).float()
def denorm(a):
    a1={'agl_1_act':a[...,0:1],'agl_2_act':a[...,1:2],'agl_3_act':a[...,2:3],'agl_4_act':a[...,3:4],'agl_5_act':a[...,4:5],'agl_6_act':a[...,5:6],'gripper_act':a[...,6:7]}
    a2={'agl2_1_act':a[...,7:8],'agl2_2_act':a[...,8:9],'agl2_3_act':a[...,9:10],'agl2_4_act':a[...,10:11],'agl2_5_act':a[...,11:12],'agl2_6_act':a[...,12:13],'gripper_act2':a[...,13:14]}
    d1=scaler.denormalize(a1); d2=scaler.denormalize(a2)
    return torch.cat([d1['agl_1_act'],d1['agl_2_act'],d1['agl_3_act'],d1['agl_4_act'],d1['agl_5_act'],d1['agl_6_act'],d1['gripper_act'],
                      d2['agl2_1_act'],d2['agl2_2_act'],d2['agl2_3_act'],d2['agl2_4_act'],d2['agl2_5_act'],d2['agl2_6_act'],d2['gripper_act2']],dim=-1)

feats=torch.from_numpy(feat.astype(np.float32)).cuda()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    cam=policy.process_vision_chunk(feats)
print('process_vision_chunk ->',tuple(cam.shape))
obs_all=torch.cat([norm_q(qpos[i]) for i in range(len(qpos))],dim=0).unsqueeze(0)
T=obs_all.shape[1]
def run_fs(k):
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        return policy.forward_sequence(obs_all[:,:k], {'top': cam[:k].unsqueeze(0)})

print('\n[F-full] forward_sequence tren ca',T,'frame:')
seq=run_fs(T); print(' out shape',tuple(seq.shape))
den=denorm(seq)
getp=(lambda i: den[0,i,0]) if den.dim()==4 else (lambda i: den[0,i])
errs=[float(np.abs(getp(i).detach().cpu().numpy()-action[i]).max()) for i in range(min(12,T))]
print(' f0 pred',np.round(getp(0).detach().cpu().numpy(),3))
print(' f0 GT  ',np.round(action[0],3))
print(' maxabsdiff f0..:',[round(e,3) for e in errs])

print('\n[F-prefix k=20] (kiem tra nhan qua):')
seqp=run_fs(20); denp=denorm(seqp)
getpp=(lambda i: denp[0,i,0]) if denp.dim()==4 else (lambda i: denp[0,i])
p19=getpp(19).detach().cpu().numpy()
print(' f19 pred(prefix)',np.round(p19,3))
print(' f19 GT          ',np.round(action[19],3),'| maxabsdiff',round(float(np.abs(p19-action[19]).max()),4))
print(' prefix-vs-full f19 diff',round(float(np.abs(p19-getp(19).detach().cpu().numpy()).max()),6),'(nho => nhan qua, online OK)')
