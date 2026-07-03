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
miss,unexp=policy.load_state_dict(fp,strict=False)
print('miss(non-bb):',[k for k in miss if 'shared_backbone' not in k],'| unexp:',unexp)
scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load('/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'); scaler.cuda()

h5path=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/*.hdf5'))[0]
stem=os.path.splitext(os.path.basename(h5path))[0]
with h5py.File(h5path,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]; img=f['/observations/images/top'][()]
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
def img_t(i): return torch.from_numpy(rearrange(img[i],'h w c -> c h w')/255.0).float().cuda().unsqueeze(0)

print('GT action[0]:',np.round(action[0],3))
print('\n=== TEST S: mamba_policy_par.step() (norm-in, denorm-out) ===')
hid=policy.init_hidden_states(batch_size=1,device='cuda'); es=[]
for i in range(8):
    with torch.no_grad():
        out,hid=policy.step(norm_q(qpos[i]),{'top':img_t(i)},hid)
        den=denorm(out).view(1,16,14)[0,0].cpu().numpy()
    es.append(float(np.abs(den-action[i]).max()))
    if i==0: print(' f0 pred',np.round(den,3))
print(' maxabsdiff f0..:',[round(e,3) for e in es])

print('\n=== TEST F: shapes of forward_sequence path (chi in shape) ===')
try:
    feats=torch.from_numpy(feat.astype(np.float32)).cuda()
    cam=policy.process_vision_chunk(feats)
    print(' process_vision_chunk(feat',tuple(feats.shape),') ->',tuple(cam.shape))
    obs=torch.cat([norm_q(qpos[i]) for i in range(len(qpos))],dim=0).unsqueeze(0)
    caminp = cam.unsqueeze(0) if cam.dim()==2 else cam
    print(' obs',tuple(obs.shape),'| caminp',tuple(caminp.shape))
    with torch.no_grad():
        seq=policy.forward_sequence(obs, caminp)
    print(' forward_sequence ->',tuple(seq.shape))
except Exception as e:
    print(' shape-probe failed:',repr(e))
