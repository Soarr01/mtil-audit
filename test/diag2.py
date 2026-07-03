import os, glob, re, h5py, numpy as np, torch
from einops import rearrange
from inference_M import MyInferenceModel
from train.mamba_policy import MambaConfig

config=MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
_vd=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_*/checkpoints'),key=os.path.getmtime)
ckpt=min(glob.glob(os.path.join(_vd[-1],'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
m=MyInferenceModel(checkpoint_path=ckpt,scaler_path=scaler_path,config=config,lowdim_dict=lowdim_dict).cuda().eval()

h5path=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/*.hdf5'))[0]
with h5py.File(h5path,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]; img=f['/observations/images/top'][()]

def norm_q(qn):
    t=lambda a:torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
    d1=m.scaler.normalize({'agl_1':t(qn[0:1]),'agl_2':t(qn[1:2]),'agl_3':t(qn[2:3]),'agl_4':t(qn[3:4]),'agl_5':t(qn[4:5]),'agl_6':t(qn[5:6]),'gripper_pos':t(qn[6:7])})
    a1=torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']],dim=1)
    d2=m.scaler.normalize({'agl2_1':t(qn[7:8]),'agl2_2':t(qn[8:9]),'agl2_3':t(qn[9:10]),'agl2_4':t(qn[10:11]),'agl2_5':t(qn[11:12]),'agl2_6':t(qn[12:13]),'gripper_pos2':t(qn[13:14])})
    a2=torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']],dim=1)
    return torch.cat([a1,a2],dim=1).float()
def img_t(i): return torch.from_numpy(rearrange(img[i],'h w c -> c h w')/255.0).float().cuda().unsqueeze(0)

def run(use_norm_in, use_denorm, label, k=6):
    m.reset_hiddens(); es=[]; first=None
    for i in range(k):
        ld = norm_q(qpos[i]) if use_norm_in else torch.from_numpy(qpos[i].astype(np.float32)).cuda().unsqueeze(0)
        with torch.no_grad():
            out=m(ld,{'top':img_t(i)})
            pred=(m.denormalize(out) if use_denorm else out).view(1,16,14)[0,0].float().cpu().numpy()
        if i==0: first=pred
        es.append(float(np.abs(pred-action[i]).max()))
    print(f'{label} f0pred',np.round(first,2),'| maxabsdiff',[round(e,2) for e in es])

print('GT action[0]:',np.round(action[0],3))
run(True,  True,  '[norm-in, denorm-out  (current eval)]')
run(True,  False, '[norm-in, raw-out                   ]')
run(False, False, '[raw-in,  raw-out                   ]')
run(False, True,  '[raw-in,  denorm-out                ]')
