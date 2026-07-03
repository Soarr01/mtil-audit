import os, glob, re, h5py, numpy as np, torch
from einops import rearrange
from inference_M import MyInferenceModel
from train.mamba_policy import MambaConfig

config = MambaConfig()
config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
_vd=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_*/checkpoints'),key=os.path.getmtime)
ckpt=min(glob.glob(os.path.join(_vd[-1],'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ckpt)
m=MyInferenceModel(checkpoint_path=ckpt,scaler_path=scaler_path,config=config,lowdim_dict=lowdim_dict).cuda().eval()

H5='/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train'
FT='/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat/train'
h5path=sorted(glob.glob(os.path.join(H5,'*.hdf5')))[0]
stem=os.path.splitext(os.path.basename(h5path))[0]; featdir=os.path.join(FT,stem)
print('h5:',h5path,'| feat:',featdir,'exists',os.path.isdir(featdir))
with h5py.File(h5path,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]; img=f['/observations/images/top'][()]
print('qpos',qpos.shape,'action',action.shape,'img',img.shape,img.dtype)
feat=np.load(os.path.join(featdir,'top.npy')); caq=np.load(os.path.join(featdir,'qpos.npy')); caa=np.load(os.path.join(featdir,'action.npy'))
print('cached feat',feat.shape,'qpos',caq.shape,'action',caa.shape)

n=min(len(qpos),len(caq))
print('\n[A] cached qpos vs hdf5 qpos  maxabsdiff =', float(np.abs(caq[:n]-qpos[:n]).max()))
print('[A] cached action vs hdf5 act maxabsdiff =', float(np.abs(caa[:n]-action[:n]).max()))

im0=torch.from_numpy(rearrange(img[0],'h w c -> c h w')/255.0).float().cuda().unsqueeze(0)
try:
    with torch.no_grad(): live=m.policy.shared_backbone(im0)
    lv=live[0].float().cpu().numpy(); cf=feat[0].astype(np.float32)
    print('\n[B] live DINOv2',tuple(live.shape),'| cached top[0]',cf.shape)
    if lv.shape==cf.shape:
        print('[B] feat maxabsdiff =',float(np.abs(lv-cf).max()),'| live mean/std',round(float(lv.mean()),3),round(float(lv.std()),3),'| cache mean/std',round(float(cf.mean()),3),round(float(cf.std()),3))
    else:
        print('[B] SHAPE MISMATCH -> tien xu ly anh khac nhau')
except Exception as e:
    print('[B] shared_backbone failed:',repr(e))

def norm_q(qn):
    t=lambda a:torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
    d1=m.scaler.normalize({'agl_1':t(qn[0:1]),'agl_2':t(qn[1:2]),'agl_3':t(qn[2:3]),'agl_4':t(qn[3:4]),'agl_5':t(qn[4:5]),'agl_6':t(qn[5:6]),'gripper_pos':t(qn[6:7])})
    a1=torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']],dim=1)
    d2=m.scaler.normalize({'agl2_1':t(qn[7:8]),'agl2_2':t(qn[8:9]),'agl2_3':t(qn[9:10]),'agl2_4':t(qn[10:11]),'agl2_5':t(qn[11:12]),'agl2_6':t(qn[12:13]),'gripper_pos2':t(qn[13:14])})
    a2=torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']],dim=1)
    return torch.cat([a1,a2],dim=1).float()

print('\n[C] open-loop (GT qpos+anh that), denorm-pred[step0] vs GT action:')
m.reset_hiddens()
errs=[]
for i in range(min(10,len(qpos))):
    imi=torch.from_numpy(rearrange(img[i],'h w c -> c h w')/255.0).float().cuda().unsqueeze(0)
    with torch.no_grad():
        out=m(norm_q(qpos[i]),{'top':imi})
        den=m.denormalize(out).view(1,16,14)[0,0].float().cpu().numpy()
    e=float(np.abs(den-action[i]).max()); errs.append(e)
    if i<3:
        print(f' f{i} pred',np.round(den,3))
        print(f' f{i} GT  ',np.round(action[i],3),'maxabsdiff',round(e,4))
print('[C] maxabsdiff per-frame:',[round(e,3) for e in errs])
