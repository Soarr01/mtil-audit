"""
cueswap_diag.py — DECISIVE diagnostic for the delayed-cue experiment: does the FULL model RESPOND
to the cue at all? For each FIXED cube pose we roll out TWICE with identical everything except the
cue (MEM_TARGET=0 LEFT vs =1 RIGHT) and record which bin the cube ends in. If the model reads the
cue, flipping it should flip the bin. We report:
  - follow rate  : fraction of (pose,cue) where chosen_bin == cued bin  (chance 50%)
  - FLIP rate    : fraction of poses where flipping the cue flips the chosen bin (the causal signal)
A near-0 flip rate => the model IGNORES the cue (memorized / cue not learned), not just "windowed can't".
Usage: WINDOW_K=0 python cueswap_diag.py <NPOSES> <ckpt_dir>
"""
import os, glob, re, sys
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler
import sim_env as SE
from sim_env import make_sim_env, BOX_POSE

NPOSES = int(sys.argv[1]) if len(sys.argv)>1 else 10
CKPT_DIR = sys.argv[2]
SEED = int(os.environ.get('SEED','1234'))
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer_mem.pth'
ckpt=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',os.path.basename(ckpt),'| NPOSES:',NPOSES,'| SEED:',SEED)
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load(scaler_path); scaler.cuda()

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
def sample_box_pose(): r=np.vstack([[0.0,0.2],[0.4,0.6],[0.05,0.05]]); p=np.random.uniform(r[:,0],r[:,1]); return np.concatenate([p,[1,0,0,0]])

env=make_sim_env('sim_transfer_cube_mem'); MAXT=500; NQ=16; SD=14

def rollout(box_pose, cue):
    SE.MEM_TARGET[0]=int(cue); SE.MEM_TAU[0]=12; BOX_POSE[0]=box_pose.copy()
    ts=env.reset(); ata=torch.zeros([MAXT,MAXT+NQ,SD]).cuda(); vf=[]; qh=[]; final=None
    for tt in range(MAXT):
        obs=ts.observation
        with torch.no_grad():
            dino=policy.shared_backbone(prep(obs['images']['top']))
            with torch.autocast(device_type='cuda',dtype=torch.float16): v=policy.process_vision_chunk(dino)
        vf.append(v); qh.append(norm_q(np.array(obs['qpos'])))
        with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.float16):
            seq=policy.forward_sequence(torch.cat(qh,0).unsqueeze(0), {'top':torch.cat(vf,0).unsqueeze(0)})
        ata[[tt],tt:tt+NQ]=denorm(seq[:,-1].float())
        cur=ata[:,tt]; pop=torch.all(cur!=0,dim=1); cur=cur[pop]
        w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
        ts=env.step((cur*w).sum(0).cpu().numpy()); final=ts.observation['env_state']
    return float(final[0])   # cube final x

np.random.seed(SEED)
poses=[sample_box_pose() for _ in range(NPOSES)]
follow=0; flips=0; rows=[]
for i,bp in enumerate(poses):
    xL=rollout(bp,0); binL=0 if xL<0 else 1     # cue=LEFT(0)
    xR=rollout(bp,1); binR=0 if xR<0 else 1     # cue=RIGHT(1)
    follow += (binL==0)+(binR==1)               # chose cued bin
    flip = (binL==0 and binR==1)                # cue flip -> bin flip in the right direction
    flips += int(flip)
    rows.append((i,xL,binL,xR,binR,flip))
    print(f'  pose{i}: cue=L x={xL:+.3f} bin={binL} | cue=R x={xR:+.3f} bin={binR} | FLIP_correct={flip}',flush=True)

print('\n========== CUE-SWAP DIAGNOSTIC ==========')
print(f' poses                 : {NPOSES} (paired, identical pose; only the cue differs)')
print(f' follow rate (chose cued bin) : {100*follow/(2*NPOSES):.1f}%   (chance 50%)')
print(f' FLIP rate (cue flip -> bin flip correctly) : {100*flips/NPOSES:.1f}%   (>~60% = model USES cue; ~0% = ignores)')
