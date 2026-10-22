"""
evaluate_fs_windowed.py — GOLD-STANDARD history ablation via CLOSED-LOOP rollout success.
Each step, forward_sequence sees only the LAST-k frames of history (windowed), and we take the
current frame = last in the window. Sweep k to see how rollout SUCCESS depends on context length.
k >= MAXT == full history (reproduces evaluate_fs.py). transfer_cube, version_6.
Usage: evaluate_fs_windowed.py <N> <k_csv>   e.g.  evaluate_fs_windowed.py 10 1,16,99999
"""
import os, json, glob, re, sys
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler
from sim_env import make_sim_env, BOX_POSE

NUM_ROLLOUTS = int(sys.argv[1]) if len(sys.argv)>1 else 10
K_LIST = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv)>2 else ['1','8','99999'])]
CKPT_DIR = sys.argv[3] if len(sys.argv)>3 else '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints'
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
# argv[3] may be a checkpoints dir (lowest ROUNDED val loss in filename) or an explicit .ckpt file
def pick_ckpt(d):
    """checkpoints dir -> lowest UNROUNDED val loss from the run's history.json (train_c3.py); filenames carry a
    rounded loss, so the filename fallback can pick a tied, non-minimal checkpoint and is only used without history"""
    files=glob.glob(os.path.join(d,'*val_epoch_loss=*.ckpt'))
    h=os.path.join(os.path.dirname(os.path.normpath(d)),'history.json')
    if os.path.exists(h):
        H=json.load(open(h)); H=H if isinstance(H,list) else H.get('epochs',H)
        best=min((e for e in H if e.get('val_epoch_loss') is not None),key=lambda e:e['val_epoch_loss'])['epoch']
        m=[f for f in files if re.search(rf'epoch={best:02d}-',os.path.basename(f))]
        if m: return m[0]
    print('WARNING: selecting by the ROUNDED loss in checkpoint filenames (no usable history.json)',flush=True)
    return min(files,key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
ckpt=CKPT_DIR if CKPT_DIR.endswith('.ckpt') else pick_ckpt(CKPT_DIR)
# NO_ENSEMBLE=1: execute only the first action of the CURRENT chunk (no ACT temporal aggregation) -> a k=1 policy is then truly memoryless
NO_ENSEMBLE=os.environ.get('NO_ENSEMBLE','0')=='1'
print('ckpt:',ckpt,'| rollouts:',NUM_ROLLOUTS,'| K_LIST:',K_LIST,'| NO_ENSEMBLE:',NO_ENSEMBLE)
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
miss,unexp=policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
# only the frozen DINOv2 (dropped from training ckpts, rebuilt from torch.hub) may be missing
assert not unexp and all(k.startswith('shared_backbone.') for k in miss), (unexp, [k for k in miss if not k.startswith('shared_backbone.')])
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
def sample_box_pose():
    r=np.vstack([[0.0,0.2],[0.4,0.6],[0.05,0.05]]); p=np.random.uniform(r[:,0],r[:,1]); return np.concatenate([p,[1,0,0,0]])

env=make_sim_env('sim_transfer_cube'); env_max_reward=env.task.max_reward
MAXT=400; NQ=16; SD=14
summary={}
for K in K_LIST:
    highest=[]; returns=[]
    np.random.seed(int(os.environ.get('SEED','1234')))   # SAME pose sequence across k; SEED env for the 3-seed Wilson protocol
    for rid in range(NUM_ROLLOUTS):
        BOX_POSE[0]=sample_box_pose(); ts=env.reset()
        ata=torch.zeros([MAXT, MAXT+NQ, SD]).cuda()
        vf_hist=[]; q_hist=[]; rewards=[]
        for tt in range(MAXT):
            obs=ts.observation
            with torch.no_grad():
                dino=policy.shared_backbone(prep(obs['images']['top']))
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    vf=policy.process_vision_chunk(dino)
            vf_hist.append(vf); q_hist.append(norm_q(np.array(obs['qpos'])))
            lo=0 if K>=MAXT else max(0, len(q_hist)-K)        # WINDOW: last-K frames
            obs_seq=torch.cat(q_hist[lo:],dim=0).unsqueeze(0)
            cam_seq={'top': torch.cat(vf_hist[lo:],dim=0).unsqueeze(0)}
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                seq=policy.forward_sequence(obs_seq, cam_seq)
            chunk=seq[:,-1].float()                            # current frame = last in window
            ata[[tt], tt:tt+NQ]=denorm(chunk)
            cur=ata[:,tt]; pop=torch.all(cur!=0,dim=1); cur=cur[pop]
            w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
            act=denorm(chunk)[0,0] if NO_ENSEMBLE else (cur*w).sum(dim=0)
            ts=env.step(act.cpu().numpy()); rewards.append(ts.reward)
        rewards=np.array([0 if r is None else r for r in rewards])
        hi=float(rewards.max()); returns.append(float(rewards.sum())); highest.append(hi)
        print(f'  [k={K}] Rollout {rid}: highest={hi} Success={hi==env_max_reward}', flush=True)
    sr=float(np.mean(np.array(highest)==env_max_reward))
    summary[K]=(sr, float(np.mean(returns)))
    print(f'=== k={K}: SUCCESS {sr*100:.1f}% over {NUM_ROLLOUTS} | avg return {np.mean(returns):.1f} ===', flush=True)

print('\n========== WINDOWED HISTORY ABLATION (closed-loop, transfer_cube v6) ==========')
print(f' {"window k":>10} | {"success%":>9} | {"avg return":>10}')
for K in K_LIST:
    sr,ar=summary[K]; lbl='FULL' if K>=MAXT else str(K)
    print(f' {lbl:>10} | {sr*100:8.1f}% | {ar:10.1f}')
