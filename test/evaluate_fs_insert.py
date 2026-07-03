import os, glob, re, sys
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler
from sim_env import make_sim_env, BOX_POSE
from visualize_episodes import save_videos

NUM_ROLLOUTS = int(sys.argv[1]) if len(sys.argv)>1 else 50
# argv[2] (optional) = path den thu muc checkpoints cua run insertion (de chac chan khong lay nham ckpt transfer_cube)
CKPT_DIR = sys.argv[2] if len(sys.argv)>2 else None
_seed=os.environ.get('SEED')
if _seed: np.random.seed(int(_seed)); torch.manual_seed(int(_seed)); print('SEED:',_seed)  # reproducible peg/socket-pose sampling for variance/CI study
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_insertion.pth'
results_dir='/data/nas07_new/PersonalData/phuocthien/mtil_data/eval_video'; os.makedirs(results_dir,exist_ok=True)
if CKPT_DIR is not None and CKPT_DIR.endswith('.ckpt'):
    ckpt=CKPT_DIR  # argv[2] tro thang vao 1 file .ckpt cu the (de test late ckpt: val_loss != rollout success)
else:
    if CKPT_DIR is None:
        _vd=sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_*/checkpoints'),key=os.path.getmtime)
        CKPT_DIR=_vd[-1]
    ckpt=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ckpt,'| rollouts:',NUM_ROLLOUTS)
policy=MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
fp={k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')}
miss,unexp=policy.load_state_dict(fp,strict=False)
print('miss(non-bb):',[k for k in miss if 'shared_backbone' not in k],'| unexp:',unexp)
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
def sample_insertion_pose():
    # Peg
    peg_r=np.vstack([[0.1,0.2],[0.4,0.6],[0.05,0.05]]); peg_p=np.random.uniform(peg_r[:,0],peg_r[:,1])
    peg_pose=np.concatenate([peg_p,[1,0,0,0]])
    # Socket
    soc_r=np.vstack([[-0.2,-0.1],[0.4,0.6],[0.05,0.05]]); soc_p=np.random.uniform(soc_r[:,0],soc_r[:,1])
    socket_pose=np.concatenate([soc_p,[1,0,0,0]])
    return np.concatenate([peg_pose,socket_pose])

env=make_sim_env('sim_insertion'); env_max_reward=env.task.max_reward
MAXT=400; NQ=16; SD=14
highest=[]; returns=[]
for rid in range(NUM_ROLLOUTS):
    BOX_POSE[0]=sample_insertion_pose(); ts=env.reset()
    ata=torch.zeros([MAXT, MAXT+NQ, SD]).cuda()
    vf_hist=[]; q_hist=[]; rewards=[]; image_list=[]
    for tt in range(MAXT):
        obs=ts.observation; image_list.append(obs['images'])
        with torch.no_grad():
            dino=policy.shared_backbone(prep(obs['images']['top']))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                vf=policy.process_vision_chunk(dino)
        vf_hist.append(vf); q_hist.append(norm_q(np.array(obs['qpos'])))
        obs_seq=torch.cat(q_hist,dim=0).unsqueeze(0)
        cam_seq={'top': torch.cat(vf_hist,dim=0).unsqueeze(0)}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            seq=policy.forward_sequence(obs_seq, cam_seq)
        chunk=seq[:,tt].float()
        chunk_dn=denorm(chunk)
        ata[[tt], tt:tt+NQ]=chunk_dn
        cur=ata[:,tt]; pop=torch.all(cur!=0,dim=1); cur=cur[pop]
        w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
        target=(cur*w).sum(dim=0).cpu().numpy()
        ts=env.step(target); rewards.append(ts.reward)
    rewards=np.array([0 if r is None else r for r in rewards])
    hi=float(rewards.max()); returns.append(float(rewards.sum())); highest.append(hi)
    print(f'Rollout {rid}: highest={hi} max={env_max_reward} Success={hi==env_max_reward}')
    try: save_videos(image_list,0.02,video_path=os.path.join(results_dir,f'fs_insert{rid}.mp4'))
    except Exception as e: print('  save_videos failed:',e)
sr=float(np.mean(np.array(highest)==env_max_reward))
print(f'\n=== INSERTION SUCCESS RATE: {sr*100:.1f}% over {NUM_ROLLOUTS} | avg return {np.mean(returns):.2f} ===')
for r in range(int(env_max_reward)+1):
    c=int((np.array(highest)>=r).sum()); print(f'Reward >= {r}: {c}/{NUM_ROLLOUTS} = {100*c/NUM_ROLLOUTS:.1f}%')
