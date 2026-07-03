"""
evaluate_fs_mem.py — CLOSED-LOOP rollout eval for the delayed-cue 2-target MEMORY task
(sim_transfer_cube_mem). Clone of evaluate_fs_windowed.py (last-k windowed forward_sequence
rollout via lo=max(0,len-K)).

Per episode, BEFORE env.reset(): a random target bin (MEM_TARGET) is drawn and the cue is
shown only for frames [0, MEM_TAU); the policy must REMEMBER the cue to place the cube in
the correct bin. Success (reward==4) is given by the env ONLY for correct-bin placement, so
binary success directly measures cue-memory. We ALSO report bin-choice accuracy (did the cube
end nearer the correct vs wrong bin, from final env_state x vs +-0.18) so we get a soft signal
even when full manipulation success is partial.

WINDOW_K knob (env var, also argv-overridable):
    WINDOW_K=0  (or >=MAXT) -> FULL history forward_sequence (== evaluate_fs.py path)
    WINDOW_K=k             -> windowed: forward_sequence sees only the last-k frames
SEED env var seeds numpy for a reproducible pose+target sequence (default 1234).

Usage:
    WINDOW_K=0 python evaluate_fs_mem.py <N> <ckpt_dir>
    WINDOW_K=16 SEED=7 python evaluate_fs_mem.py 20 /abs/.../version_X/checkpoints
Reuses the transfer_cube scaler (scaler_transfer.pth) + same MambaConfig (14-dim, top cam,
chunk 16) — the mem model has identical I/O dims.
"""
import os, glob, re, sys
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler
import sim_env as SE                         # import module so we mutate the RIGHT globals
from sim_env import make_sim_env, BOX_POSE

NUM_ROLLOUTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CKPT_DIR = sys.argv[2] if len(sys.argv) > 2 else \
    '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints'
# WINDOW_K: 0 (default) == FULL history; k>0 == windowed last-k frames.
WINDOW_K = int(os.environ.get('WINDOW_K', '0'))
SEED = int(os.environ.get('SEED', '1234'))

lowdim_dict = {'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config = MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
scaler_path = '/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer_mem.pth'   # MUST match the mem-training scaler (mem demos have a place-down -> different action range)
ckpt = min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),
           key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:', ckpt, '| rollouts:', NUM_ROLLOUTS, '| WINDOW_K:', WINDOW_K, '| SEED:', SEED)
policy = MP.MambaPolicy(camera_names=config.camera_names,embed_dim=config.embed_dim,lowdim_dim=config.lowdim_dim,
  d_model=config.d_model,action_dim=config.action_dim,sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks,mamba_cfg=config).cuda().eval()
sd = torch.load(ckpt, map_location='cuda:0')['state_dict']
policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')}, strict=False)
scaler = Scaler(lowdim_dict=lowdim_dict); scaler.load(scaler_path); scaler.cuda()

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

env = make_sim_env('sim_transfer_cube_mem'); env_max_reward = env.task.max_reward
MAXT = 500; NQ = 16; SD = 14                  # mem task: longer horizon
K = WINDOW_K if WINDOW_K > 0 else MAXT          # 0 -> full history

highest=[]; returns=[]; bin_correct=[]
np.random.seed(SEED)
for rid in range(NUM_ROLLOUTS):
    # set the per-episode cue BEFORE reset (so initialize_episode reads them)
    SE.MEM_TARGET[0] = int(np.random.randint(2))   # 0=LEFT, 1=RIGHT
    SE.MEM_TAU[0] = 12
    target_bin = SE.MEM_TARGET[0]
    BOX_POSE[0] = sample_box_pose()
    ts = env.reset()
    ata=torch.zeros([MAXT, MAXT+NQ, SD]).cuda()
    vf_hist=[]; q_hist=[]; rewards=[]
    final_state=None
    for tt in range(MAXT):
        obs=ts.observation
        with torch.no_grad():
            dino=policy.shared_backbone(prep(obs['images']['top']))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                vf=policy.process_vision_chunk(dino)
        vf_hist.append(vf); q_hist.append(norm_q(np.array(obs['qpos'])))
        lo = 0 if K>=MAXT else max(0, len(q_hist)-K)   # WINDOW: last-K frames
        obs_seq=torch.cat(q_hist[lo:],dim=0).unsqueeze(0)
        cam_seq={'top': torch.cat(vf_hist[lo:],dim=0).unsqueeze(0)}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            seq=policy.forward_sequence(obs_seq, cam_seq)
        chunk=seq[:,-1].float()                          # current frame = last in window
        ata[[tt], tt:tt+NQ]=denorm(chunk)
        cur=ata[:,tt]; pop=torch.all(cur!=0,dim=1); cur=cur[pop]
        w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
        ts=env.step((cur*w).sum(dim=0).cpu().numpy()); rewards.append(ts.reward)
        final_state=ts.observation['env_state']          # qpos[16:] -> cube is first 7
    rewards=np.array([0 if r is None else r for r in rewards])
    hi=float(rewards.max()); returns.append(float(rewards.sum())); highest.append(hi)
    # bin-choice accuracy: cube final x vs +-0.18 (correct bin center)
    cube_x=float(final_state[0])
    chosen_bin = 0 if cube_x < 0 else 1              # nearer LEFT(-0.18) or RIGHT(+0.18)
    is_bin_correct = (chosen_bin == target_bin)
    bin_correct.append(is_bin_correct)
    print(f'  Rollout {rid}: target_bin={target_bin} highest={hi} '
          f'Success={hi==env_max_reward} cube_x={cube_x:+.3f} chosen_bin={chosen_bin} '
          f'BinOK={is_bin_correct}', flush=True)

sr=float(np.mean(np.array(highest)==env_max_reward))
ba=float(np.mean(bin_correct))
lbl='FULL' if K>=MAXT else str(K)
print('\n========== DELAYED-CUE MEMORY EVAL (closed-loop, sim_transfer_cube_mem) ==========')
print(f' window k         : {lbl}')
print(f' rollouts         : {NUM_ROLLOUTS}  (seed {SEED})')
print(f' SUCCESS (rew==4) : {sr*100:.1f}%   ({int(np.sum(np.array(highest)==env_max_reward))}/{NUM_ROLLOUTS})')
print(f' bin-choice acc   : {ba*100:.1f}%   (cube nearer correct bin; chance=50%)')
print(f' avg return       : {np.mean(returns):.1f}')
