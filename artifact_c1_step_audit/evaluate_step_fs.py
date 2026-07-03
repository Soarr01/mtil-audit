"""
evaluate_step_fs.py — CONFIRM: the FIXED step() (mamba_policy_par.py, 2026-06-19) drives a correct
O(T) closed-loop rollout. Mirrors evaluate_fs.py (transfer_cube) but replaces the O(T^2)
forward_sequence-on-full-history with policy.step() using maintained per-block hidden states.
Expected: ~96% success (matching evaluate_fs.py), at O(1) work per step.
Run in fp32 (Ampere) — matches the diag5 kill-test that proved step()==forward_sequence.
"""
import os, glob, re, sys, json
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
# PRISTINE=1 -> load the published (broken) MambaPolicy from the clean upstream clone (C1 audit):
# expect ~0% rollout, vs the fixed local copy at ~96-100%. Default = local FIXED copy.
if os.environ.get('PRISTINE') == '1':
    import importlib.util as _il
    _spec=_il.spec_from_file_location('mp_pristine', os.environ.get('PRISTINE_PATH','/data/nas07_new/PersonalData/phuocthien/mtil_original_audit/train/mamba_policy_par.py'))
    MP=_il.module_from_spec(_spec); sys.modules['mp_pristine']=MP; _spec.loader.exec_module(MP)
    print('*** USING PRISTINE/PUBLISHED step() (upstream HEAD 9bea607) — expect ~0% ***')
else:
    import mamba_policy_par as MP
from train.scaler_M import Scaler
from sim_env import make_sim_env, BOX_POSE
from visualize_episodes import save_videos

NUM_ROLLOUTS = int(sys.argv[1]) if len(sys.argv)>1 else 50
# Reproducibility: SEED via argv[3] or $SEED (default 1234) -> deterministic pose sequence (np.random drives sample_box_pose).
SEED = int(sys.argv[3]) if len(sys.argv)>3 else int(os.environ.get('SEED','1234'))
np.random.seed(SEED); torch.manual_seed(SEED)
# ARTIFACT_ROOT: point at the published reproduction bundle (checkpoint.ckpt + scaler_transfer.pth); else local /data defaults.
ART = os.environ.get('ARTIFACT_ROOT','')
def artpath(rel, fallback): return os.path.join(ART, rel) if ART else fallback
# argv[2] (optional) = checkpoints dir or .ckpt; default = bundle checkpoint or transfer_cube run version_6
CKPT_ARG = sys.argv[2] if len(sys.argv)>2 else artpath('checkpoint.ckpt','/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints')
lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config=MP.MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.lowdim_dim=14
config.d_model=2048; config.action_dim=14; config.sum_camera_feats=False; config.num_blocks=4
scaler_path=artpath('scaler_transfer.pth','/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth')
results_dir=os.environ.get('RESULTS_DIR', artpath('results','/data/nas07_new/PersonalData/phuocthien/mtil_data/eval_video')); os.makedirs(results_dir,exist_ok=True)
ckpt = CKPT_ARG if CKPT_ARG.endswith('.ckpt') else min(glob.glob(os.path.join(CKPT_ARG,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ckpt,'| rollouts:',NUM_ROLLOUTS,'| seed:',SEED,'| MODE=step() O(T) fp32')
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
def sample_box_pose():
    r=np.vstack([[0.0,0.2],[0.4,0.6],[0.05,0.05]]); p=np.random.uniform(r[:,0],r[:,1]); return np.concatenate([p,[1,0,0,0]])

env=make_sim_env('sim_transfer_cube'); env_max_reward=env.task.max_reward
MAXT=400; NQ=16; SD=14
highest=[]; returns=[]; records=[]
for rid in range(NUM_ROLLOUTS):
    box_pose=sample_box_pose(); BOX_POSE[0]=box_pose
    try:
        ts=env.reset()
    except Exception as e:                     # a prior rollout's invalid physics can poison the env; rebuild it
        print(f'  [rollout {rid} env.reset failed: {type(e).__name__}] -> recreating env', flush=True)
        env=make_sim_env('sim_transfer_cube'); env_max_reward=env.task.max_reward; ts=env.reset()
    ata=torch.zeros([MAXT, MAXT+NQ, SD]).cuda()
    hidden=policy.init_hidden_states(1)          # per-block (conv_state, ssm_state); O(T) recurrence
    rewards=[]; image_list=[]
    for tt in range(MAXT):
        obs=ts.observation; image_list.append(obs['images'])
        q=norm_q(np.array(obs['qpos']))          # [1,14]
        img=prep(obs['images']['top'])           # [1,3,H,W]
        with torch.no_grad():
            chunk, hidden = policy.step(q, {'top': img}, hidden)   # [1,16,14], fp32 O(1)/step
        chunk_dn=denorm(chunk.float())
        ata[[tt], tt:tt+NQ]=chunk_dn
        cur=ata[:,tt]; pop=torch.all(cur!=0,dim=1); cur=cur[pop]
        w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
        target=(cur*w).sum(dim=0).cpu().numpy()
        try:
            ts=env.step(target); rewards.append(ts.reward)
        except Exception as e:                 # a broken policy's garbage actions can drive MuJoCo into an
            rewards.append(0)                  # invalid physics state (mjWARN_BADQACC); that rollout is a failure
            print(f'  [rollout {rid} physics-invalid at t={tt}: {type(e).__name__}] -> counted as 0-reward failure', flush=True)
            break
    rewards=np.array([0 if r is None else r for r in rewards])
    hi=float(rewards.max()); returns.append(float(rewards.sum())); highest.append(hi)
    records.append({'seed':SEED,'rollout':rid,'box_pose':[round(float(x),5) for x in box_pose.tolist()],
                    'reward_sequence':[int(x) for x in rewards.tolist()],'highest':hi,'success':bool(hi==env_max_reward)})
    print(f'Rollout {rid}: highest={hi} max={env_max_reward} Success={hi==env_max_reward}')
    try: save_videos(image_list,0.02,video_path=os.path.join(results_dir,f'step{rid}.mp4'))
    except Exception as e: print('  save_videos failed:',e)
sr=float(np.mean(np.array(highest)==env_max_reward))
print(f'\n=== STEP() O(T) SUCCESS RATE: {sr*100:.1f}% over {NUM_ROLLOUTS} | avg return {np.mean(returns):.2f} ===')
ladder={int(r):int((np.array(highest)>=r).sum()) for r in range(int(env_max_reward)+1)}
for r in range(int(env_max_reward)+1):
    c=ladder[r]; print(f'Reward >= {r}: {c}/{NUM_ROLLOUTS} = {100*c/NUM_ROLLOUTS:.1f}%')
# machine-readable dump for third-party diffing (cert requirement)
summary={'mode':'pristine' if os.environ.get('PRISTINE')=='1' else 'fixed','seed':SEED,'n':NUM_ROLLOUTS,
         'success_rate':sr,'avg_return':float(np.mean(returns)),'ladder_counts':ladder,'ckpt':os.path.basename(ckpt)}
out=os.path.join(results_dir,f'results_step_{summary["mode"]}_seed{SEED}.json')
json.dump({'summary':summary,'rollouts':records}, open(out,'w'), indent=1)
print('wrote',out)
