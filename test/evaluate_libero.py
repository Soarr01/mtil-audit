"""
evaluate_libero.py — closed-loop rollout eval for LIBERO-Mem (C4). Run in ssm_vla_eval env
(torch+mamba+robosuite+libero) AFTER sourcing eval_render.sh (EGL). forward_sequence O(T^2)
closed-loop (the proven-correct path). success = env 'done'. proprio = ee_pos(3)+quat2axisangle(eef_quat)(3)+gripper_qpos(2).
Usage: evaluate_libero.py <task_name_no_demo> <ckpt_dir> <n_rollouts> [max_steps]
"""
import os, sys, glob, re
import numpy as np, torch
import mamba_policy_par as MP
from train.scaler_M import Scaler
from torchvision import transforms
def get_image_transform(resize_hw=(640,480)):
    t=transforms.Compose([transforms.ToPILImage(), transforms.Resize((resize_hw[1],resize_hw[0])), transforms.ToTensor()])
    return lambda img: t(img)
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

TASK = sys.argv[1]                      # e.g. KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate
CKPT_DIR = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv)>3 else 10
MAXT = int(sys.argv[4]) if len(sys.argv)>4 else 250
SCALER = f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_libero_{(TASK+"_demo")[:20]}.pth'

# ---- model ----
ck=min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
print('ckpt:',ck,'| rollouts:',N,'| maxT:',MAXT)
cfg=MP.MambaConfig(); cfg.camera_names=['agentview']; cfg.embed_dim=2048; cfg.d_model=2048
cfg.action_dim=7; cfg.lowdim_dim=8; cfg.img_size=(640,480); cfg.num_blocks=4; cfg.sum_camera_feats=False
policy=MP.MambaPolicy(camera_names=['agentview'],embed_dim=2048,lowdim_dim=8,d_model=2048,action_dim=7,
   sum_camera_feats=False,num_blocks=4,future_steps=16,img_size=(640,480),mamba_cfg=cfg).cuda().eval()
sd=torch.load(ck,map_location='cpu')['state_dict']   # CPU load: avoids putting the ckpt's optimizer tensors on GPU
miss,unexp=policy.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
print('miss(non-bb):',[k for k in miss if 'shared_backbone' not in k],'| unexp:',unexp)
scaler=Scaler(lowdim_dict={**{f'obs_{i}':1 for i in range(8)},**{f'act_{i}':(16,1) for i in range(7)}}); scaler.load(SCALER); scaler.cuda()
tf=get_image_transform((640,480))

def t_(a): return torch.from_numpy(np.array(a,dtype=np.float32)).cuda()
ZERO_EEORI=os.environ.get('ZERO_EEORI','0')=='1'
def norm_q(p8):  # [8] -> normalized [1,8]
    d=scaler.normalize({f'obs_{i}':t_(p8[i:i+1]) for i in range(8)})
    out=torch.cat([d[f'obs_{i}'].unsqueeze(1) for i in range(8)],dim=1).float()
    if ZERO_EEORI: out[:,3:6]=0.0   # TEST: zero singularity-prone ee_ori dims (=train mean) to localize the proprio issue
    return out
def denorm(a):   # [...,7] normalized -> raw
    d=scaler.denormalize({f'act_{i}':a[...,i:i+1] for i in range(7)})
    return torch.cat([d[f'act_{i}'] for i in range(7)],dim=-1)
def proprio(obs):
    q=obs['robot0_eef_quat']; q=-q if q[3]<0 else q   # canonicalize sign: at angle~pi singularity, q and -q are the same rotation -> avoid OOD spike (ee_ori scaler std~7e-4)
    return np.concatenate([obs['robot0_eef_pos'], quat2axisangle(q), obs['robot0_gripper_qpos']]).astype(np.float32)

# ---- env + init states ----
LIBERO_ROOT='/data/nas07_new/PersonalData/phuocthien/ssm_vla/libero-mem/libero/libero'
bddl=os.path.join(LIBERO_ROOT,'bddl_files','libero_mem',TASK+'.bddl')
assert os.path.exists(bddl), f"bddl not found: {bddl}"
env=OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); env.seed(0)  # seed affects object positions even with fixed init (LIBERO recipe)
_isf=f'/data/nas07_new/PersonalData/phuocthien/ssm_vla/libero-mem/scripts/__init_data/{TASK}.pruned_init'
init_states=torch.load(_isf,weights_only=False) if os.path.exists(_isf) else None
print('bddl:',os.path.basename(bddl),'| init_states:', None if init_states is None else init_states.shape)

WINDOW_K=int(os.environ.get('WINDOW_K','0'))   # 0=full-history step(); k>0=forward_sequence on last-K frames (for windowed-trained models)
print('WINDOW_K:',WINDOW_K,'(0=full)')
NQ=16; SD=7; succ=0
for rid in range(N):
    env.reset()
    obs=env.set_init_state(init_states[rid % len(init_states)]) if init_states is not None else env.reset()
    env.env.reset_subgoal_progress()
    try: env.env._overshot=False
    except: pass
    for _ in range(20): obs,_,_,_=env.step(np.array([0,0,0,0,0,0,-1.0]))   # 20 dummy settle steps (gripper open) — LIBERO recipe
    ata=torch.zeros([MAXT,MAXT+NQ,SD]).cuda(); hidden=policy.init_hidden_states(1); done_any=False; vf=[]; qh=[]
    for tt in range(MAXT):
        env.env._check_success(inc=True)   # advance sequential-subgoal timers EVERY step (LIBERO-Mem success machinery)
        img=tf(np.ascontiguousarray(obs['agentview_image'][::-1])).unsqueeze(0).cuda()  # [::-1] vertical flip: live render is upside-down vs upright training images
        q=norm_q(proprio(obs))                                       # [1,8]
        with torch.no_grad():
            if WINDOW_K>0:   # windowed-trained model: forward_sequence on the last-K frames (matches windowed training)
                d=policy.shared_backbone(img)
                with torch.autocast('cuda',dtype=torch.float16): f=policy.process_vision_chunk(d)
                vf.append(f); qh.append(q); lo=max(0,len(qh)-WINDOW_K)
                with torch.autocast('cuda',dtype=torch.float16):
                    seq=policy.forward_sequence(torch.cat(qh[lo:],0).unsqueeze(0), {'agentview':torch.cat(vf[lo:],0).unsqueeze(0)})
                chunk=seq[:,-1]
            else:
                chunk,hidden=policy.step(q, {'agentview': img}, hidden)  # full-history O(T) recurrent step -> [1,16,7]
        ata[[tt],tt:tt+NQ]=denorm(chunk.float())[0]
        cur=ata[:,tt]; pop=torch.any(cur!=0,dim=1); cur=cur[pop]   # any (not all): LIBERO rotation dims are always 0
        w=np.exp(-0.01*np.arange(len(cur))); w=torch.from_numpy(w/w.sum()).float().cuda().unsqueeze(1)
        act=(cur*w).sum(0).cpu().numpy()   # ACT temporal ensemble (proven on this architecture in evaluate_fs.py)
        act=list(act); act[-1]=act[-1] if act[-1]!=0 else -1   # gripper 0 -> -1 (LIBERO recipe)
        obs,r,done,info=env.step(act)
        if done: done_any=True; break   # done = final-goal _check_success() = task success
    succ+=int(done_any)
    print(f'Rollout {rid}: success={done_any} (step {tt+1})',flush=True)
print(f'\n=== LIBERO {TASK[:30]} SUCCESS: {100*succ/N:.1f}% over {N} ===')
env.close()
