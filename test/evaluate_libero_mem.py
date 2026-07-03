"""evaluate_libero_mem.py — closed-loop LIBERO-Mem eval for the FULL-replication C4 model (dual-cam +
square 128^2 + recipe). Run in ssm_vla_eval AFTER `source eval_render.sh` (EGL).

Merges: (a) the mtil_plus gold runtime — dual camera (agentview + eye_in_hand), proprio = robot_states[9]
(gripper_qpos2 + eef_pos3 + eef_quat4), images resized to 128^2, clamp[-5,5], ACT temporal ensemble;
with (b) the cracked LIBERO-Mem success machinery — .pruned_init init states, reset_subgoal_progress,
20 dummy settle steps, _check_success(inc=True) every step, gripper 0->-1, success = env 'done', and the
[::-1] vertical flip (live render is upside-down vs the upright stored training images).

WINDOW_K=0 -> full-history O(T) step();  WINDOW_K=k>0 -> forward_sequence on the last-k frames (windowed model).
Usage: WINDOW_K=<k> evaluate_libero_mem.py <task_name_no_demo> <ckpt_dir> <n_rollouts> [max_steps]
"""
import os, sys, glob, re
import numpy as np, torch
import mamba_policy_libero as MP
from train.scaler_M import Scaler
from torchvision import transforms
from libero.libero.envs import OffScreenRenderEnv

TASK = sys.argv[1]
CKPT_DIR = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 400
SCALER = f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_libero_mem_{TASK[:20]}.pth'
CAMS = ['agentview', 'eye_in_hand']
RUNTIME_KEY = {'agentview': 'agentview_image', 'eye_in_hand': 'robot0_eye_in_hand_image'}
FLIP = os.environ.get('FLIP', 'both')   # 'both' (default), 'agentview', or 'none' — eye_in_hand flip is an assumption to verify
WINDOW_K = int(os.environ.get('WINDOW_K', '0'))

_t = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])
def prep(img_hwc, cam):   # env render [256,256,3] uint8 -> flip -> resize 128^2 -> [1,3,128,128]
    do_flip = FLIP == 'both' or (FLIP == 'agentview' and cam == 'agentview')
    a = np.ascontiguousarray(img_hwc[::-1]) if do_flip else np.ascontiguousarray(img_hwc)
    return _t(a).unsqueeze(0).cuda()

# ---- model ----
ck = min(glob.glob(os.path.join(CKPT_DIR, '*val_epoch_loss=*.ckpt')),
         key=lambda f: float(re.search(r'val_epoch_loss=(\d+\.\d+)', f).group(1)))
print('ckpt:', ck, '| rollouts:', N, '| maxT:', MAXT, '| WINDOW_K:', WINDOW_K, '| FLIP:', FLIP)
cfg = MP.MambaConfig(); cfg.camera_names = CAMS; cfg.embed_dim = 2048; cfg.d_model = 2048
cfg.action_dim = 7; cfg.lowdim_dim = 9; cfg.img_size = (128, 128); cfg.num_blocks = 4; cfg.sum_camera_feats = False
policy = MP.MambaPolicy(camera_names=CAMS, embed_dim=2048, lowdim_dim=9, d_model=2048, action_dim=7,
                        sum_camera_feats=False, num_blocks=4, future_steps=8, img_size=(128, 128),
                        mamba_cfg=cfg, lang_dim=None).cuda().eval()
sd = torch.load(ck, map_location='cpu')['state_dict']   # CPU load: avoids the ckpt optimizer tensors on GPU
miss, unexp = policy.load_state_dict({k.replace('policy.', ''): v for k, v in sd.items() if k.startswith('policy.')}, strict=False)
print('miss(non-bb):', [k for k in miss if 'shared_backbone' not in k], '| unexp:', unexp)
scaler = Scaler(lowdim_dict={'qpos': 9, 'act': 7}); scaler.load(SCALER); scaler.cuda()
NQ = policy.future_steps   # 8

def alloc_states(bs=1, dtype=torch.float16):   # fp16 states to match the fp16-autocast step() input
    st = []                                    # (init_hidden_states uses dtype=None=fp32 -> causal_conv1d_update dtype mismatch)
    for blk in policy.blocks:
        if hasattr(blk.mixer, 'allocate_inference_cache'):
            cs, ss = blk.mixer.allocate_inference_cache(bs, max_seqlen=1, dtype=dtype)
        else:
            cs, ss = None, None
        st.append((cs, ss))
    return st

def get_robot_state(obs):   # live LIBERO obs -> the 9-dim robot_states the HDF5 stored (VERIFIED)
    return np.concatenate([obs['robot0_gripper_qpos'], obs['robot0_eef_pos'], obs['robot0_eef_quat']]).astype(np.float32)
def norm_q(p9):
    return scaler.normalize({'qpos': torch.from_numpy(p9).float().cuda().unsqueeze(0)})['qpos'].clamp(-5, 5)  # [1,9]
def denorm(chunk):   # [...,7] normalized -> raw
    return scaler.denormalize({'act': chunk})['act']

# ---- env + init states ----
LIBERO_ROOT = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/libero-mem/libero/libero'
bddl = os.path.join(LIBERO_ROOT, 'bddl_files', 'libero_mem', TASK + '.bddl')
assert os.path.exists(bddl), f"bddl not found: {bddl}"
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); env.seed(0)
_isf = f'/data/nas07_new/PersonalData/phuocthien/ssm_vla/libero-mem/scripts/__init_data/{TASK}.pruned_init'
init_states = torch.load(_isf, weights_only=False) if os.path.exists(_isf) else None
print('bddl:', os.path.basename(bddl), '| init_states:', None if init_states is None else init_states.shape)

succ = 0; subgoals = []; overs = []
try:
    _gs = env.env.parsed_problem['goal_state']
    NSUB = len(_gs[0]) - 1 if (_gs and _gs[0] and _gs[0][0] == 'sequence') else len(_gs)  # sequence: gs=[['sequence',sg1,..]]
except Exception: NSUB = None
print('total subgoals:', NSUB)
for rid in range(N):
    env.reset()
    obs = env.set_init_state(init_states[rid % len(init_states)]) if init_states is not None else env.reset()
    env.env.reset_subgoal_progress()
    try: env.env._overshot = False
    except: pass
    for _ in range(20): obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))   # settle, gripper open
    ata = torch.zeros([MAXT, MAXT + NQ, 7]).cuda(); hidden = alloc_states(1, torch.float16)
    done_any = False; vf = {c: [] for c in CAMS}; qh = []; maxsub = 0
    for tt in range(MAXT):
        env.env._check_success(inc=True)   # advance sequential-subgoal timers EVERY step
        maxsub = max(maxsub, len(getattr(env.env, '_satisfied_subgoals', [])))   # partial-credit: max subgoals reached
        imgs = {cam: prep(obs[RUNTIME_KEY[cam]], cam) for cam in CAMS}
        q = norm_q(get_robot_state(obs))                                  # [1,9]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            if WINDOW_K > 0:
                for cam in CAMS:
                    vf[cam].append(policy.process_vision_chunk(policy.shared_backbone(imgs[cam])))
                qh.append(q); lo = max(0, len(qh) - WINDOW_K)
                seq = policy.forward_sequence(torch.cat(qh[lo:], 0).unsqueeze(0),
                                              {cam: torch.cat(vf[cam][lo:], 0).unsqueeze(0) for cam in CAMS})
                chunk = seq[:, -1]
            else:
                chunk, hidden = policy.step(q, imgs, hidden)             # [1,8,7], O(T) recurrent
        ata[[tt], tt:tt + NQ] = denorm(chunk.float())[0]
        cur = ata[:, tt]; pop = torch.any(cur != 0, dim=1); cur = cur[pop]
        w = np.exp(-0.01 * np.arange(len(cur))); w = torch.from_numpy(w / w.sum()).float().cuda().unsqueeze(1)
        act = (cur * w).sum(0).cpu().numpy()
        act = list(act); act[-1] = act[-1] if act[-1] != 0 else -1       # gripper 0 -> -1
        obs, r, done, info = env.step(act)
        if done: done_any = True; break
    ov = bool(getattr(env.env, '_overshot', False))   # counting tasks: did it do MORE than N reps?
    if done_any and NSUB: maxsub = NSUB   # success = all subgoals reached (maxsub lags by 1 at the final step's break)
    succ += int(done_any); subgoals.append(maxsub); overs.append(ov)
    print(f'Rollout {rid}: success={done_any} subgoals={maxsub} overshoot={ov} (step {tt+1})', flush=True)
_avg = sum(subgoals) / max(1, len(subgoals))
_dist = {k: subgoals.count(k) for k in sorted(set(subgoals))}
_ovr = 100 * sum(overs) / max(1, len(overs))
_tag = "full" if WINDOW_K == 0 else "win" + str(WINDOW_K)
print(f'\n=== LIBERO-Mem {TASK[:30]} [{_tag}] SUCCESS: {100*succ/N:.1f}% over {N} | avg_subgoals={_avg:.2f}/{NSUB} | overshoot={_ovr:.0f}% | dist(subgoals->count)={_dist} ===')
env.close()
