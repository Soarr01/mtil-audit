"""
time_rollout.py — DIRECT end-to-end stopwatch of the closed-loop control tick (resubmission, C2).
Answers the reviewer caveat that §7's numbers are a component-composed timing model: here every
control tick of a REAL transfer_cube rollout is wall-clocked (torch.cuda.synchronize at boundaries).

Two controllers on the SAME checkpoint and SAME pose sequence:
  step : fixed recurrent step()  -> O(1) Mamba work per tick   (evaluate_step_fs.py)
  fs   : forward_sequence on the full accumulated history -> O(t) per tick (evaluate_fs.py)
Per tick we record: vision (DINOv2 + adapter; fs only, step() fuses it), controller (policy call total),
aggregation (temporal ensembling + .cpu()), env (physics + MuJoCo render of the next obs), total.
PREC=amp (default): DINOv2 fp32, everything after it fp16 autocast = the §7 profiling setting.
PREC=fp32: all fp32 (= evaluate_step_fs.py's rollout setting).
Rollout success is also recorded, so the timed runs double as extra evaluation rollouts.

Usage: python time_rollout.py <N_rollouts> <modes_csv step,fs> [ckpt_dir]   (SEED, PREC, WARMUP env vars)
Writes: $RESULTS_DIR (default mtil_data/timing)/timing_<mode>_<prec>_seed<SEED>.json
"""
import os, glob, re, sys, json, time
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP
from train.scaler_M import Scaler
from sim_env import make_sim_env, BOX_POSE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MODES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['step', 'fs']
CKPT_DIR = sys.argv[3] if len(sys.argv) > 3 else '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints'
SEED = int(os.environ.get('SEED', '0'))
PREC = os.environ.get('PREC', 'amp')
OUT = os.environ.get('RESULTS_DIR', '/data/nas07_new/PersonalData/phuocthien/mtil_data/timing'); os.makedirs(OUT, exist_ok=True)
lowdim_dict = {'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
config = MP.MambaConfig(); config.camera_names = ['top']; config.embed_dim = 2048; config.lowdim_dim = 14
config.d_model = 2048; config.action_dim = 14; config.sum_camera_feats = False; config.num_blocks = 4
ckpt = CKPT_DIR if CKPT_DIR.endswith('.ckpt') else min(glob.glob(os.path.join(CKPT_DIR, '*val_epoch_loss=*.ckpt')), key=lambda f: float(re.search(r'val_epoch_loss=(\d+\.\d+)', f).group(1)))
print('ckpt:', ckpt, '| N:', N, '| modes:', MODES, '| seed:', SEED, '| prec:', PREC, '| gpu:', torch.cuda.get_device_name(0), flush=True)
policy = MP.MambaPolicy(camera_names=config.camera_names, embed_dim=config.embed_dim, lowdim_dim=config.lowdim_dim,
  d_model=config.d_model, action_dim=config.action_dim, sum_camera_feats=config.sum_camera_feats,
  num_blocks=config.num_blocks, mamba_cfg=config).cuda().eval()
sd = torch.load(ckpt, map_location='cuda:0')['state_dict']
policy.load_state_dict({k.replace('policy.', ''): v for k, v in sd.items() if k.startswith('policy.')}, strict=False)
scaler = Scaler(lowdim_dict=lowdim_dict); scaler.load('/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'); scaler.cuda()

def t_(a): return torch.from_numpy(np.array(a, dtype=np.float32)).cuda()
def norm_q(qn):
    d1 = scaler.normalize({'agl_1':t_(qn[0:1]),'agl_2':t_(qn[1:2]),'agl_3':t_(qn[2:3]),'agl_4':t_(qn[3:4]),'agl_5':t_(qn[4:5]),'agl_6':t_(qn[5:6]),'gripper_pos':t_(qn[6:7])})
    a1 = torch.cat([d1[k].unsqueeze(1) for k in ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos']], dim=1)
    d2 = scaler.normalize({'agl2_1':t_(qn[7:8]),'agl2_2':t_(qn[8:9]),'agl2_3':t_(qn[9:10]),'agl2_4':t_(qn[10:11]),'agl2_5':t_(qn[11:12]),'agl2_6':t_(qn[12:13]),'gripper_pos2':t_(qn[13:14])})
    a2 = torch.cat([d2[k].unsqueeze(1) for k in ['agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']], dim=1)
    return torch.cat([a1, a2], dim=1).float()
def denorm(a):
    a1 = {'agl_1_act':a[...,0:1],'agl_2_act':a[...,1:2],'agl_3_act':a[...,2:3],'agl_4_act':a[...,3:4],'agl_5_act':a[...,4:5],'agl_6_act':a[...,5:6],'gripper_act':a[...,6:7]}
    a2 = {'agl2_1_act':a[...,7:8],'agl2_2_act':a[...,8:9],'agl2_3_act':a[...,9:10],'agl2_4_act':a[...,10:11],'agl2_5_act':a[...,11:12],'agl2_6_act':a[...,12:13],'gripper_act2':a[...,13:14]}
    d1 = scaler.denormalize(a1); d2 = scaler.denormalize(a2)
    return torch.cat([d1['agl_1_act'],d1['agl_2_act'],d1['agl_3_act'],d1['agl_4_act'],d1['agl_5_act'],d1['agl_6_act'],d1['gripper_act'],
                      d2['agl2_1_act'],d2['agl2_2_act'],d2['agl2_3_act'],d2['agl2_4_act'],d2['agl2_5_act'],d2['agl2_6_act'],d2['gripper_act2']], dim=-1)
def prep(im): return torch.from_numpy(rearrange(im, 'h w c -> c h w') / 255.0).float().cuda().unsqueeze(0)
def sample_box_pose():
    r = np.vstack([[0.0, 0.2], [0.4, 0.6], [0.05, 0.05]]); p = np.random.uniform(r[:, 0], r[:, 1]); return np.concatenate([p, [1, 0, 0, 0]])
def amp():
    return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(PREC == 'amp'))
def init_h():
    # amp: recurrent states in fp16 so the fused step kernels see matching dtypes (as in profile_oT.py §7)
    if PREC == 'amp':
        return [tuple(blk.mixer.allocate_inference_cache(1, max_seqlen=1, dtype=torch.float16)) for blk in policy.blocks]
    return policy.init_hidden_states(1)
def now():
    torch.cuda.synchronize(); return time.perf_counter()

env = make_sim_env('sim_transfer_cube'); env_max_reward = env.task.max_reward
MAXT = 400; NQ = 16; SD = 14

class _Cached(torch.nn.Module):
    """stand-in backbone that returns the already-computed fp32 DINOv2 map (nn.Module so setattr is legal)"""
    def __init__(self, raw): super().__init__(); self.raw = raw
    def forward(self, x): return self.raw

def step_controller(obs, st):
    q = norm_q(np.array(obs['qpos'])); img = prep(obs['images']['top'])
    t0 = now()
    with torch.no_grad():
        if PREC == 'amp':
            # DINOv2 fp32 (as in §7), rest fp16: run the backbone outside autocast, feed its output through a
            # backbone-bypass so step()'s remaining math runs under autocast
            raw = policy.shared_backbone(img)
            t1 = now()
            bb = policy.shared_backbone; policy.shared_backbone = _Cached(raw)
            try:
                with amp(): chunk, st['h'] = policy.step(q, {'top': img}, st['h'])
            finally:
                policy.shared_backbone = bb
        else:
            t1 = None
            chunk, st['h'] = policy.step(q, {'top': img}, st['h'])
    t2 = now()
    return chunk.float(), ((t1 - t0) if t1 else float('nan')), t2 - t0

def fs_controller(obs, st):
    q = norm_q(np.array(obs['qpos'])); img = prep(obs['images']['top'])
    t0 = now()
    with torch.no_grad():
        dino = policy.shared_backbone(img)
        with amp(): vf = policy.process_vision_chunk(dino)
        t1 = now()
        st['vf'].append(vf); st['q'].append(q)
        with amp():
            seq = policy.forward_sequence(torch.cat(st['q'], 0).unsqueeze(0), {'top': torch.cat(st['vf'], 0).unsqueeze(0)})
        chunk = seq[:, -1]
    t2 = now()
    return chunk.float(), t1 - t0, t2 - t0

# warm-up (cuDNN autotune, Triton JIT of the chunked scan / step kernels) — excluded from timing
WARMUP = int(os.environ.get('WARMUP', '20'))
for mode in MODES:
    BOX_POSE[0] = sample_box_pose(); ts = env.reset()
    st = {'h': init_h(), 'vf': [], 'q': []}
    for _ in range(WARMUP):
        (step_controller if mode == 'step' else fs_controller)(ts.observation, st)
print('warm-up done', flush=True)

for mode in MODES:
    ctrl = step_controller if mode == 'step' else fs_controller
    np.random.seed(SEED); torch.manual_seed(SEED)        # same pose sequence for both modes
    recs = []
    for rid in range(N):
        BOX_POSE[0] = sample_box_pose(); ts = env.reset()
        st = {'h': init_h(), 'vf': [], 'q': []}
        ata = torch.zeros([MAXT, MAXT + NQ, SD]).cuda()
        vis, ctl, agg, envt, tot, rewards = [], [], [], [], [], []
        t_roll = time.perf_counter()
        for tt in range(MAXT):
            ta = now()
            chunk, tv, tc = ctrl(ts.observation, st)
            tb = now()
            ata[[tt], tt:tt + NQ] = denorm(chunk)
            cur = ata[:, tt]; pop = torch.all(cur != 0, dim=1); cur = cur[pop]
            w = np.exp(-0.01 * np.arange(len(cur))); w = torch.from_numpy(w / w.sum()).float().cuda().unsqueeze(1)
            target = (cur * w).sum(dim=0).cpu().numpy()
            tcc = now()
            ts = env.step(target); rewards.append(ts.reward)
            td = time.perf_counter()
            vis.append(tv); ctl.append(tc); agg.append(tcc - tb); envt.append(td - tcc); tot.append(td - ta)
        wall = time.perf_counter() - t_roll
        rw = np.array([0 if r is None else r for r in rewards]); hi = float(rw.max())
        recs.append({'rollout': rid, 'highest': hi, 'success': bool(hi == env_max_reward), 'wall_s': wall,
                     'vision_ms': [1e3 * x for x in vis], 'controller_ms': [1e3 * x for x in ctl],
                     'agg_ms': [1e3 * x for x in agg], 'env_ms': [1e3 * x for x in envt], 'tick_ms': [1e3 * x for x in tot]})
        c = np.array(recs[-1]['controller_ms']); k = np.array(recs[-1]['tick_ms'])
        print(f'[{mode}/{PREC}] rollout {rid}: success={hi == env_max_reward} wall={wall:.1f}s | controller ms '
              f't<50 {c[:50].mean():.1f} t>350 {c[350:].mean():.1f} mean {c.mean():.1f} | tick mean {k.mean():.1f}', flush=True)
    C = np.array([r['controller_ms'] for r in recs]); K = np.array([r['tick_ms'] for r in recs])
    summ = {'mode': mode, 'prec': PREC, 'seed': SEED, 'n': N, 'gpu': torch.cuda.get_device_name(0), 'ckpt': os.path.basename(ckpt),
            'success': int(sum(r['success'] for r in recs)),
            'controller_ms_mean': float(C.mean()), 'controller_ms_first50': float(C[:, :50].mean()), 'controller_ms_last50': float(C[:, 350:].mean()),
            'tick_ms_mean': float(K.mean()), 'rollout_wall_s_mean': float(np.mean([r['wall_s'] for r in recs]))}
    print('SUMMARY', json.dumps(summ), flush=True)
    json.dump({'summary': summ, 'rollouts': recs}, open(os.path.join(OUT, f'timing_{mode}_{PREC}_seed{SEED}.json'), 'w'))
print('TIMING DONE', flush=True)
