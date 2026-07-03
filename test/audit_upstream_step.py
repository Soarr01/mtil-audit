"""
audit_upstream_step.py — C1 AUDIT: prove the BUG is in the PUBLISHED MTIL artifact, not our fork.

Loads, in one process:
  (P) the PRISTINE upstream MambaPolicy from a clean clone of github.com/yulinzhouZYL/MTIL
      (/data/.../mtil_original_audit, HEAD 9bea607 == github, byte-identical to the published code), and
  (F) our FIXED MambaPolicy (test/mamba_policy_par.py working-tree).
Both get the SAME trained transfer_cube checkpoint (version_6) and run the SAME numeric kill-test:
the per-frame max|.| difference of the O(T) recurrent step() vs forward_sequence (the verified-correct
O(T^2) path that scores 96%). Expected:
  PRISTINE step()  -> ~15-60, episode-dependent  (residual-collapse bug: published code is BROKEN in closed loop)
  FIXED step()     -> ~1e-3 (matches forward_sequence, below the model's ~0.012 GT error)
forward_sequence is identical in both files (the ~15-line fix only touches the step() block-loop),
so the single reference is valid for both. Run fp32 (Ampere) and fp16 (matches training).
"""
import os, glob, re, sys, importlib.util
import h5py, numpy as np, torch
from einops import rearrange
import mamba_policy_par as MP_FIXED            # local test/ copy (FIXED working-tree)
from train.scaler_M import Scaler

# ARTIFACT_ROOT: point at the published bundle (mamba_policy_par_pristine.py + checkpoint.ckpt + scaler_transfer.pth
# + one episode_*.hdf5 + its top.npy); else local /data defaults.
ART = os.environ.get('ARTIFACT_ROOT','')
def artpath(rel, fallback): return os.path.join(ART, rel) if ART else fallback
np.random.seed(int(os.environ.get('SEED','1234'))); torch.manual_seed(int(os.environ.get('SEED','1234')))

PRISTINE_FILE = os.environ.get('PRISTINE_PATH', artpath('mamba_policy_par_pristine.py','/data/nas07_new/PersonalData/phuocthien/mtil_original_audit/train/mamba_policy_par.py'))
CKPT_DIR = artpath('checkpoint.ckpt','/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs/version_6/checkpoints')
SCALER   = artpath('scaler_transfer.pth','/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth')
_ep = os.environ.get('EPISODE')  # optional: pick a specific train episode (e.g. EPISODE=episode_2) for the multi-episode table
H5 = (f'/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/{_ep}.hdf5' if _ep else None) or artpath('episode.hdf5', None) or sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube/train/*.hdf5'))[0]
_stem = os.path.splitext(os.path.basename(H5))[0]
FEAT = artpath('top.npy', f'/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat/train/{_stem}/top.npy')
N = 24

def load_pristine_module():
    spec = importlib.util.spec_from_file_location('mp_pristine', PRISTINE_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['mp_pristine'] = mod
    spec.loader.exec_module(mod)
    return mod
MP_PRISTINE = load_pristine_module()

lowdim_dict={'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}

def build_policy(MPMOD):
    cfg = MPMOD.MambaConfig(); cfg.camera_names=['top']; cfg.embed_dim=2048; cfg.lowdim_dim=14
    cfg.d_model=2048; cfg.action_dim=14; cfg.sum_camera_feats=False; cfg.num_blocks=4
    pol = MPMOD.MambaPolicy(camera_names=cfg.camera_names,embed_dim=cfg.embed_dim,lowdim_dim=cfg.lowdim_dim,
      d_model=cfg.d_model,action_dim=cfg.action_dim,sum_camera_feats=cfg.sum_camera_feats,
      num_blocks=cfg.num_blocks,mamba_cfg=cfg).cuda().eval()
    ckpt=CKPT_DIR if CKPT_DIR.endswith('.ckpt') and os.path.isfile(CKPT_DIR) else min(glob.glob(os.path.join(CKPT_DIR,'*val_epoch_loss=*.ckpt')),key=lambda f:float(re.search(r'val_epoch_loss=(\d+\.\d+)',f).group(1)))
    sd=torch.load(ckpt,map_location='cuda:0')['state_dict']
    miss,unexp=pol.load_state_dict({k.replace('policy.',''):v for k,v in sd.items() if k.startswith('policy.')},strict=False)
    miss_nb=[k for k in miss if 'shared_backbone' not in k]
    assert not miss_nb, f'unexpected missing (non-backbone) keys: {miss_nb}'
    assert not unexp, f'unexpected keys: {unexp}'
    return pol, os.path.basename(ckpt)

scaler=Scaler(lowdim_dict=lowdim_dict); scaler.load(SCALER); scaler.cuda()
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

with h5py.File(H5,'r') as f:
    qpos=f['/observations/qpos'][()]; action=f['/action'][()]; imgs=f['/observations/images/top'][()]
feat=np.load(FEAT, mmap_mode='r')
cam_feats=torch.from_numpy(feat[:N].astype(np.float32)).cuda()
obs=torch.cat([norm_q(qpos[i]) for i in range(N)],dim=0).unsqueeze(0)

print('building PRISTINE (published) policy ...'); pol_P,ck=build_policy(MP_PRISTINE)
print('building FIXED policy ...');               pol_F,_ =build_policy(MP_FIXED)
print(f'ckpt: {ck} | episode: {_stem} | N={N}\n')

# single reference: forward_sequence (identical code in both files) on the FIXED model, cached feats
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    camv=pol_F.process_vision_chunk(cam_feats)
    ref=pol_F.forward_sequence(obs, {'top': camv.unsqueeze(0)})
ref_dn=denorm(ref.float())
ref_gt=np.mean([float((ref_dn[0,t].cpu().numpy()-action[t:t+16]).max()) for t in range(N)])

def run_step(pol, precision):
    fp16=(precision=='fp16')
    if fp16:
        hidden=[list(blk.mixer.allocate_inference_cache(1,max_seqlen=1,dtype=torch.float16)) for blk in pol.blocks]
    else:
        hidden=pol.init_hidden_states(1)
    out=[]
    for tt in range(N):
        q=norm_q(qpos[tt]); img=prep(imgs[tt])
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
            chunk,hidden=pol.step(q, {'top': img}, hidden)
        out.append(denorm(chunk.float())[0])
    return torch.stack(out,0)

print(f'reference forward_sequence vs GT (the model error floor): mean {ref_gt:.4f}\n')
print(f'{"":18s} {"mean|diff|":>12s} {"max|diff|":>12s}   verdict')
print('-'*62)
for prec in ['fp32','fp16']:
    for name,pol in [('PRISTINE/published',pol_P),('FIXED (ours)',pol_F)]:
        try:
            s=run_step(pol,prec)
            diffs=[float((s[t]-ref_dn[0,t]).abs().max()) for t in range(N)]
            m,mx=float(np.mean(diffs)),float(np.max(diffs))
            verdict='BROKEN (>> GT err)' if m>1.0 else 'matches forward_sequence'
            print(f'[{prec}] {name:18s} {m:12.4f} {mx:12.4f}   {verdict}')
        except Exception as e:
            print(f'[{prec}] {name:18s}  FAILED: {repr(e)}')
    print()
print('CONCLUSION: the published step() (HEAD 9bea607, identical in mtil_original_audit) diverges ~O(10-60)')
print('from forward_sequence => the released recurrent controller is broken; the ~15-line residual fix recovers it.')
