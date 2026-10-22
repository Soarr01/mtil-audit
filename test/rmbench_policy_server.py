"""
rmbench_policy_server.py — serve an MTIL policy (this repo's test/ model code, ssm_vla env) to the RMBench simulator
(separate `rmbench` env: SAPIEN/torch-2.4, no mamba_ssm) over a local socket. The simulator-side adapter is
Chronos/RMBench/policy/MTIL/deploy_policy.py.

Controllers (same as the transfer_cube protocol):
  --mode full : C1-fixed recurrent step(), fp32, O(1) per tick (evaluate_step_fs.py)
  --mode win  : forward_sequence on the last k frames, fp16 autocast, current frame = last (evaluate_fs_windowed.py)
Both use ACT temporal ensembling (exp(-0.01 i)) over 16-step chunks, exactly as the ALOHA evaluators.
Image: RMBench head camera RGB uint8 240x320 -> the TRAINING transform of M_dataset_par.get_image_transform
(PIL Resize to 480x640, ToTensor) -> DINOv2 (fp32) inside the policy.
Protocol (multiprocessing.connection, pickled tuples of plain bytes, numpy-version independent):
  ('reset',) -> 'ok' ;  ('act', qpos_bytes(float32[14]), img_bytes(uint8[H,W,3]), H, W) -> action_bytes(float32[14])
Usage: PYTHONPATH=<MTIL> python rmbench_policy_server.py --task put_back_block --mode full --ckpt_dir <dir> [--k 16] [--port 6010]
"""
import os, sys, glob, re, argparse, time
import numpy as np, torch
from multiprocessing.connection import Listener
from torchvision import transforms
import mamba_policy_par as MP
from train.scaler_M import Scaler

ap = argparse.ArgumentParser()
ap.add_argument('--task', default='put_back_block')
ap.add_argument('--mode', choices=['full', 'win'], required=True)
ap.add_argument('--k', type=int, default=16)
ap.add_argument('--ckpt_dir', required=True)
ap.add_argument('--port', type=int, default=6010)
ap.add_argument('--maxt', type=int, default=520)          # >= RMBench step_lim (put_back_block: 500)
a = ap.parse_args()

lowdim_dict = {'agl_1':1,'agl_2':1,'agl_3':1,'agl_4':1,'agl_5':1,'agl_6':1,
 'agl2_1':1,'agl2_2':1,'agl2_3':1,'agl2_4':1,'agl2_5':1,'agl2_6':1,'gripper_pos':1,'gripper_pos2':1,
 'agl_1_act':(16,1),'agl_2_act':(16,1),'agl_3_act':(16,1),'agl_4_act':(16,1),'agl_5_act':(16,1),'agl_6_act':(16,1),
 'agl2_1_act':(16,1),'agl2_2_act':(16,1),'agl2_3_act':(16,1),'agl2_4_act':(16,1),'agl2_5_act':(16,1),'agl2_6_act':(16,1),
 'gripper_act':(16,1),'gripper_act2':(16,1)}
# 14-D order (RMBench joint_action.vector): [L arm 6, L grip, R arm 6, R grip] -> scaler keys below (same as ALOHA)
QK = ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos','agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']
AK = ['agl_1_act','agl_2_act','agl_3_act','agl_4_act','agl_5_act','agl_6_act','gripper_act','agl2_1_act','agl2_2_act','agl2_3_act','agl2_4_act','agl2_5_act','agl2_6_act','gripper_act2']
config = MP.MambaConfig(); config.camera_names = ['top']; config.embed_dim = 2048; config.lowdim_dim = 14
config.d_model = 2048; config.action_dim = 14; config.sum_camera_feats = False; config.num_blocks = 4
ckpt = min(glob.glob(os.path.join(a.ckpt_dir, '*val_epoch_loss=*.ckpt')), key=lambda f: float(re.search(r'val_epoch_loss=(\d+\.\d+)', f).group(1)))
policy = MP.MambaPolicy(camera_names=config.camera_names, embed_dim=config.embed_dim, lowdim_dim=config.lowdim_dim, d_model=config.d_model,
                        action_dim=config.action_dim, sum_camera_feats=config.sum_camera_feats, num_blocks=config.num_blocks, mamba_cfg=config).cuda().eval()
sd = torch.load(ckpt, map_location='cuda:0')['state_dict']
miss, unexp = policy.load_state_dict({k.replace('policy.', ''): v for k, v in sd.items() if k.startswith('policy.')}, strict=False)
print('ckpt:', ckpt, '| miss(non-bb):', [k for k in miss if 'shared_backbone' not in k], '| unexp:', unexp, flush=True)
scaler = Scaler(lowdim_dict=lowdim_dict); scaler.load(f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_rmbench_{a.task}.pth'); scaler.cuda()
tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 640)), transforms.ToTensor()])   # = M_dataset_par.get_image_transform((640,480))


def norm_q(q):
    d = scaler.normalize({k: torch.tensor([float(q[i])], device='cuda') for i, k in enumerate(QK)})
    return torch.stack([d[k].reshape(()) for k in QK]).float().unsqueeze(0)          # [1,14]


def denorm(c):                                                                        # c [16,14] normalized
    d = scaler.denormalize({k: c[:, i:i + 1] for i, k in enumerate(AK)})
    return torch.cat([d[k] for k in AK], dim=-1)


NQ, SD = 16, 14
st = {}


def reset():
    st['t'] = 0; st['ata'] = torch.zeros([a.maxt, a.maxt + NQ, SD], device='cuda')
    st['h'] = policy.init_hidden_states(1); st['vf'] = []; st['q'] = []


@torch.no_grad()
def act(q, img):
    t = st['t']; qn = norm_q(q); x = tf(img).unsqueeze(0).cuda()
    if a.mode == 'full':
        chunk, st['h'] = policy.step(qn, {'top': x}, st['h']); chunk = chunk[0].float()
    else:
        dino = policy.shared_backbone(x)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            vf = policy.process_vision_chunk(dino)
        st['vf'].append(vf); st['q'].append(qn)
        lo = max(0, len(st['q']) - a.k)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            seq = policy.forward_sequence(torch.cat(st['q'][lo:], 0).unsqueeze(0), {'top': torch.cat(st['vf'][lo:], 0).unsqueeze(0)})
        chunk = seq[0, -1].float()
    ata = st['ata']; ata[t, t:t + NQ] = denorm(chunk)
    cur = ata[:, t]; cur = cur[torch.all(cur != 0, dim=1)]
    w = torch.from_numpy(np.exp(-0.01 * np.arange(len(cur))) / np.exp(-0.01 * np.arange(len(cur))).sum()).float().cuda().unsqueeze(1)
    st['t'] = t + 1
    return (cur * w).sum(0).cpu().numpy().astype(np.float32)


reset()
lst = Listener(('localhost', a.port), authkey=b'mtil-rmbench')
print(f'[server] mode={a.mode} k={a.k} listening on {a.port}', flush=True)
while True:
    conn = lst.accept(); print('[server] client connected', flush=True)
    try:
        while True:
            msg = conn.recv()
            if msg[0] == 'reset':
                reset(); conn.send('ok')
            elif msg[0] == 'act':
                q = np.frombuffer(msg[1], np.float32); img = np.frombuffer(msg[2], np.uint8).reshape(msg[3], msg[4], 3)
                conn.send(act(q, img.copy()).tobytes())
            elif msg[0] == 'close':
                conn.close(); break
    except EOFError:
        print('[server] client disconnected', flush=True)
