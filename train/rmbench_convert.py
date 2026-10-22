"""
rmbench_convert.py — convert RMBench demo_clean episodes (RoboTwin 2.0 / aloha-agilex, XPolicyLab hdf5 v1.0 as shipped
in the HF demo_clean.zip) into the MTIL/ALOHA hdf5 layout, so the unchanged MTIL pipeline (DINOv2 feature extraction
-> scaler -> train_par / windowed) applies.
State/action (14-D, joint space, same order as the simulator's obs['joint_action']['vector'] and take_action('qpos')):
  [left_arm 6, left gripper 1, right_arm 6, right gripper 1]; the file stores state/* and action/* with
  action[t] == state[t+1] exactly (verified on episode 0), i.e. RoboTwin's joint-space ACT/DP convention.
Camera: head camera RGB (240x320), decoded with RMBench's own decode_image_bit (legacy vs marked JPEG handled; RGB
out, as the simulator hands the policy at eval), stored raw — the MTIL transform resizes to 480x640 as for ALOHA.
Usage: python rmbench_convert.py <task> [n_val=5]
  src: ssm_vla/Chronos/RMBench/data/<task>/<task>/aloha_agilex/data/episode_XXXXXXX.hdf5
  dst: mtil_data/rmbench_<task>/{train,test}/episode_{i}.hdf5   (last n_val episodes -> test = checkpoint-selection set)
"""
import os, sys, glob, re
import h5py, numpy as np
from rmbench_decode_image_bit import decode_image_bit   # verbatim copy of RMBench data/decode_image_bit.py

task = sys.argv[1] if len(sys.argv) > 1 else 'put_back_block'
n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SRC = f'/data/nas07_new/PersonalData/phuocthien/ssm_vla/Chronos/RMBench/data/{task}/{task}/aloha_agilex/data'
DST = f'/data/nas07_new/PersonalData/phuocthien/mtil_data/rmbench_{task}'
eps = sorted(glob.glob(os.path.join(SRC, 'episode_*.hdf5')), key=lambda p: int(re.search(r'episode_(\d+)', p).group(1)))
print(f'{task}: {len(eps)} episodes -> {DST} ({len(eps) - n_val} train / {n_val} test)', flush=True)


def vec14(f, g):
    return np.concatenate([f[f'{g}/left_arm_joint_states'][()], f[f'{g}/left_ee_joint_states'][()],
                           f[f'{g}/right_arm_joint_states'][()], f[f'{g}/right_ee_joint_states'][()]], 1).astype(np.float32)


lens = []
for i, p in enumerate(eps):
    idx = int(re.search(r'episode_(\d+)', p).group(1))
    split = 'test' if i >= len(eps) - n_val else 'train'
    os.makedirs(os.path.join(DST, split), exist_ok=True)
    out = os.path.join(DST, split, f'episode_{idx}.hdf5')
    if os.path.exists(out): continue
    with h5py.File(p, 'r') as f:
        s, a = vec14(f, 'state'), vec14(f, 'action')
        assert np.abs(a[:-1] - s[1:]).max() < 1e-6, 'action[t] != state[t+1]'
        imgs = decode_image_bit(f['vision/cam_head/colors'][()])                     # [T,240,320,3] RGB
    with h5py.File(out + '.tmp', 'w') as g:
        g.create_dataset('observations/qpos', data=s)
        g.create_dataset('action', data=a)
        g.create_dataset('observations/images/top', data=imgs, chunks=(1,) + imgs.shape[1:])
    os.replace(out + '.tmp', out)
    lens.append(len(s))
    print(f'  [{split}] episode {idx}: T={len(s)}', flush=True)
if lens: print(f'lengths: min {min(lens)} max {max(lens)} mean {np.mean(lens):.0f}')
print('CONVERT DONE', flush=True)
