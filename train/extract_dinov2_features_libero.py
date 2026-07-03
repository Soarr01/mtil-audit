"""
extract_dinov2_features_libero.py — DINOv2 feature extraction for LIBERO-Mem (C4).
Reuses the ALOHA pipeline: resize agentview_rgb (256x256) -> FrozenDinov2 -> [T,1024,35,46]
(VERIFIED identical to ALOHA, so spatial_adapter is reused unchanged; only aspect distortion).
LIBERO hdf5 layout: data/demo_X/{actions [T,7], obs/agentview_rgb [T,256,256,3],
obs/ee_pos[T,3], obs/ee_ori[T,3], obs/gripper_states[T,2]}.
Saves per-demo: agentview.npy [T,1024,35,46], action.npy [T,7], qpos.npy [T,8] (proprio).
"""
import os, glob, h5py, numpy as np, torch
from tqdm import tqdm
from mamba_policy_par import FrozenDinov2
from M_dataset_par import get_image_transform

DATA_FILES = sorted(glob.glob('/data/nas07_new/PersonalData/phuocthien/mtil_data/libero_mem/full/*demo.hdf5'))
OUTPUT_ROOT = '/data/nas07_new/PersonalData/phuocthien/mtil_data/libero_mem_feat'
BATCH_SIZE = 16
DEVICE = 'cuda'

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print('Loading DINOv2...'); bb = FrozenDinov2(layer_index=-4).to(DEVICE).eval()
tf = get_image_transform((640, 480))   # -> FrozenDinov2 outputs [.,1024,35,46]

def proprio(d):
    # 8-dim: ee_pos(3) + ee_ori(3) + gripper_states(2)
    return np.concatenate([d['obs']['ee_pos'][:], d['obs']['ee_ori'][:], d['obs']['gripper_states'][:]], axis=1).astype(np.float32)

for fpath in DATA_FILES:
    task = os.path.splitext(os.path.basename(fpath))[0]
    out_task = os.path.join(OUTPUT_ROOT, task); os.makedirs(out_task, exist_ok=True)
    with h5py.File(fpath, 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda s: int(s.split('_')[1]))
        print(f'\n=== {task}: {len(demos)} demos ===', flush=True)
        for dm in tqdm(demos, desc=task[:30]):
            traj_dir = os.path.join(out_task, dm); os.makedirs(traj_dir, exist_ok=True)
            if os.path.exists(os.path.join(traj_dir, 'agentview.npy')):  # resume
                continue
            d = f['data'][dm]
            np.save(os.path.join(traj_dir, 'action.npy'), d['actions'][:].astype(np.float32))
            np.save(os.path.join(traj_dir, 'qpos.npy'), proprio(d))
            imgs = d['obs']['agentview_rgb']; T = imgs.shape[0]
            feats = np.zeros((T, 1024, 35, 46), dtype=np.float32)
            for i in range(0, T, BATCH_SIZE):
                batch = torch.stack([tf(imgs[j]) for j in range(i, min(i+BATCH_SIZE, T))]).to(DEVICE)
                with torch.no_grad():
                    feats[i:i+batch.shape[0]] = bb(batch).cpu().numpy()
            np.save(os.path.join(traj_dir, 'agentview.npy'), feats)
print('\nDONE. Features at', OUTPUT_ROOT)
