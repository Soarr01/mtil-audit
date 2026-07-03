"""LIBERO-Mem dataset (C4) — full-replication of the mtil_plus gold pipeline (docs/LIBERO_SR_FIX.md).

Single-task. Loads the dual-cam 128^2 cache (libero_mem_feat2/<task>/demo_i/): RAW DINOv2 maps per camera
[L,1024,10,10] + qpos[L,9] + action[L,7]. Returns the FULL trajectory (windowing, if any, is done in the
trainer), per-dim z-scored qpos/action via Scaler({'qpos':9,'act':7}) clamped to [-5,5] (the fp16/bf16
overflow guard from mtil_plus). No language (single-task -> lang_dim=None model). Per-task train/val split.

Fit+save scaler:  python M_dataset_libero_mem.py --task <TASK> --save-scaler <P>/mtil_data/scaler_libero_mem_<...>.pth
"""
import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scaler_M import Scaler

NAS = "/data/nas07_new/PersonalData/phuocthien"
FEAT_ROOT = os.environ.get('FEAT_ROOT', f"{NAS}/mtil_data/libero_mem_feat2")  # override to /tmp to avoid NAS I/O stalls


class LiberoMemDataset(Dataset):
    def __init__(self, feat_root: str, task: str, mode: str = "train",
                 cameras: List[str] = ('agentview', 'eye_in_hand'),
                 future_steps: int = 8, scaler: Optional[Scaler] = None,
                 val_frac: float = 0.1, lowdim_dim: int = 9, action_dim: int = 7):
        super().__init__()
        self.cameras = list(cameras)
        self.future_steps = future_steps
        self.lowdim_dim = lowdim_dim
        self.action_dim = action_dim
        self.mode = mode
        task_dir = os.path.join(feat_root, task)
        demos = sorted(glob.glob(os.path.join(task_dir, 'demo_*')), key=lambda d: int(d.split('_')[-1]))
        assert demos, f"no demos under {task_dir}"
        n_val = max(1, int(len(demos) * val_frac))
        self.records = demos[:-n_val] if mode == 'train' else demos[-n_val:]
        print(f"[{mode}] {len(self.records)} demos from {task}")

        self.lowdim_shapes = {'qpos': lowdim_dim, 'act': action_dim}
        self.scaler = scaler if scaler is not None else Scaler(lowdim_dict=self.lowdim_shapes)
        self.fitting = False
        # PRELOAD=1: read all features fully into RAM ONCE (no mmap) -> no per-step file-backed page faults
        # (the folio_wait_bit_common D-state stalls on an I/O-contended shared server). Abundant RAM (251GB).
        self.preload = os.environ.get('PRELOAD') == '1'
        self.cache = {}
        if self.preload:
            for rp in tqdm(self.records, desc=f'preload[{mode}]'):
                q = np.array(np.load(os.path.join(rp, 'qpos.npy'))).astype(np.float32)
                a = np.array(np.load(os.path.join(rp, 'action.npy'))).astype(np.float32)
                rgb = {cam: np.array(np.load(os.path.join(rp, f'{cam}.npy'))) for cam in self.cameras}  # fp16 in RAM
                self.cache[rp] = (q, a, rgb)

    def __len__(self):
        return len(self.records)

    def _load_raw(self, rp):
        if self.preload and rp in self.cache:
            c = self.cache[rp]; return c[0], c[1]
        qpos = np.array(np.load(os.path.join(rp, 'qpos.npy'), mmap_mode='r')).astype(np.float32)   # [L,9]
        act = np.array(np.load(os.path.join(rp, 'action.npy'), mmap_mode='r')).astype(np.float32)  # [L,7]
        return qpos, act

    def _act_chunks(self, act):
        L, H = act.shape[0], self.future_steps
        if H > 0:
            ap = np.pad(act, ((0, H - 1), (0, 0)), mode='edge')
            chunks = np.lib.stride_tricks.as_strided(
                ap, shape=(L, H, act.shape[1]),
                strides=(ap.strides[0], ap.strides[0], ap.strides[1])).copy()
        else:
            chunks = act[:, None, :]
        return chunks  # [L,H,7]

    def __getitem__(self, idx):
        rp = self.records[idx]
        qpos, act = self._load_raw(rp)
        chunks = self._act_chunks(act)              # [L,H,7]
        if self.fitting:
            return {'qpos': torch.from_numpy(qpos), 'act': torch.from_numpy(act)}  # raw for scaler fit
        L = qpos.shape[0]
        # clamp z-scored values to [-5,5]: small-std dim + outlier -> huge normalized magnitude -> fp16/bf16 NaN
        qn = self.scaler.normalize({'qpos': torch.from_numpy(qpos)})['qpos'].clamp(-5, 5)   # [L,9]
        an = self.scaler.normalize({'act': torch.from_numpy(chunks)})['act'].clamp(-5, 5)   # [L,H,7]
        rgb = {}
        for cam in self.cameras:
            if self.preload and rp in self.cache:
                f = self.cache[rp][2][cam]                                            # from RAM (no mmap)
            else:
                f = np.load(os.path.join(rp, f'{cam}.npy'), mmap_mode='r')            # [L,1024,10,10] fp16
            rgb[cam] = torch.from_numpy(np.array(f)).float()
        return {'obs': qn.float(), 'actions': an.float(), 'rgb': rgb, 'length': L}

    def fit_scaler(self, num_workers=4):
        self.fitting = True
        loader = DataLoader(self, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=lambda x: x[0])
        qs, as_ = [], []
        for it in tqdm(loader, desc='fit scaler'):
            qs.append(it['qpos']); as_.append(it['act'])
        self.scaler.fit({'qpos': torch.cat(qs, 0), 'act': torch.cat(as_, 0)})  # per-dim mean/std
        self.fitting = False
        return self.scaler


def libero_mem_collate_fn(batch):
    B = len(batch)
    lengths = [b['length'] for b in batch]
    maxL = max(lengths)
    od = batch[0]['obs'].shape[1]
    ash = batch[0]['actions'].shape[1:]   # (H,7)
    obs = torch.zeros(B, maxL, od)
    actions = torch.zeros(B, maxL, *ash)
    mask = torch.ones(B, maxL, dtype=torch.bool)   # True = pad
    rgb_batch = {cam: [b['rgb'][cam] for b in batch] for cam in batch[0]['rgb']}
    for i, b in enumerate(batch):
        L = b['length']
        obs[i, :L] = b['obs']; actions[i, :L] = b['actions']; mask[i, :L] = False
    return {'obs': obs, 'actions': actions, 'mask': mask, 'rgb_batch': rgb_batch}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--feat-root', default=FEAT_ROOT)
    ap.add_argument('--future-steps', type=int, default=8)
    ap.add_argument('--save-scaler', default=None)
    args = ap.parse_args()
    ds = LiberoMemDataset(args.feat_root, args.task, mode='train', future_steps=args.future_steps)
    sc = ds.fit_scaler()
    print("scaler qpos mean[:3]:", sc.mean_dict['qpos'].data[:3].tolist())
    print("scaler act std[:3]:", sc.std_dict['act'].data[:3].tolist())
    if args.save_scaler:
        sc.save(args.save_scaler); print("saved scaler ->", args.save_scaler)
