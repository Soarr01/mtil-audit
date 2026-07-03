"""DINOv2 feature precompute for LIBERO-Mem (C4 memory experiment), full-replication of the mtil_plus
gold pipeline (docs/LIBERO_SR_FIX.md): dual camera (agentview_rgb + eye_in_hand_rgb) resized to a SQUARE
128x128 (no 640x480 aspect distortion), RAW DINOv2 layer-(-4) map per camera, proprio = robot_states[9]
(gripper_qpos2 + eef_pos3 + eef_quat4, VERIFIED == mtil_plus), action[7]. Reads the LIBERO-Mem HDF5
directly -> NO libero env needed -> runs in `ssm_vla`.

Layout: <OUT>/<task>/demo_<i>/{agentview.npy [T,1024,10,10], eye_in_hand.npy, qpos.npy [T,9], action.npy [T,7]}

Run (smoke 1 demo first, then full):
  cd MTIL/train && CUDA_VISIBLE_DEVICES=<free> python extract_dinov2_features_libero_mem.py \
      --task KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate [--limit-demos 1]
"""
import argparse
import glob
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from mamba_policy_libero import FrozenDinov2

NAS = "/data/nas07_new/PersonalData/phuocthien"
CAM_MAP = {'agentview': 'agentview_rgb', 'eye_in_hand': 'eye_in_hand_rgb'}  # policy name -> HDF5 obs key


def get_image_transform(size):
    # ToPILImage -> Resize(square) -> ToTensor[0,1]; NO imagenet norm (matches DINOv2 usage in this repo)
    return transforms.Compose([transforms.ToPILImage(), transforms.Resize(size), transforms.ToTensor()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, help='LIBERO-Mem task name WITHOUT the _demo suffix')
    ap.add_argument('--cameras', default='agentview,eye_in_hand')
    ap.add_argument('--data-root', default=f"{NAS}/mtil_data/libero_mem/full")
    ap.add_argument('--out-root', default=f"{NAS}/mtil_data/libero_mem_feat2")
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit-demos', type=int, default=None, help='cap demos (smoke test)')
    ap.add_argument('--fp16', action='store_true', default=True)
    args = ap.parse_args()

    cams = args.cameras.split(',')
    dev = 'cuda'
    out_dir = os.path.join(args.out_root, args.task)
    os.makedirs(out_dir, exist_ok=True)

    backbone = FrozenDinov2(layer_index=-4).to(dev).eval()
    transform = get_image_transform((128, 128))

    tf_path = os.path.join(args.data_root, f"{args.task}_demo.hdf5")
    assert os.path.exists(tf_path), f"missing {tf_path}"
    dtype = np.float16 if args.fp16 else np.float32
    with h5py.File(tf_path, 'r') as f:
        data = f['data']
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split('_')[-1]))
        if args.limit_demos:
            demo_keys = demo_keys[:args.limit_demos]
        print(f"[extract] task={args.task} cams={cams} demos={len(demo_keys)} -> {out_dir} (fp16={args.fp16})")
        for dk in tqdm(demo_keys, desc=args.task[:40]):
            demo = data[dk]
            ddir = os.path.join(out_dir, dk)
            if os.path.exists(os.path.join(ddir, f"{cams[-1]}.npy")):
                continue  # resume
            os.makedirs(ddir, exist_ok=True)
            T = demo['actions'].shape[0]
            np.save(os.path.join(ddir, 'action.npy'), demo['actions'][:].astype(np.float32))      # [T,7]
            np.save(os.path.join(ddir, 'qpos.npy'), demo['robot_states'][:, :9].astype(np.float32))  # [T,9]
            for cam in cams:
                imgs = demo['obs'][CAM_MAP[cam]]   # [T,256,256,3] uint8
                feats = None
                for i in range(0, T, args.batch_size):
                    end = min(i + args.batch_size, T)
                    bt = torch.stack([transform(imgs[j]) for j in range(i, end)]).to(dev)
                    with torch.no_grad():
                        out = backbone(bt)   # [b,1024,h,w]
                    if feats is None:
                        feats = np.zeros((T,) + tuple(out.shape[1:]), dtype=dtype)
                    feats[i:end] = out.cpu().numpy().astype(dtype)
                np.save(os.path.join(ddir, f'{cam}.npy'), feats)
    print(f"[extract] done task={args.task}")


if __name__ == '__main__':
    main()
