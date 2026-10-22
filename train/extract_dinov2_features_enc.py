"""
extract_dinov2_features_enc.py — cache DINOv2 ViT-S/14 or ViT-B/14 features for transfer_cube (lighter-encoder
experiment). Same pipeline as extract_dinov2_features.py (same transform, same adaptive resize, same
4th-from-last block, fp32 storage); only the hub model differs.
Usage (from MTIL/train): CUDA_VISIBLE_DEVICES=N python -u extract_dinov2_features_enc.py vits14
       DATA_ROOT=<hdf5 root> OUTPUT_ROOT=<feat root> ... vitl14   (any dataset in the ALOHA hdf5 layout, e.g. RMBench)
Output: mtil_data/transfer_cube_feat_<enc>/{train,test}/episode_X/{top.npy,action.npy,qpos.npy}
"""
import os, sys
import h5py, torch, numpy as np
from tqdm import tqdm
import mamba_policy_par as MP
from M_dataset_par import get_image_transform

ENC = sys.argv[1] if len(sys.argv) > 1 else 'vits14'
DATA_ROOT = os.environ.get('DATA_ROOT', "/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube")
OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT', f"/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat_{ENC}")
CAM = 'top'; BATCH_SIZE = 16; DEVICE = "cuda"; RESIZE_HW = (640, 480)


def make_frozen_dino(enc, layer_index=-4):
    if enc == 'vitl14': return MP.FrozenDinov2(layer_index=layer_index)     # the paper's encoder, unchanged
    fd = MP.FrozenDinov2.__new__(MP.FrozenDinov2); torch.nn.Module.__init__(fd)
    fd.dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_{enc}')
    fd.patch_size = 14; fd.layer_index = layer_index
    for p in fd.dino.parameters(): p.requires_grad_(False)
    fd.feature_hook = fd._register_hook()
    return fd


def extract_for_split(mode, backbone, tf):
    ind = os.path.join(DATA_ROOT, mode); outd = os.path.join(OUTPUT_ROOT, mode); os.makedirs(outd, exist_ok=True)
    files = sorted(f for f in os.listdir(ind) if f.endswith('.hdf5') or f.endswith('.h5'))
    for fn in tqdm(files, desc=f"{ENC} {mode}", mininterval=30):
        td = os.path.join(outd, os.path.splitext(fn)[0]); os.makedirs(td, exist_ok=True)
        if os.path.exists(os.path.join(td, f'{CAM}.npy')): continue
        with h5py.File(os.path.join(ind, fn), 'r') as f:
            np.save(os.path.join(td, 'action.npy'), f['action'][:])
            np.save(os.path.join(td, 'qpos.npy'), f['observations/qpos'][:])
            imgs = f[f'observations/images/{CAM}'][:]
        feats = None
        for i in range(0, len(imgs), BATCH_SIZE):
            x = torch.stack([tf(im) for im in imgs[i:i + BATCH_SIZE]]).to(DEVICE)
            with torch.no_grad(): o = backbone(x).cpu().numpy()
            if feats is None: feats = np.zeros((len(imgs),) + o.shape[1:], dtype=np.float32)
            feats[i:i + len(o)] = o
        np.save(os.path.join(td, f'{CAM}.npy') + '.tmp.npy', feats)          # atomic: partial files never look done
        os.replace(os.path.join(td, f'{CAM}.npy') + '.tmp.npy', os.path.join(td, f'{CAM}.npy'))


if __name__ == "__main__":
    bb = make_frozen_dino(ENC).to(DEVICE).eval(); tf = get_image_transform(RESIZE_HW)
    with torch.no_grad(): print(ENC, 'feature map', tuple(bb(torch.zeros(1, 3, 480, 640, device=DEVICE)).shape), flush=True)
    extract_for_split("train", bb, tf); extract_for_split("test", bb, tf)
    print("EXTRACT DONE", ENC, flush=True)
