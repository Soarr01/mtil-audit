"""
build_shm_cache.py — one-time fp16 copy of a cached-feature tree into /dev/shm (RAM-backed), so that every C3
training run reads features at memory speed instead of re-reading 123 GB over the contended NAS (~50-400 MB/s).
fp16 storage is EXACT for training: the first op on the features (spatial_adapter Conv2d) runs under
precision='16-mixed' autocast, which casts its fp32 input to fp16 with the same round-to-nearest.
Usage: python build_shm_cache.py <src_root> <dst_root>
  e.g. python build_shm_cache.py /data/.../mtil_data/transfer_cube_feat /dev/shm/c3feat_vitl14
Remove with: rm -rf <dst_root>
"""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

src, dst = sys.argv[1], sys.argv[2]
jobs = []
for split in ('train', 'test'):
    for ep in sorted(os.listdir(os.path.join(src, split))):
        if os.path.isdir(os.path.join(src, split, ep)): jobs.append((split, ep))


def one(j):
    split, ep = j; s = os.path.join(src, split, ep); d = os.path.join(dst, split, ep)
    if os.path.exists(os.path.join(d, 'DONE')): return ep
    os.makedirs(d, exist_ok=True)
    for f in ('qpos.npy', 'action.npy'):
        np.save(os.path.join(d, f), np.load(os.path.join(s, f)))
    np.save(os.path.join(d, 'top.npy'), np.load(os.path.join(s, 'top.npy')).astype(np.float16))
    open(os.path.join(d, 'DONE'), 'w').close()
    return ep


t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    for i, ep in enumerate(ex.map(one, jobs)):
        print(f'{i + 1}/{len(jobs)} {ep} {time.time() - t0:.0f}s', flush=True)
open(os.path.join(dst, 'COMPLETE'), 'w').close()
print('SHM CACHE DONE', dst, flush=True)
