"""
M_dataset_libero.py — LIBERO-Mem dataset + scaler (C4). Generic n_obs/n_act (default 8 proprio, 7 action).
Reads libero_mem_feat/<task>/demo_X/{agentview.npy [T,1024,35,46], action.npy [T,7], qpos.npy [T,8]}.
Returns obs [L,8], actions [L,16,7] (normalized), rgb {agentview:[L,1024,35,46]}. Reuses parallel_collate_fn.
Run as main to FIT + SAVE the scaler for a task.
"""
import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scaler_M import Scaler
from M_dataset_par import parallel_collate_fn

FEAT_ROOT = '/data/nas07_new/PersonalData/phuocthien/mtil_data/libero_mem_feat'

class LiberoMemDataset(Dataset):
    def __init__(self, feat_root, task, mode='train', scaler=None, future_steps=16,
                 n_val=10, n_obs=8, n_act=7, cam='agentview'):
        super().__init__()
        self.future_steps=future_steps; self.n_obs=n_obs; self.n_act=n_act; self.cam=cam
        td=os.path.join(feat_root, task)
        demos=sorted([d for d in os.listdir(td) if d.startswith('demo_') and os.path.isdir(os.path.join(td,d))],
                     key=lambda s:int(s.split('_')[1]))
        demos=[os.path.join(td,d) for d in demos if os.path.exists(os.path.join(td,d,cam+'.npy'))]
        self.records = demos[:-n_val] if mode=='train' else demos[-n_val:]
        print(f"[{task}/{mode}] {len(self.records)} demos (of {len(demos)} extracted)")
        self.obs_keys=[f'obs_{i}' for i in range(n_obs)]; self.act_keys=[f'act_{i}' for i in range(n_act)]
        self.lowdim_keys=self.obs_keys+self.act_keys
        self.lowdim_shapes={**{k:1 for k in self.obs_keys}, **{k:(future_steps,1) for k in self.act_keys}}
        self.scaler=scaler; self.fitting=False

    def __len__(self): return len(self.records)

    def _raw(self, idx):
        rp=self.records[idx]
        qpos=np.array(np.load(os.path.join(rp,'qpos.npy'), mmap_mode='r'))   # [T,n_obs]
        act=np.array(np.load(os.path.join(rp,'action.npy'), mmap_mode='r'))  # [T,n_act]
        L=qpos.shape[0]
        def tt(a):
            t=torch.from_numpy(np.ascontiguousarray(a)).float(); return t.unsqueeze(1) if t.dim()==1 else t
        raw={f'obs_{i}': tt(qpos[:,i]) for i in range(self.n_obs)}
        fp=np.pad(act, ((0,self.future_steps-1),(0,0)), mode='edge')
        chunks=np.lib.stride_tricks.as_strided(fp, (L,self.future_steps,act.shape[1]),
                                               (fp.strides[0],fp.strides[0],fp.strides[1]))
        for i in range(self.n_act): raw[f'act_{i}']=torch.from_numpy(np.ascontiguousarray(chunks[:,:,i:i+1])).float()
        return rp, qpos, raw, L

    def __getitem__(self, idx):
        rp, qpos, raw, L = self._raw(idx)
        if self.fitting: return {'lowdim_raw': raw}
        feat=torch.from_numpy(np.array(np.load(os.path.join(rp,self.cam+'.npy'), mmap_mode='r')))  # [L,1024,35,46]
        norm=self.scaler.normalize(raw)
        obs=torch.cat([norm[k] for k in self.obs_keys], dim=1)        # [L,n_obs]
        actions=torch.cat([norm[k] for k in self.act_keys], dim=2)    # [L,future,n_act]
        return {'obs':obs, 'actions':actions, 'rgb':{self.cam:feat}, 'length':L}

    def fit_scaler(self, num_workers=4):
        self.scaler=Scaler(lowdim_dict=self.lowdim_shapes); self.fitting=True
        cache={k:[] for k in self.lowdim_keys}
        loader=DataLoader(self, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=lambda x:x[0])
        for item in tqdm(loader, desc='fit scaler'):
            for k in self.lowdim_keys: cache[k].append(item['lowdim_raw'][k])
        self.scaler.fit({k:torch.cat(v,dim=0) for k,v in cache.items()})
        self.fitting=False; return self.scaler

if __name__=='__main__':
    import sys
    task=sys.argv[1] if len(sys.argv)>1 else 'KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo'
    save=sys.argv[2] if len(sys.argv)>2 else f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_libero_{task[:20]}.pth'
    ds=LiberoMemDataset(FEAT_ROOT, task, mode='train')
    ds.fit_scaler(num_workers=4); ds.scaler.save(save)
    print('scaler saved:', save, '| keys:', len(ds.lowdim_keys))
    ds.scaler.load(save); ds.scaler  # sanity
    it=ds[0]; print('obs', it['obs'].shape, '| actions', it['actions'].shape, '| feat', it['rgb'][ds.cam].shape)
