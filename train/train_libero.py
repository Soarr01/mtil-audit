"""
train_libero.py — LIBERO-Mem training (C4). One script for all three context regimes:
  WINDOW_K=0  -> full-history (forward_sequence over whole episode)
  WINDOW_K=k  -> windowed (W random k-frame windows/step, fresh per window) [k=1 = single-frame]
Config: lowdim_dim=8 (proprio), action_dim=7, camera 'agentview', img resized to 640x480 -> [1024,35,46]
(spatial_adapter reused). Usage: train_libero.py <task> <WINDOW_K> [max_epochs]
"""
import os, sys, copy, glob, torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from M_dataset_libero import LiberoMemDataset, FEAT_ROOT
from M_dataset_par import parallel_collate_fn
from scaler_M import Scaler
from mamba_policy_par import MambaPolicy, MambaConfig

TASK = sys.argv[1] if len(sys.argv)>1 else 'KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate_demo'
WINDOW_K = int(sys.argv[2]) if len(sys.argv)>2 else 0          # 0=full-history; k>0=windowed (1=single-frame)
MAX_EPOCHS = int(sys.argv[3]) if len(sys.argv)>3 else 60
WINDOWS_PER_STEP = 16
CAM = 'agentview'
SCALER_PATH = f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_libero_{TASK[:20]}.pth'
TAG = 'full' if WINDOW_K==0 else f'win{WINDOW_K}'

class LitLibero(pl.LightningModule):
    def __init__(self, config, scaler):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.config=config; self.scaler=scaler; self.K=WINDOW_K; self.W=WINDOWS_PER_STEP
        self.policy=MambaPolicy(camera_names=config.camera_names, embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim, d_model=config.d_model, action_dim=config.action_dim,
            sum_camera_feats=config.sum_camera_feats, num_blocks=config.num_blocks,
            future_steps=16, img_size=config.img_size, mamba_cfg=config)
        self.policy.shared_backbone.eval()
        for p in self.policy.shared_backbone.parameters(): p.requires_grad=False
        if getattr(config,'load_features',False): self.policy.shared_backbone=None
        self.lr=1e-4; self.wd=1e-4; self.tr=[]; self.va=[]

    def configure_optimizers(self):
        opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,self.policy.parameters()),lr=self.lr,weight_decay=self.wd)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.trainer.max_epochs,eta_min=1e-6)
        return {"optimizer":opt,"lr_scheduler":sch}

    def compute_vision_features_chunked(self, rgb_batch):
        dev=self.device; cams=list(rgb_batch.keys())
        if not cams: return {}
        B=len(rgb_batch[cams[0]]); maxL=max(t.shape[0] for t in rgb_batch[cams[0]])
        fs={cam:torch.zeros(B,maxL,self.config.embed_dim,device=dev,dtype=torch.float32) for cam in self.config.camera_names}
        CH=16
        for b in range(B):
            for cam in self.config.camera_names:
                raw=rgb_batch[cam][b]; L=raw.shape[0]
                for i in range(0,L,CH):
                    img=raw[i:i+CH].to(dev,non_blocking=True)
                    dino=img if getattr(self.config,'load_features',False) else self.policy.shared_backbone(img)
                    fs[cam][b,i:i+CH]=self.policy.process_vision_chunk(dino)
                    del img,dino
        return fs

    def _sample_windows(self, obs, actions, mask, rgb):
        K,W=self.K,self.W; Lv=int((~mask[0]).sum().item())
        starts=[0] if Lv<=K else torch.randint(0,Lv-K+1,(W,)).tolist()
        ow=torch.stack([obs[0,s:s+K] for s in starts],0); aw=torch.stack([actions[0,s:s+K] for s in starts],0)
        mw=torch.stack([mask[0,s:s+K] for s in starts],0)
        rw={cam:[rgb[cam][0][s:s+K] for s in starts] for cam in rgb}
        return ow,aw,mw,rw

    def _loss(self, batch):
        obs=batch['obs'].to(self.device); actions=batch['actions'].to(self.device)
        mask=batch['mask'].to(self.device); rgb=batch['rgb_batch']
        if self.K==0:
            cam=self.compute_vision_features_chunked(rgb)
            pred=self.policy.forward_sequence(obs, cam); aw, mw = actions, mask
        else:
            ow,aw,mw,rw=self._sample_windows(obs,actions,mask,rgb)
            cam=self.compute_vision_features_chunked(rw)
            pred=self.policy.forward_sequence(ow, cam)
        loss=F.mse_loss(pred,aw,reduction='none').mean(dim=-1).mean(dim=-1)
        lm=(~mw).float()
        return (loss*lm).sum()/(lm.sum()+1e-6)

    def training_step(self,b,i):
        l=self._loss(b); self.log("train_loss",l,prog_bar=True,batch_size=1); self.tr.append(l); return l
    def validation_step(self,b,i):
        l=self._loss(b); self.log("val_loss",l,prog_bar=True,sync_dist=True,batch_size=1); self.va.append(l); return l
    def on_validation_epoch_end(self):
        self.log("val_epoch_loss", torch.stack(self.va).mean() if self.va else torch.tensor(0.0), prog_bar=True); self.va.clear()
    def on_train_epoch_end(self):
        self.log("train_epoch_loss", torch.stack(self.tr).mean() if self.tr else torch.tensor(0.0)); self.tr.clear()

def main():
    seed_everything(42)
    c=MambaConfig(); c.camera_names=[CAM]; c.embed_dim=2048; c.d_model=2048; c.action_dim=7
    c.lowdim_dim=8; c.img_size=(640,480); c.num_blocks=4; c.load_features=True; c.sum_camera_feats=False
    sc=Scaler(lowdim_dict={**{f'obs_{i}':1 for i in range(8)}, **{f'act_{i}':(16,1) for i in range(7)}})
    sc.load(SCALER_PATH); print('scaler loaded', SCALER_PATH); scg=copy.deepcopy(sc)
    tr=LiberoMemDataset(FEAT_ROOT, TASK, mode='train', scaler=sc, n_obs=8, n_act=7, cam=CAM)
    va=LiberoMemDataset(FEAT_ROOT, TASK, mode='test',  scaler=sc, n_obs=8, n_act=7, cam=CAM)
    trl=DataLoader(tr,batch_size=1,num_workers=4,collate_fn=parallel_collate_fn,shuffle=True,pin_memory=False,persistent_workers=True)
    val=DataLoader(va,batch_size=1,num_workers=2,collate_fn=parallel_collate_fn,shuffle=False,pin_memory=False,persistent_workers=True)
    model=LitLibero(c, scg)
    cb=ModelCheckpoint(monitor='val_epoch_loss', filename=f'libero-{TAG}-{{epoch:02d}}-{{val_epoch_loss:.4f}}', save_top_k=3, mode='min', save_last=True)
    tr_=pl.Trainer(accelerator='gpu', devices=[0], max_epochs=MAX_EPOCHS, callbacks=[cb, LearningRateMonitor('step')],
                   precision='16-mixed', gradient_clip_val=1.0, log_every_n_steps=1,
                   default_root_dir=f'/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs_libero/{TASK[:20]}_{TAG}')
    print(f"Training LIBERO {TASK[:30]} | mode={TAG} (WINDOW_K={WINDOW_K}) | epochs={MAX_EPOCHS}")
    tr_.fit(model, trl, val)

if __name__=='__main__': main()
