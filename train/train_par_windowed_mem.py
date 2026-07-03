"""
train_par_windowed.py — WINDOWED-TRAINED BASELINE for the history ablation.
Trains a LIMITED-CONTEXT (k-frame) model: each step samples W random k-length windows from a
trajectory and runs forward_sequence on each window FRESH (no history before the window start) —
exactly matching how test/evaluate_fs_windowed.py evaluates. This removes the cold-start caveat of
the windowed-EVAL on the full-history model: if a model TRAINED for k-context still fails the task,
the task genuinely needs history; if it succeeds, full history is not essential.
Based on train_par.py (transfer_cube). Set TRAIN_WINDOW_K below.
"""
import os, copy, torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.checkpoint as _gradckpt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from M_dataset_par import MambaTrajectoryDataset, parallel_collate_fn
from scaler_M import Scaler
from metric_M import my_Metric
from mamba_policy_par import MambaPolicy, MambaConfig

TRAIN_WINDOW_K = 16      # context length the model is trained on
WINDOWS_PER_STEP = 16    # W random windows sampled per trajectory per step (batched)

class LitMambaWindowed(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.config = config
        self.scaler = scaler
        self.metric = my_Metric()
        self.policy = MambaPolicy(
            camera_names=config.camera_names, embed_dim=config.embed_dim, lowdim_dim=config.lowdim_dim,
            d_model=config.d_model, action_dim=config.action_dim, sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks, future_steps=16, img_size=config.img_size, mamba_cfg=config)
        self.policy.shared_backbone.eval()
        for p in self.policy.shared_backbone.parameters(): p.requires_grad = False
        if getattr(config, 'load_features', False): self.policy.shared_backbone = None
        self.lr = 1e-4; self.weight_decay = 1e-4
        self.K = config.train_window_k; self.W = config.windows_per_step
        self.train_step_outputs = []; self.val_step_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch}

    def denormalize(self, actions):
        arm1={'agl_1_act':actions[...,0:1],'agl_2_act':actions[...,1:2],'agl_3_act':actions[...,2:3],'agl_4_act':actions[...,3:4],'agl_5_act':actions[...,4:5],'agl_6_act':actions[...,5:6],'gripper_act':actions[...,6:7]}
        arm2={'agl2_1_act':actions[...,7:8],'agl2_2_act':actions[...,8:9],'agl2_3_act':actions[...,9:10],'agl2_4_act':actions[...,10:11],'agl2_5_act':actions[...,11:12],'agl2_6_act':actions[...,12:13],'gripper_act2':actions[...,13:14]}
        d1=self.scaler.denormalize(arm1); d2=self.scaler.denormalize(arm2)
        return torch.cat([d1['agl_1_act'],d1['agl_2_act'],d1['agl_3_act'],d1['agl_4_act'],d1['agl_5_act'],d1['agl_6_act'],d1['gripper_act'],
                          d2['agl2_1_act'],d2['agl2_2_act'],d2['agl2_3_act'],d2['agl2_4_act'],d2['agl2_5_act'],d2['agl2_6_act'],d2['gripper_act2']],dim=2)

    def compute_vision_features_chunked(self, rgb_batch):
        device=self.device; cam_names=list(rgb_batch.keys())
        if not cam_names: return {}
        B=len(rgb_batch[cam_names[0]]); lengths=[t.shape[0] for t in rgb_batch[cam_names[0]]]; max_len=max(lengths)
        features_seq={cam: torch.zeros(B,max_len,self.config.embed_dim,device=device,dtype=torch.float32) for cam in self.config.camera_names}
        chunk_size=16
        for b in range(B):
            for cam in self.config.camera_names:
                raw=rgb_batch[cam][b]; L=raw.shape[0]
                for i in range(0,L,chunk_size):
                    img=raw[i:i+chunk_size].to(device,non_blocking=True)
                    dino=img if getattr(self.config,'load_features',False) else self.policy.shared_backbone(img)
                    if self.training and getattr(self.config,'use_grad_ckpt',False):
                        emb=_gradckpt.checkpoint(self.policy.process_vision_chunk, dino, use_reentrant=False)
                    else:
                        emb=self.policy.process_vision_chunk(dino)
                    features_seq[cam][b,i:i+chunk_size]=emb
                    del img,dino,emb
        return features_seq

    def _sample_windows(self, obs, actions, mask, rgb_batch):
        """From a single trajectory (B=1) sample W random k-windows -> batch [W,k,...]."""
        K=self.K; W=self.W
        Lv=int((~mask[0]).sum().item())
        if Lv<=K:
            starts=[0]
        else:
            starts=torch.randint(0, Lv-K+1, (W,)).tolist()
        obs_w=torch.stack([obs[0, s:s+K] for s in starts],0)           # [W,K,14]
        act_w=torch.stack([actions[0, s:s+K] for s in starts],0)        # [W,K,16,14]
        mask_w=torch.stack([mask[0, s:s+K] for s in starts],0)          # [W,K]
        rgb_w={cam:[rgb_batch[cam][0][s:s+K] for s in starts] for cam in rgb_batch}
        return obs_w, act_w, mask_w, rgb_w

    def _step_loss(self, batch):
        obs=batch['obs'].to(self.device); actions=batch['actions'].to(self.device); mask=batch['mask'].to(self.device); rgb=batch['rgb_batch']
        obs_w, act_w, mask_w, rgb_w = self._sample_windows(obs, actions, mask, rgb)
        cam_feats=self.compute_vision_features_chunked(rgb_w)
        pred=self.policy.forward_sequence(obs_w, cam_feats)              # [W,K,16,14]
        loss=F.mse_loss(pred, act_w, reduction='none').mean(dim=-1).mean(dim=-1)  # [W,K]
        lm=(~mask_w).float()
        return (loss*lm).sum()/(lm.sum()+1e-6), pred, act_w, mask_w

    def training_step(self, batch, batch_idx):
        loss,_,_,_=self._step_loss(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.W)
        self.train_step_outputs.append(loss); return loss

    def validation_step(self, batch, batch_idx):
        val_loss,pred,act_w,mask_w=self._step_loss(batch)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=self.W)
        self.val_step_outputs.append(val_loss)
        vi=~mask_w
        self.metric.update(self.denormalize(pred[vi]), self.denormalize(act_w[vi]))
        return val_loss

    def on_validation_epoch_end(self):
        avg=torch.stack(self.val_step_outputs).mean() if self.val_step_outputs else 0
        self.log("val_epoch_loss", avg, prog_bar=True); self.val_step_outputs.clear()
        for k,v in self.metric.compute().items(): self.log(f"val_{k}", v, sync_dist=True)
        self.metric.reset()

    def on_train_epoch_end(self):
        avg=torch.stack(self.train_step_outputs).mean() if self.train_step_outputs else 0
        self.log("train_epoch_loss", avg); self.train_step_outputs.clear()

def main():
    seed_everything(42)
    config=MambaConfig(); config.camera_names=['top']; config.embed_dim=2048; config.d_model=2048
    config.action_dim=14; config.img_size=(640,480); config.num_blocks=4; config.load_features=True; config.use_grad_ckpt=True
    config.train_window_k=TRAIN_WINDOW_K; config.windows_per_step=WINDOWS_PER_STEP
    scaler_path='/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer_mem.pth'
    OBS=['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos','gripper_pos2']
    ACT=['agl_1_act','agl_2_act','agl_3_act','agl_4_act','agl_5_act','agl_6_act','agl2_1_act','agl2_2_act','agl2_3_act','agl2_4_act','agl2_5_act','agl2_6_act','gripper_act','gripper_act2']
    fd={k:1 for k in OBS}; fd.update({k:(16,1) for k in ACT})
    scaler_cpu=Scaler(lowdim_dict=fd); scaler_cpu.load(scaler_path); print('scaler loaded'); scaler_gpu=copy.deepcopy(scaler_cpu)
    root="/tmp/transfer_cube_mem_feat"
    train_ds=MambaTrajectoryDataset(root_dir=root, mode="train", load_features=True, scaler=scaler_cpu)
    val_ds=MambaTrajectoryDataset(root_dir=root, mode="test", load_features=True, scaler=scaler_cpu)
    train_loader=DataLoader(train_ds, batch_size=1, num_workers=4, collate_fn=parallel_collate_fn, shuffle=True, pin_memory=False, persistent_workers=True)
    val_loader=DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=parallel_collate_fn, shuffle=False, pin_memory=False, persistent_workers=True)
    model=LitMambaWindowed(config, scaler=scaler_gpu)
    ckpt_cb=ModelCheckpoint(monitor='val_epoch_loss', filename=f'mamba-win{TRAIN_WINDOW_K}-{{epoch:02d}}-{{val_epoch_loss:.4f}}', save_top_k=5, mode='min', save_last=True)
    trainer=pl.Trainer(accelerator='gpu', devices=[0], max_epochs=100, default_root_dir='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs_mem_win16', callbacks=[ckpt_cb, LearningRateMonitor(logging_interval='step')], precision='16-mixed', gradient_clip_val=1.0, log_every_n_steps=1)
    print(f"Starting WINDOWED training (k={TRAIN_WINDOW_K}, W={WINDOWS_PER_STEP})...")
    trainer.fit(model, train_loader, val_loader)

if __name__=="__main__": main()
