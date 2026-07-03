"""train_libero_mem.py — LIBERO-Mem (C4) training, FULL-replication of the mtil_plus gold recipe
(docs/LIBERO_SR_FIX.md: dual-cam + square 128^2 + EMA + smooth_l1 + grad-accum 8 + bf16) PLUS the
windowed-context ablation needed for the memory experiment:
  WINDOW_K=0  -> full-history (forward_sequence over the whole episode)
  WINDOW_K=k  -> windowed (W random k-frame windows/step, fresh state per window)   [k=1 = single-frame]

Model: dual-cam (agentview+eye_in_hand) [1024,10,10] -> spatial_adapter_low, lowdim_dim=9 (robot_states),
action_dim=7, lang_dim=None (single-task -> no FiLM). precision=bf16-mixed (Ampere).

Usage: CUDA_VISIBLE_DEVICES=<free> python train_libero_mem.py <task> <WINDOW_K> [max_epochs] [accum] [ema_decay]
"""
import os, sys, copy, torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from M_dataset_libero_mem import LiberoMemDataset, libero_mem_collate_fn, FEAT_ROOT
from scaler_M import Scaler
from mamba_policy_libero import MambaPolicy, MambaConfig

NAS = "/data/nas07_new/PersonalData/phuocthien"
TASK = sys.argv[1] if len(sys.argv) > 1 else 'KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate'
WINDOW_K = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
ACCUM = int(sys.argv[4]) if len(sys.argv) > 4 else 8
EMA_DECAY = float(sys.argv[5]) if len(sys.argv) > 5 else 0.999
NW = int(os.environ.get('NW', '4'))   # DataLoader workers; NW=0 (no fork) avoids the mmap-in-worker segfault on long full-history trajs
WINDOWS_PER_STEP = 16
FUTURE_STEPS = 8
SCALER_PATH = f'{NAS}/mtil_data/scaler_libero_mem_{TASK[:20]}.pth'
TAG = 'full' if WINDOW_K == 0 else f'win{WINDOW_K}'


class EMACallback(Callback):
    """EMA of policy weights, EVALUATED on the EMA copy (val_loss + saved ckpt reflect EMA). mtil_plus Stage-1."""
    def __init__(self, decay=0.999):
        self.decay = decay; self.shadow = None; self.backup = None
    def _params(self, m):
        return {n: p for n, p in m.policy.named_parameters() if p.requires_grad}
    def on_fit_start(self, tr, m):
        self.shadow = {n: p.detach().clone().float() for n, p in self._params(m).items()}
    def on_train_batch_end(self, tr, m, *a, **k):
        if self.shadow is None: return
        d = self.decay
        for n, p in self._params(m).items():
            self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
    def _swap_in(self, m):
        if self.shadow is None: return
        self.backup = {}
        for n, p in self._params(m).items():
            self.backup[n] = p.detach().clone(); p.data.copy_(self.shadow[n].to(p.dtype))
    def _swap_out(self, m):
        if self.backup is None: return
        for n, p in self._params(m).items(): p.data.copy_(self.backup[n])
        self.backup = None
    def on_validation_start(self, tr, m):
        if not tr.sanity_checking: self._swap_in(m)
    def on_train_epoch_start(self, tr, m): self._swap_out(m)
    def on_fit_end(self, tr, m): self._swap_in(m)


class LitLiberoMem(pl.LightningModule):
    def __init__(self, config, scaler):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.config = config; self.scaler = scaler; self.K = WINDOW_K; self.W = WINDOWS_PER_STEP
        self.policy = MambaPolicy(
            camera_names=config.camera_names, embed_dim=config.embed_dim, lowdim_dim=config.lowdim_dim,
            d_model=config.d_model, action_dim=config.action_dim, sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks, future_steps=FUTURE_STEPS, img_size=config.img_size,
            mamba_cfg=config, lang_dim=None)   # single-task LIBERO-Mem -> no language/FiLM
        self.policy.shared_backbone = None     # cached features -> backbone not needed
        self.lr = 1e-4; self.wd = 1e-4; self.tr = []; self.va = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sch}

    def _vision(self, rgb_batch):
        cams = self.config.camera_names
        B = len(rgb_batch[cams[0]]); maxL = max(t.shape[0] for t in rgb_batch[cams[0]]); dev = self.device
        feats = {c: torch.zeros(B, maxL, self.config.embed_dim, device=dev) for c in cams}
        cs = 16
        for bi in range(B):
            for cam in cams:
                imgs = rgb_batch[cam][bi]; L = imgs.shape[0]
                for i in range(0, L, cs):
                    chunk = imgs[i:i + cs].to(dev, non_blocking=True)
                    feats[cam][bi, i:i + chunk.shape[0]] = self.policy.process_vision_chunk(chunk)
        return feats

    def _sample_windows(self, obs, actions, mask, rgb):
        K, W = self.K, self.W; Lv = int((~mask[0]).sum().item())
        starts = [0] if Lv <= K else torch.randint(0, Lv - K + 1, (W,)).tolist()
        ow = torch.stack([obs[0, s:s + K] for s in starts], 0)
        aw = torch.stack([actions[0, s:s + K] for s in starts], 0)
        mw = torch.stack([mask[0, s:s + K] for s in starts], 0)
        rw = {cam: [rgb[cam][0][s:s + K] for s in starts] for cam in rgb}
        return ow, aw, mw, rw

    def _loss(self, batch):
        obs = batch['obs'].to(self.device); actions = batch['actions'].to(self.device)
        mask = batch['mask'].to(self.device); rgb = batch['rgb_batch']
        if self.K == 0:
            feats = self._vision(rgb); pred = self.policy.forward_sequence(obs, feats); aw, mw = actions, mask
        else:
            ow, aw, mw, rw = self._sample_windows(obs, actions, mask, rgb)
            feats = self._vision(rw); pred = self.policy.forward_sequence(ow, feats)
        # smooth_l1 (Huber): robust to demo-noise/action outliers (mtil_plus Stage-1)
        loss = F.smooth_l1_loss(pred, aw, reduction='none').mean(dim=(-1, -2))   # [B,L]
        m = (~mw).float()
        return (loss * m).sum() / (m.sum() + 1e-6)

    def training_step(self, b, i):
        l = self._loss(b); self.log("train_loss", l, prog_bar=True, batch_size=1); self.tr.append(l.detach()); return l
    def validation_step(self, b, i):
        l = self._loss(b); self.log("val_loss", l, prog_bar=True, sync_dist=True, batch_size=1); self.va.append(l.detach()); return l
    def on_validation_epoch_end(self):
        self.log("val_epoch_loss", torch.stack(self.va).mean() if self.va else torch.tensor(0.0), prog_bar=True); self.va.clear()
    def on_train_epoch_end(self):
        self.log("train_epoch_loss", torch.stack(self.tr).mean() if self.tr else torch.tensor(0.0)); self.tr.clear()


def main():
    seed_everything(42)
    c = MambaConfig(); c.camera_names = ['agentview', 'eye_in_hand']; c.embed_dim = 2048; c.d_model = 2048
    c.action_dim = 7; c.lowdim_dim = 9; c.img_size = (128, 128); c.num_blocks = 4
    c.load_features = True; c.sum_camera_feats = False
    c.use_grad_ckpt = os.environ.get('GRADCKPT', '0') == '1'   # gradient checkpointing: ~15GB->~6GB for long full-history trajs (exact, slower), survives GPU contention
    sc = Scaler(lowdim_dict={'qpos': 9, 'act': 7}); sc.load(SCALER_PATH); print('scaler loaded', SCALER_PATH)
    scg = copy.deepcopy(sc)
    tr = LiberoMemDataset(FEAT_ROOT, TASK, mode='train', scaler=sc, future_steps=FUTURE_STEPS)
    va = LiberoMemDataset(FEAT_ROOT, TASK, mode='test', scaler=sc, future_steps=FUTURE_STEPS)
    trl = DataLoader(tr, batch_size=1, num_workers=NW, collate_fn=libero_mem_collate_fn, shuffle=True, pin_memory=False, persistent_workers=(NW > 0))
    val = DataLoader(va, batch_size=1, num_workers=min(2, NW), collate_fn=libero_mem_collate_fn, shuffle=False, pin_memory=False, persistent_workers=(NW > 0))
    model = LitLiberoMem(c, scg)
    cb = ModelCheckpoint(monitor='val_epoch_loss', filename=f'libero-{TAG}-{{epoch:02d}}-{{val_epoch_loss:.4f}}', save_top_k=3, mode='min', save_last=True)
    callbacks = [cb, LearningRateMonitor('epoch')]
    if EMA_DECAY > 0:
        callbacks.append(EMACallback(decay=EMA_DECAY))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=MAX_EPOCHS, callbacks=callbacks,
                         precision='bf16-mixed', gradient_clip_val=1.0, accumulate_grad_batches=ACCUM,
                         log_every_n_steps=5,
                         default_root_dir=f'{NAS}/ssm_vla/MTIL/train/lightning_logs_libero_mem/{TASK[:20]}_{TAG}')
    print(f"Training LIBERO-Mem {TASK[:30]} | mode={TAG} (WINDOW_K={WINDOW_K}) | epochs={MAX_EPOCHS} accum={ACCUM} ema={EMA_DECAY}")
    trainer.fit(model, trl, val)
    print("[train] done. best:", cb.best_model_path)


if __name__ == '__main__':
    main()
