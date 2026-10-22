"""
train_c3.py — resubmission experiments for C3 (history ablation), transfer_cube.
Seed-replication of the full-history and windowed-trained conditions + new context lengths (k=1,4),
WITHOUT editing the audited originals: imports LitMambaParallel (train_par.py) and
LitMambaWindowed (train_par_windowed.py) and only overrides seed / k / W / epochs / output dir.

Usage (run from MTIL/train, CUDA_VISIBLE_DEVICES picks the GPU):
  python -u train_c3.py --mode full --seed 1
  python -u train_c3.py --mode win  --k 16 --seed 1
  python -u train_c3.py --mode win  --k 1  --seed 42 --w 256
Defaults match the originals: full = 150 epochs (version_6 recipe), win = 45 epochs, W=16 (version_16 recipe).
Frame-instance budget per step for windowed runs = k*W (original k=16,W=16 -> 256); pass --w to keep it matched.
Output: train/c3_runs/<tag>/checkpoints (+ history.json, lightning_logs/)
--smoke: 1 epoch, 4 train + 2 val trajectories (pipeline check only).
"""
import os, sys, copy, time, json, argparse, gc
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from M_dataset_par import MambaTrajectoryDataset, parallel_collate_fn
from scaler_M import Scaler
from mamba_policy_par import MambaConfig
from train_par import LitMambaParallel
from train_par_windowed import LitMambaWindowed

ROOT = "/data/nas07_new/PersonalData/phuocthien/mtil_data/transfer_cube_feat"
SCALER = '/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer.pth'
OUT_BASE = "/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/c3_runs"


from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO


class CPULoadCheckpointIO(TorchCheckpointIO):
    """Resume-time fix: load last.ckpt onto the CPU. Lightning's default map_location keeps the saved (cuda) device,
    putting the whole 4.6 GB checkpoint (weights + AdamW state) on the GPU on top of the live model/optimizer ->
    OOM on a shared 24 GB card. Restored tensors are moved to the GPU by Lightning as usual; saving is unchanged."""
    def load_checkpoint(self, path, map_location=None, weights_only=None):
        return super().load_checkpoint(path, map_location='cpu', weights_only=False)


class EpochLog(Callback):
    """print-based progress (Rich bar does not flush to nohup logs) + per-epoch JSON history."""
    def __init__(self, path):
        self.path = path; self.hist = []; self.t0 = None
    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()
    def on_train_epoch_end(self, trainer, pl_module):
        # runs after this epoch's validation, so val_epoch_loss (logged by the module) is available
        m = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.is_tensor(v) and v.numel() == 1}
        rec = {'epoch': trainer.current_epoch, 'sec': round(time.time() - (self.t0 or time.time()), 1),
               'val_epoch_loss': m.get('val_epoch_loss'), 'train_loss': m.get('train_loss'),
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        self.hist.append(rec)
        print(f"[epoch {rec['epoch']}] {rec['sec']}s val_epoch_loss={rec['val_epoch_loss']} "
              f"train_loss={rec['train_loss']} peak={rec['peak_mem_gb']}GB", flush=True)
        with open(self.path, 'w') as f: json.dump(self.hist, f, indent=1)


class FixedValWindows:
    """--fixed_val: validation draws the SAME random windows every epoch (window starts come from the CPU generator,
    re-seeded per val batch; the val loader is unshuffled, so batch_idx -> trajectory is fixed) -> a stable
    checkpoint-selection signal for windowed runs. CPU and CUDA RNG states are forked and restored, so training
    randomness is untouched. (The two 2026-10-05 runs used devices=[], i.e. CUDA RNG was re-seeded by each val batch:
    deterministic, but training-time dropout after validation followed that re-seeded CUDA stream.)"""
    def validation_step(self, batch, batch_idx):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(100003 + batch_idx)
            return super().validation_step(batch, batch_idx)


class LitMambaWindowedMicro(LitMambaWindowed):
    """Same objective as LitMambaWindowed._step_loss, but the W windows are pushed through the model in
    micro-batches of `micro` windows with gradient accumulation into ONE optimizer step. Each micro-batch's
    masked loss SUM is divided by the TOTAL valid-frame count of all W windows, so the accumulated gradient
    equals that of the single W-window batch (same updates, same frame-instance budget).
    Needed for small k: mamba's chunked scan keeps a [W, nheads, headdim, d_state=512] state per sequence per
    block, so k=1 x W=256 needs ~16x the SSM-state memory of k=16 x W=16 and OOMs on 20 GB.
    Manual optimization under 16-mixed: opt.step(closure) runs the closure, unscales, then calls
    on_before_optimizer_step (where we clip at 1.0 on UNSCALED grads = Trainer(gradient_clip_val=1.0)), then
    scaler.step (skipped on inf/nan, as in automatic mode). CosineAnnealingLR is stepped once per epoch."""
    def __init__(self, config, scaler, micro):
        super().__init__(config, scaler)
        self.micro = micro
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        obs = batch['obs'].to(self.device); actions = batch['actions'].to(self.device); mask = batch['mask'].to(self.device)
        obs_w, act_w, mask_w, rgb_w = self._sample_windows(obs, actions, mask, batch['rgb_batch'])
        lm = (~mask_w).float(); denom = lm.sum() + 1e-6
        acc = {'loss': torch.zeros((), device=self.device)}

        def closure():
            for i in range(0, obs_w.shape[0], self.micro):
                sl = slice(i, i + self.micro)
                feats = self.compute_vision_features_chunked({cam: v[sl] for cam, v in rgb_w.items()})
                pred = self.policy.forward_sequence(obs_w[sl], feats)
                l = torch.nn.functional.mse_loss(pred, act_w[sl], reduction='none').mean(dim=-1).mean(dim=-1)
                part = (l * lm[sl]).sum() / denom
                self.manual_backward(part)
                acc['loss'] += part.detach()
                del feats, pred, l, part

        opt.zero_grad()
        opt.step(closure=closure)
        loss = acc['loss']
        self.log("train_loss", loss, prog_bar=True, batch_size=self.W)
        self.train_step_outputs.append(loss)
        return loss

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad and p.grad is not None], 1.0)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.lr_schedulers().step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['full', 'win'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--w', type=int, default=16)
    ap.add_argument('--micro', type=int, default=0, help='windows per micro-batch (0 = all W at once, the original)')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--fixed_val', action='store_true', help='windowed: fixed validation windows + 6-decimal ckpt names (tag suffix _fv)')
    ap.add_argument('--task', default='transfer_cube',
                    help="transfer_cube (paper) or rmbench_<task> (features mtil_data/rmbench_<task>_feat, scaler_rmbench_<task>.pth)")
    ap.add_argument('--enc', default='vitl14', choices=['vitl14', 'vitb14', 'vits14'],
                    help='frozen DINOv2 size; non-default reads transfer_cube_feat_<enc> and resizes the adapter input conv')
    a = ap.parse_args()
    epochs = a.epochs or (150 if a.mode == 'full' else 45)
    if a.smoke: epochs = 1
    tag = f"full_s{a.seed}" if a.mode == 'full' else f"win_k{a.k}_w{a.w}_s{a.seed}"
    if a.epochs and a.epochs != (150 if a.mode == 'full' else 45): tag = f"{tag}_e{a.epochs}"   # non-default budget
    if a.enc != 'vitl14': tag = f"{a.enc}_{tag}"
    if a.fixed_val: tag = f"{tag}_fv"
    root = ROOT if a.enc == 'vitl14' else f"{ROOT}_{a.enc}"
    scaler_path = SCALER
    shm = f"/dev/shm/c3feat_{a.enc}"          # fp16 RAM copy made by build_shm_cache.py (exact under 16-mixed)
    if a.task != 'transfer_cube':
        assert a.task.startswith('rmbench_') and a.enc == 'vitl14'
        tag = f"{a.task}_{tag}"
        root = f"/data/nas07_new/PersonalData/phuocthien/mtil_data/{a.task}_feat"
        scaler_path = f"/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_{a.task}.pth"
        shm = f"/dev/shm/c3feat_{a.task}"
    if os.path.exists(os.path.join(shm, 'COMPLETE')):
        root = shm; os.environ['PRELOAD'] = '0'   # already in RAM: read via mmap, no second in-process copy
    if a.smoke: tag = 'smoke_' + tag
    out = os.path.join(OUT_BASE, tag); os.makedirs(out, exist_ok=True)
    print(f"C3 run: task={a.task} mode={a.mode} seed={a.seed} k={a.k} W={a.w} micro={a.micro} epochs={epochs} out={out} scaler={scaler_path}", flush=True)

    seed_everything(a.seed)
    config = MambaConfig(); config.camera_names = ['top']; config.embed_dim = 2048; config.d_model = 2048
    config.action_dim = 14; config.img_size = (640, 480); config.num_blocks = 4; config.load_features = True
    OBS = ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos','gripper_pos2']
    ACT = ['agl_1_act','agl_2_act','agl_3_act','agl_4_act','agl_5_act','agl_6_act','agl2_1_act','agl2_2_act','agl2_3_act','agl2_4_act','agl2_5_act','agl2_6_act','gripper_act','gripper_act2']
    fd = {k: 1 for k in OBS}; fd.update({k: (16, 1) for k in ACT})
    scaler_cpu = Scaler(lowdim_dict=fd); scaler_cpu.load(scaler_path); scaler_gpu = copy.deepcopy(scaler_cpu)

    train_ds = MambaTrajectoryDataset(root_dir=root, mode="train", load_features=True, scaler=scaler_cpu)
    val_ds = MambaTrajectoryDataset(root_dir=root, mode="test", load_features=True, scaler=scaler_cpu)
    if a.smoke:
        train_ds = Subset(train_ds, list(range(4))); val_ds = Subset(val_ds, list(range(2)))
    print(f"train trajs={len(train_ds)} val trajs={len(val_ds)} root={root} PRELOAD={os.environ.get('PRELOAD')}", flush=True)
    # pin_memory=False: pinning 2.6GB/traj feature tensors is the CPU bottleneck (CLAUDE.md, insertion notes)
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=4, collate_fn=parallel_collate_fn, shuffle=True,
                              pin_memory=False, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=parallel_collate_fn, shuffle=False,
                            pin_memory=False, persistent_workers=True)

    if a.mode == 'full':
        model = LitMambaParallel(config, scaler=scaler_gpu); fname = 'mamba-par-{epoch:02d}-{val_epoch_loss:.4f}'
    else:
        config.train_window_k = a.k; config.windows_per_step = a.w
        model = (LitMambaWindowedMicro(config, scaler=scaler_gpu, micro=a.micro) if 0 < a.micro < a.w
                 else LitMambaWindowed(config, scaler=scaler_gpu)); fname = f'mamba-win{a.k}-{{epoch:02d}}-{{val_epoch_loss:.4f}}'
        if a.fixed_val:
            model.__class__ = type('FixedVal' + type(model).__name__, (FixedValWindows, type(model)), {})
            fname = f'mamba-win{a.k}-{{epoch:02d}}-{{val_epoch_loss:.6f}}'
    if a.enc != 'vitl14':   # same swap as test/mamba_policy_enc.py (only the adapter's input width changes)
        C = {'vits14': 384, 'vitb14': 768}[a.enc]
        for name in ('spatial_adapter', 'spatial_adpater_low'):
            seq = getattr(model.policy, name, None)
            if seq is not None and isinstance(seq[0], torch.nn.Conv2d):
                old = seq[0]; seq[0] = torch.nn.Conv2d(C, old.out_channels, old.kernel_size, padding=old.padding)
        print(f"encoder={a.enc}: adapter input conv -> {C} channels, features from {root}", flush=True)
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(out, 'checkpoints'), monitor='val_epoch_loss', filename=fname, save_top_k=5, mode='min', save_last=True)
    trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=epochs, default_root_dir=out,
                         callbacks=[ckpt_cb, LearningRateMonitor(logging_interval='step'), EpochLog(os.path.join(out, 'history.json'))],
                         precision='16-mixed', gradient_clip_val=(None if model.automatic_optimization is False else 1.0),
                         log_every_n_steps=1, enable_progress_bar=False, plugins=[CPULoadCheckpointIO()])
    # resume if a previous attempt of this exact run died (GPU fault / preemption)
    last = os.path.join(out, 'checkpoints', 'last.ckpt')
    ckpt_path = last if (os.path.exists(last) and not a.smoke) else None
    if ckpt_path: print(f"RESUMING from {ckpt_path}", flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    print(f"C3 RUN DONE {tag} best={ckpt_cb.best_model_path} ({ckpt_cb.best_model_score})", flush=True)
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
