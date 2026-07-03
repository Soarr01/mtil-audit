import os
import torch
import copy  # <--- 关键库
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.utils.checkpoint as _gradckpt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# 假设你的文件结构如下，请确保 import 路径正确
from M_dataset_par import MambaTrajectoryDataset, parallel_collate_fn
from scaler_M import Scaler
from metric_M import my_Metric # 你的 metric 代码
from mamba_policy_par import MambaPolicy, MambaConfig

class LitMambaParallel(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.config = config
        self.scaler = scaler
        self.metric = my_Metric() # 初始化你的 Metric
        
        # 初始化 Policy
        self.policy = MambaPolicy(
            camera_names = config.camera_names,
            embed_dim = config.embed_dim,
            lowdim_dim = config.lowdim_dim,
            d_model = config.d_model,
            action_dim = config.action_dim,
            sum_camera_feats = config.sum_camera_feats,
            num_blocks = config.num_blocks,
            future_steps = 16,
            img_size = config.img_size,
            mamba_cfg = config
        )
        
        # 冻结 Backbone
        self.policy.shared_backbone.eval()
        for p in self.policy.shared_backbone.parameters():
            p.requires_grad = False
        # cached features => DINOv2 khong duoc goi trong training; giai phong ~1.2GB
        if getattr(config, 'load_features', False):
            self.policy.shared_backbone = None
            
        self.lr = 1e-4
        self.weight_decay = 1e-4

        # --- DELAYED-CUE auxiliary cue-retention head (the fix for weak cue usage) ---
        # The action-MSE alone gives the 1-bit cue ~15% of the loss budget, so the model
        # defaults to a positional prior. This tiny head predicts the cued bin from the
        # per-frame Mamba hidden state and is supervised at every frame AFTER the cue vanishes,
        # forcing the recurrent state to CARRY the cue across the ~350-frame gap.
        # TRAIN-ONLY: eval/cue-swap call forward_sequence without this head, so the
        # full-vs-windowed comparison stays vision-only and fair.
        self.cue_head = nn.Linear(config.d_model, 1)
        self.cue_tau = getattr(config, 'cue_tau', 12)          # cue visible frames [0,tau) -> supervise t>=tau
        self.aux_lambda = getattr(config, 'aux_lambda', 1.0)   # weight of the BCE cue-retention loss
        self.place_w = getattr(config, 'place_weight', 5.0)    # upweight the cue-discriminative place phase
        self.place_t0 = getattr(config, 'place_t0', 350)       # place phase begins ~frame 360

        # 容器用于记录 epoch loss
        self.train_step_outputs = []
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),   # self.* -> includes the aux cue_head
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def denormalize(self, actions):
            # [B,20]: arm1 => [0:10], arm2 => [10:20]
            # arm1 => pose_act(9)+gripper_act(1)
            # arm2 => pose_act2(9)+gripper_act2(1)
            arm1_dict = {
                'agl_1_act': actions[..., 0:1],'agl_2_act': actions[..., 1:2],'agl_3_act': actions[..., 2:3],
                'agl_4_act': actions[..., 3:4],'agl_5_act': actions[..., 4:5],'agl_6_act': actions[..., 5:6],
                'gripper_act': actions[..., 6:7]
            }
            arm2_dict = {
                'agl2_1_act': actions[..., 7:8],'agl2_2_act': actions[..., 8:9],'agl2_3_act': actions[..., 9:10],
                'agl2_4_act': actions[..., 10:11],'agl2_5_act': actions[..., 11:12],'agl2_6_act': actions[..., 12:13],
                'gripper_act2': actions[..., 13:14]
            }
            arm1_denorm = self.scaler.denormalize(arm1_dict)
            arm2_denorm = self.scaler.denormalize(arm2_dict)
            out = torch.cat([
                arm1_denorm['agl_1_act'],arm1_denorm['agl_2_act'],arm1_denorm['agl_3_act'],
                arm1_denorm['agl_4_act'],arm1_denorm['agl_5_act'],arm1_denorm['agl_6_act'],
                arm1_denorm['gripper_act'],
                arm2_denorm['agl2_1_act'],arm2_denorm['agl2_2_act'],arm2_denorm['agl2_3_act'],
                arm2_denorm['agl2_4_act'],arm2_denorm['agl2_5_act'],arm2_denorm['agl2_6_act'],
                arm2_denorm['gripper_act2']
            ], dim=2)
            return out
    
    def compute_vision_features_chunked(self, rgb_batch):
        """
        显存安全：分块计算视觉特征 (Chunked Vision Encoding)
        Args:
            rgb_batch: Dict {cam: List[Tensor(L_traj, 3, H, W)]} (在 CPU)
        Returns:
            features_seq: Dict {cam: [B, MaxL, embed_dim]} (在 GPU)
        """
        device = self.device
        # 假设 batch 中至少有一条数据
        cam_names = list(rgb_batch.keys())
        if len(cam_names) == 0: return {}
        
        B = len(rgb_batch[cam_names[0]]) 
        lengths = [t.shape[0] for t in rgb_batch[cam_names[0]]]
        max_len = max(lengths)
        
        # 预分配结果 Tensor (GPU)
        features_seq = {}
        for cam in self.config.camera_names:
            features_seq[cam] = torch.zeros(
                B, max_len, self.config.embed_dim, 
                device=device, dtype=torch.float32
            )

        chunk_size = 16 # 可根据显存调整 (16, 32, 64, 128 and so on...)
        
        # 逐轨迹处理
        for b_idx in range(B):
            for cam in self.config.camera_names:
                # 取出单条轨迹图片 (CPU)
                raw_imgs = rgb_batch[cam][b_idx] # [L, 3, H, W]
                L = raw_imgs.shape[0]
                
                # 分块循环 (For Loop)
                for i in range(0, L, chunk_size):
                    # 1. 搬运一小块到 GPU
                    img_chunk = raw_imgs[i : i+chunk_size].to(device, non_blocking=True)
                    
                    # 2. Backbone (No Grad, Freeze)
                    if getattr(self.config, 'load_features', False):
                        dino_out = img_chunk  # cached DINOv2 features [chunk,1024,Hf,Wf]
                    else:
                        with torch.no_grad():
                            dino_out = self.policy.shared_backbone(img_chunk)
                    
                    # 3. Adapter (With Grad if training) — grad-ckpt for T=500 (else OOM 20GB)
                    if self.training and getattr(self.config, 'use_grad_ckpt', False):
                        embed_chunk = _gradckpt.checkpoint(self.policy.process_vision_chunk, dino_out, use_reentrant=False)
                    else:
                        embed_chunk = self.policy.process_vision_chunk(dino_out)
                    # embed_chunk: [chunk, embed_dim]
                    
                    # 4. 填入结果并释放显存
                    features_seq[cam][b_idx, i : i+chunk_size] = embed_chunk
                    
                    # 显式删除引用，确保显存释放
                    del img_chunk, dino_out, embed_chunk
        
        return features_seq

    def training_step(self, batch, batch_idx):
        # 数据解包 (Tensor 在 CPU 还是 GPU 取决于 Collate 和 Prefetch，通常这里还是 CPU 或 GPU)
        # 为了安全，手动 to(device)
        obs = batch['obs'].to(self.device)           # [B, MaxL, 14]
        actions = batch['actions'].to(self.device)   # [B, MaxL, 16, 14]
        mask = batch['mask'].to(self.device)         # [B, MaxL] (True=Pad)
        rgb_batch = batch['rgb_batch']               # List of CPU Tensors
        
        # 1. Chunked Vision Encoding (Backbone + Adapter)
        cam_features_seq = self.compute_vision_features_chunked(rgb_batch)

        # 2. Mamba Sequence Forward (also return the per-frame hidden state for the aux cue-head)
        pred_actions, hidden = self.policy.forward_sequence(obs, cam_features_seq, return_hidden=True)

        B, L = mask.shape
        loss_mask = (~mask).float()                          # [B,L] 1=valid
        fidx = torch.arange(L, device=self.device)

        # 3a. Action MSE — upweight the cue-discriminative PLACE phase (frames >= place_t0)
        pf = F.mse_loss(pred_actions, actions, reduction='none').mean(dim=-1).mean(dim=-1)  # [B,L]
        w = torch.ones(L, device=self.device)
        w[fidx >= self.place_t0] = self.place_w
        wm = loss_mask * w[None, :]
        mse_loss = (pf * wm).sum() / (wm.sum() + 1e-6)

        # 3b. Auxiliary cue-retention BCE — predict target_bin from the hidden state at every
        #     post-cue frame (t >= tau), forcing the recurrent state to carry the 1-bit cue.
        target = batch['target_bin'].to(self.device).view(B, 1).expand(B, L)   # [B,L]
        cue_logits = self.cue_head(hidden).squeeze(-1)                         # [B,L]
        cue_mask = loss_mask * (fidx[None, :] >= self.cue_tau).float()
        bce = F.binary_cross_entropy_with_logits(cue_logits, target, reduction='none')
        aux_loss = (bce * cue_mask).sum() / (cue_mask.sum() + 1e-6)

        loss = mse_loss + self.aux_lambda * aux_loss

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_mse", mse_loss, prog_bar=False, batch_size=B)
        self.log("train_aux", aux_loss, prog_bar=True, batch_size=B)
        self.train_step_outputs.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        mask = batch['mask'].to(self.device)
        rgb_batch = batch['rgb_batch']
        
        # 1. Vision Encoding
        # Validation 时 Lightning 会自动包裹 torch.no_grad()
        cam_features_seq = self.compute_vision_features_chunked(rgb_batch)
        
        # 2. Forward (return hidden for the cue-accuracy probe)
        pred_actions, hidden = self.policy.forward_sequence(obs, cam_features_seq, return_hidden=True)

        # 3. Loss (PLAIN masked MSE — keep val_epoch_loss comparable to the baseline + ckpt monitor)
        loss = F.mse_loss(pred_actions, actions, reduction='none')
        loss = loss.mean(dim=-1).mean(dim=-1)
        loss_mask = (~mask).float()
        val_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=obs.shape[0])
        self.val_step_outputs.append(val_loss)

        # 3b. CUE-RETENTION accuracy at the decision phase (frames >= place_t0): does the hidden
        #     state still encode the cued bin long after the cue vanished? (the positive thesis signal)
        B, L = mask.shape
        fidx = torch.arange(L, device=self.device)
        dec_mask = ((~mask) & (fidx[None, :] >= self.place_t0)).float()
        if dec_mask.sum() > 0:
            target = batch['target_bin'].to(self.device).view(B, 1).expand(B, L)
            pred_bin = (torch.sigmoid(self.cue_head(hidden).squeeze(-1)) > 0.5).float()
            cue_acc = ((pred_bin == target).float() * dec_mask).sum() / (dec_mask.sum() + 1e-6)
            self.log("val_cue_acc", cue_acc, prog_bar=True, sync_dist=True, batch_size=B)
        
        valid_indices = ~mask # [B, L]
        
        # 展平并过滤 Padding
        pred_valid = pred_actions[valid_indices] # [Total_Valid_L, 16, 14]
        gt_valid = actions[valid_indices]        # [Total_Valid_L, 16, 14]
        
        # 反归一化
        pred_denorm = self.denormalize(pred_valid)
        gt_denorm = self.denormalize(gt_valid)
        
        # 更新指标
        self.metric.update(pred_denorm, gt_denorm)
        
        return val_loss

    def on_validation_epoch_end(self):
        # 1. 计算平均 Loss
        avg_loss = torch.stack(self.val_step_outputs).mean() if self.val_step_outputs else 0
        self.log("val_epoch_loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()
        
        # 2. 计算并打印 Metric
        metric_results = self.metric.compute() # 假设返回字典
        print("\nValidation Metrics:")
        for k, v in metric_results.items():
            print(f"{k}: {v}")
            # 记录到 tensorboard / wandb
            self.log(f"val_{k}", v, sync_dist=True)
            
        # 3. 重置 Metric
        self.metric.reset()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean() if self.train_step_outputs else 0
        self.log("train_epoch_loss", avg_loss)
        self.train_step_outputs.clear()

# ================= Main Execution =================
def main():
    seed_everything(42)

    # Claim GPU memory UP FRONT so other users can't steal the card during the ~13min PRELOAD.
    # (A prior run OOM'd: 2 users piled onto the GPU mid-preload, leaving 23 MiB free at fit time.)
    # PyTorch's caching allocator retains the freed block IN-PROCESS, so trainer.fit() reuses it;
    # we hold the reference through dataset/preload setup and release it right before fit WITHOUT
    # empty_cache -> the block never returns to the CUDA global pool -> no steal window.
    # Reservation DISABLED: the big cached block fragmented vs the ~12GB training peak and
    # OOM'd (held 13.86GB then a 3.07GB alloc couldn't reuse it). Training is ~12GB; just
    # launch onto a GPU with >=14GB free (the run claims it within ~60s of fit()).
    _gpu_reserve = None

    # 1. Config
    config = MambaConfig()
    config.camera_names = ['top']
    config.embed_dim = 2048
    config.d_model = 2048
    config.action_dim = 14
    config.img_size = (640, 480) 
    config.num_blocks = 4
    config.load_features = True
    config.use_grad_ckpt = True   # T=500 mem features OOM 20GB without gradient checkpointing
    # --- delayed-cue auxiliary-loss fix (makes the full model reliably READ the cue) ---
    config.cue_tau = 12           # cue visible frames [0,12) -> supervise cue retention at t>=12
    config.aux_lambda = 1.0       # weight of the BCE cue-retention loss
    config.place_weight = 3.0     # upweight the cue-discriminative place phase (5->3 for stability)
    config.place_t0 = 350         # place phase begins ~frame 360
    
    # 2. Scaler (Load) -- MEM variant: delayed-cue mem demos include a place-down motion,
    # so the action distribution differs from transfer_cube => use the mem-fit scaler.
    scaler_path = '/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer_mem.pth'
    OBS_KEYS = [
        'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
        'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6',
        'gripper_pos', 'gripper_pos2'
    ]
    ACT_KEYS = [
        'agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
        'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act',
        'gripper_act', 'gripper_act2'
    ]

    # 构造用于初始化 Scaler 的字典
    full_lowdim_dict = {}
    for k in OBS_KEYS:
        full_lowdim_dict[k] = 1          # shape (1,)
    for k in ACT_KEYS:
        full_lowdim_dict[k] = (16, 1)    # shape (16, 1)
    
    scaler_cpu = Scaler(lowdim_dict=full_lowdim_dict)
    if os.path.exists(scaler_path):
        scaler_cpu.load(scaler_path)
        print(f"Loading scaler from {scaler_path}...")
    else:
        print("Error: scaler_params.pth not found. Run Dataset script first.")
        return
    scaler_gpu = copy.deepcopy(scaler_cpu)
    # 3. Dataset & Loader (Parallel!)
    train_dataset = MambaTrajectoryDataset(
        root_dir="/tmp/transfer_cube_mem_feat",
        mode="train", load_features=True,
        scaler=scaler_cpu
    )
    val_dataset = MambaTrajectoryDataset(
        root_dir="/tmp/transfer_cube_mem_feat",
        mode="test", load_features=True,
        scaler=scaler_cpu
    )
    
    # Batch Size = 2 表示 2 条完整轨迹 (可能有 1000 帧)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, #现在并行化，可以根据显存调大不为1
        num_workers=4, 
        collate_fn=parallel_collate_fn,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True # 持久化 Workers，进一步加速
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        num_workers=2, 
        collate_fn=parallel_collate_fn,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True 
    )

    # 4. Model
    model = LitMambaParallel(config, scaler=scaler_gpu)
    
    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        filename='mamba-par-{epoch:02d}-{val_epoch_loss:.4f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 6. Trainer -- MEM FULL-history variant writes to its own lightning_logs dir
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=150,
        default_root_dir='/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL/train/lightning_logs_mem_full_aux',
        callbacks=[checkpoint_callback, lr_monitor],
        precision='bf16-mixed',   # fp16 diverged to NaN ~epoch 19 (aux+place-weight overflow);
                                  # bf16 has fp32 range on Ampere -> no overflow, stable.
        gradient_clip_val=1.0,
        log_every_n_steps=1
    )

    if _gpu_reserve is not None:
        del _gpu_reserve   # back into PyTorch's cache (NOT empty_cache) -> fit reuses it, no steal window
    print("Starting Parallel Training (MEM full-history, aux cue-loss)...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()