"""
test_micro_equiv.py — check that LitMambaWindowedMicro (micro-batched, manual optimization) performs the SAME
updates as the original LitMambaWindowed (single batch, automatic optimization, Trainer clip 1.0).
Both start from identical weights, see identical windows (dropout disabled so the global RNG is consumed only by
window sampling), run 3 optimizer steps under 16-mixed; we compare every trainable parameter afterwards.
Usage (MTIL/train): CUDA_VISIBLE_DEVICES=N python -u test_micro_equiv.py [k] [W] [micro]
"""
import sys, copy, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import seed_everything
from M_dataset_par import MambaTrajectoryDataset, parallel_collate_fn
from scaler_M import Scaler
from mamba_policy_par import MambaConfig
from train_par_windowed import LitMambaWindowed
from train_c3 import LitMambaWindowedMicro, ROOT, SCALER

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MICRO = int(sys.argv[3]) if len(sys.argv) > 3 else 4
STEPS = 3

config = MambaConfig(); config.camera_names = ['top']; config.embed_dim = 2048; config.d_model = 2048
config.action_dim = 14; config.img_size = (640, 480); config.num_blocks = 4; config.load_features = True
config.train_window_k = K; config.windows_per_step = W
OBS = ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos','gripper_pos2']
ACT = [o.replace('gripper_pos2', 'gripper_act2').replace('gripper_pos', 'gripper_act') if 'gripper' in o else o + '_act' for o in OBS]
fd = {k: 1 for k in OBS}; fd.update({k: (16, 1) for k in ACT})
sc = Scaler(lowdim_dict=fd); sc.load(SCALER)
ds = Subset(MambaTrajectoryDataset(root_dir=ROOT, mode="train", load_features=True, scaler=sc), list(range(STEPS)))


def run(micro):
    seed_everything(0)
    m = (LitMambaWindowedMicro(config, copy.deepcopy(sc), micro) if micro else LitMambaWindowed(config, copy.deepcopy(sc)))
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout): mod.p = 0.0
    init = {n: p.detach().float().cpu().clone() for n, p in m.policy.named_parameters() if p.requires_grad}
    grads = {}
    orig = m.on_before_optimizer_step
    def hook(optimizer):      # called after unscale_, before clipping, in both modes -> raw accumulated gradient
        if not grads:
            for n, p in m.policy.named_parameters():
                if p.requires_grad and p.grad is not None: grads[n] = p.grad.detach().float().cpu().clone()
        return orig(optimizer)
    m.on_before_optimizer_step = hook
    seed_everything(1)
    dl = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=parallel_collate_fn, shuffle=False)
    tr = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=1, precision='16-mixed', logger=False, enable_checkpointing=False,
                    gradient_clip_val=(None if micro else 1.0), limit_val_batches=0, enable_progress_bar=False, num_sanity_val_steps=0)
    tr.fit(m, dl)
    delta = {n: p.detach().float().cpu() - init[n] for n, p in m.policy.named_parameters() if p.requires_grad}
    return grads, delta


gA, dA = run(0)
gB, dB = run(MICRO)
ga = torch.cat([g.flatten() for n, g in sorted(gA.items())]); gb = torch.cat([gB[n].flatten() for n, _ in sorted(gA.items())])
da = torch.cat([d.flatten() for n, d in sorted(dA.items())]); db = torch.cat([dB[n].flatten() for n, _ in sorted(dA.items())])
g_rel = ((ga - gb).norm() / ga.norm()).item(); d_rel = ((da - db).norm() / da.norm()).item()
print(f"k={K} W={W} micro={MICRO}: step-1 gradient ||gA||={ga.norm():.4e} ||gB||={gb.norm():.4e} rel.diff={g_rel:.2e}; "
      f"param update after {STEPS} steps ||dA||={da.norm():.4e} rel.diff={d_rel:.2e}")
print("RESULT", "MATCH" if (g_rel < 1e-2 and d_rel < 5e-2) else "MISMATCH")
