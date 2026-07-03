"""
build_mem_scaler.py — fit the normalization Scaler on the delayed-cue MEM transfer_cube demos.
Uses the SAME machinery as transfer_cube (M_dataset_par.MambaTrajectoryDataset.fit_scaler),
just pointed at the mem feature root. The mem demos include a place-down motion, so the action
distribution differs from transfer_cube => regenerate, do NOT reuse scaler_transfer.pth.
Saves to scaler_transfer_mem.pth.
"""
import os
import torch
from M_dataset_par import MambaTrajectoryDataset

ROOT = "/tmp/transfer_cube_mem_feat"
OUT = "/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_transfer_mem.pth"

def main():
    ds = MambaTrajectoryDataset(root_dir=ROOT, mode="train", load_features=True)
    print(f"Fitting scaler on {len(ds.records)} train trajectories from {ROOT}...")
    scaler = ds.fit_scaler(num_workers=4)
    ds.save_scaler(OUT)
    print(f"Scaler saved to {OUT}")
    # quick stats sanity
    sd = torch.load(OUT, map_location='cpu', weights_only=False)
    print(f"Scaler state_dict keys: {len(sd)}")
    print("=== sample stats (mem scaler) ===")
    for k in ['agl_1', 'gripper_pos', 'agl_1_act', 'gripper_act']:
        m = scaler.mean_dict[k].data
        s = scaler.std_dict[k].data
        print(f"  {k}: mean={m.flatten().mean().item():.4f}  std={s.flatten().mean().item():.4f}")

if __name__ == "__main__":
    main()
