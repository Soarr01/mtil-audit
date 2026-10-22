"""
build_scaler_rmbench.py — per-joint normalization for an RMBench task, fitted on its TRAIN split with the same
dataset class / Scaler as scaler_transfer.pth (M_dataset.py main()). Usage: python build_scaler_rmbench.py <task>
"""
import sys
from M_dataset import MambaSequenceDataset
task = sys.argv[1] if len(sys.argv) > 1 else 'put_back_block'
ds = MambaSequenceDataset(root_dir=f"/data/nas07_new/PersonalData/phuocthien/mtil_data/rmbench_{task}", mode="train", use_pose10d=True)
ds.fit_scaler(batch_size=64, num_workers=4)
ds.save_scaler(f'/data/nas07_new/PersonalData/phuocthien/mtil_data/scaler_rmbench_{task}.pth')
print('SCALER DONE', task)
