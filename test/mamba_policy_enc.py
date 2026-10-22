"""
mamba_policy_enc.py — lighter-vision-encoder variant of MambaPolicy (resubmission, C2 "real lighter encoder").
Does NOT copy or edit mamba_policy_par.py: builds the ordinary MambaPolicy from whichever copy is passed
in (train/ or test/), then swaps
  * shared_backbone -> DINOv2 ViT-S/14 or ViT-B/14 (same patch 14, same adaptive resize, same "4th-from-last block")
  * the first Conv2d of spatial_adapter / spatial_adpater_low -> in_channels = that encoder's width (384 / 768)
Everything downstream (adapter spatial size 35x46 -> 23x18, Mamba blocks, heads) is unchanged, so the only
difference from the paper's model is the frozen encoder's size. Parameter names are unchanged.
"""
import torch
import torch.nn as nn

ENC_DIM = {'vits14': 384, 'vitb14': 768, 'vitl14': 1024}


def make_frozen_dino(MP, enc, layer_index=-4):
    """FrozenDinov2 from module MP, but loading dinov2_<enc> instead of the hardcoded vitl14."""
    fd = MP.FrozenDinov2.__new__(MP.FrozenDinov2)
    nn.Module.__init__(fd)
    fd.dino = torch.hub.load('facebookresearch/dinov2', f'dinov2_{enc}')
    fd.patch_size = 14; fd.layer_index = layer_index
    for p in fd.dino.parameters(): p.requires_grad_(False)
    fd.feature_hook = fd._register_hook()
    return fd


def make_policy(MP, enc, build_backbone=True, **kw):
    pol = MP.MambaPolicy(**kw)
    if enc == 'vitl14': return pol
    C = ENC_DIM[enc]
    if build_backbone:
        pol.shared_backbone = make_frozen_dino(MP, enc)
    for name in ('spatial_adapter', 'spatial_adpater_low'):
        seq = getattr(pol, name, None)
        if seq is not None and isinstance(seq[0], nn.Conv2d):
            old = seq[0]; seq[0] = nn.Conv2d(C, old.out_channels, old.kernel_size, padding=old.padding)
    return pol
