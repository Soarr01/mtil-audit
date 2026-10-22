"""
scan_step_equiv.py — reusable parallel-scan ≡ recurrent-step() equivalence checker (resubmission, C1 class audit).
For each target policy it builds the model with RANDOM weights (no checkpoint / data / simulator needed),
feeds ONE random input sequence through (a) the parallel training path (chunked SSD scan) and (b) the
deployed recurrent step() frame-by-frame, and reports per-frame max|diff| and relative error.
A correct recurrent controller matches the scan to float32 round-off; a wiring bug diverges by O(1).

Targets:
  mtil_fixed     test/mamba_policy_par.py (our C1-fixed copy)                  forward_sequence vs step()
  mtil_pristine  mtil_original_audit/train/mamba_policy_par.py (upstream 9bea607) forward_sequence vs step()
  chronos3d      third_party_audit/chronos_0f2bf8d/mamba_policy_par_3D_IMLE.py  forward_features vs step()
  chronos2d      third_party_audit/chronos_0f2bf8d/mamba_policy_par_2D_IMLE.py  forward_features vs step()
  chronos2d_cpupatch  same, but step() under the real-world deploy runtime patches of inference_choronos.py
                 (Torch RMSNormGated, no fused step kernels) vs the Triton training path
For Chronos the step() output is a stochastic IMLE+flow action (fresh latent noise every tick), so the
compared quantity is the deterministic Mamba conditioning `mamba_cond` that step() hands to imle_generator
(captured by a forward-pre-hook on the deployed code path, not re-implemented), against forward_features().

Usage: PYTHONPATH=<MTIL> python scan_step_equiv.py [targets_csv] [T]      (default: all, T=300 > chunk_size 256)
Writes mtil_data/equiv/scan_step_equiv_T<T>.json
"""
import os, sys, json, importlib.util, math
import numpy as np, torch

M = '/data/nas07_new/PersonalData/phuocthien/ssm_vla/MTIL'
TARGETS = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else ['mtil_fixed', 'mtil_pristine', 'chronos3d', 'chronos2d', 'chronos2d_cpupatch']
T = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = '/data/nas07_new/PersonalData/phuocthien/mtil_data/equiv'; os.makedirs(OUT, exist_ok=True)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # true fp32 reference


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


class _Cached(torch.nn.Module):
    """stand-in vision backbone returning a given feature map (lets step() run on synthetic features)"""
    def __init__(self, raw): super().__init__(); self.raw = raw
    def forward(self, x): return self.raw


def report(name, ref, rec, note):
    d = (ref - rec).abs().reshape(ref.shape[0], -1).max(dim=1).values.cpu().numpy()      # per-frame max|diff|
    scale = float(ref.abs().max())
    r = {'target': name, 'T': T, 'compared': note, 'max_abs_diff': float(d.max()), 'ref_max_abs': scale,
         'rel_err': float(d.max() / (scale + 1e-12)), 'diff_first10': float(d[:10].max()), 'diff_last10': float(d[-10:].max()),
         'per_frame_max_abs_diff': [float(x) for x in d]}
    print(f"[{name}] {note}: max|diff|={r['max_abs_diff']:.3e} (ref scale {scale:.3e}, rel {r['rel_err']:.2e}) "
          f"first10 {r['diff_first10']:.2e} last10 {r['diff_last10']:.2e}", flush=True)
    return r


def run_mtil(name, path):
    MP = load(name, path)
    cfg = MP.MambaConfig(); cfg.camera_names = ['top']; cfg.embed_dim = 2048; cfg.lowdim_dim = 14
    cfg.d_model = 2048; cfg.action_dim = 14; cfg.sum_camera_feats = False; cfg.num_blocks = 4
    torch.manual_seed(0)
    pol = MP.MambaPolicy(camera_names=cfg.camera_names, embed_dim=cfg.embed_dim, lowdim_dim=cfg.lowdim_dim, d_model=cfg.d_model,
                         action_dim=cfg.action_dim, sum_camera_feats=cfg.sum_camera_feats, num_blocks=cfg.num_blocks, mamba_cfg=cfg).to(dev).eval()
    g = torch.Generator().manual_seed(1)
    raw = torch.randn(T, 1024, 35, 46, generator=g)            # synthetic DINOv2 maps (CPU; moved per chunk)
    q = torch.randn(1, T, 14, generator=g).to(dev)
    with torch.no_grad():
        cam = torch.cat([pol.process_vision_chunk(raw[i:i + 16].to(dev)) for i in range(0, T, 16)], 0).unsqueeze(0)
        ref = pol.forward_sequence(q, {'top': cam})[0]          # [T,16,14]
        h = pol.init_hidden_states(1); outs = []
        bb = pol.shared_backbone
        dummy = torch.zeros(1, 3, 8, 8, device=dev)
        for t in range(T):
            pol.shared_backbone = _Cached(raw[t:t + 1].to(dev))
            a, h = pol.step(q[:, t], {'top': dummy}, h); outs.append(a[0])
        pol.shared_backbone = bb
    rec = torch.stack(outs, 0)
    return report(name, ref.float(), rec.float(), 'action chunk [16x14] per frame: forward_sequence vs step()')


class _StubSIGReg(torch.nn.Module):
    """Chronos 2D (real_wolrd/common) constructs `SIGReg(...)` but never defines or imports it, and never uses
    `self.sigreg` afterwards -> the released module raises NameError at construction. We stub it (unused) to
    test the rest of the file; the NameError itself is recorded as a finding."""
    def __init__(self, *a, **k): super().__init__()


def _torch_rmsnorm_gated_from_inference():
    """load TorchRMSNormGated verbatim from Chronos' real-world inference script (its CPU deploy path)"""
    import ast, torch.nn.functional as F
    src = open(f'{M}/test/third_party_audit/chronos_0f2bf8d/inference_choronos.py').read()
    node = [n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'TorchRMSNormGated'][0]
    ns = {'torch': torch, 'nn': torch.nn, 'F': F, 'Optional': None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'inference_choronos.py', 'exec'), ns)
    return ns['TorchRMSNormGated']


def run_chronos(name, path):
    import torchvision.models as tvm
    _r18 = tvm.resnet18; tvm.resnet18 = lambda *a, **k: _r18(weights=None)    # random init, no download
    cpu_patch = name.endswith('_cpupatch')
    C = load(name, path)
    if not hasattr(C, 'SIGReg'): C.SIGReg = _StubSIGReg
    cfg = C.MambaConfig()
    # dimensions from Chronos' own training script (RMBench/policy/Chronos/train_par_3D_IMLE_EE.py:365-370)
    cfg.embed_dim = 1024; cfg.d_model = 1024; cfg.action_dim = 16; cfg.lowdim_dim = 16; cfg.num_blocks = 6; cfg.future_steps = 16
    torch.manual_seed(0)
    pol = C.MambaPolicy(camera_names=cfg.camera_names, embed_dim=cfg.embed_dim, lowdim_dim=cfg.lowdim_dim, d_model=cfg.d_model,
                        action_dim=cfg.action_dim, num_blocks=cfg.num_blocks, future_steps=cfg.future_steps, mamba_cfg=cfg).to(dev).eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, T, cfg.d_model, generator=g).to(dev)       # fused per-frame feature (in_proj is Identity)
    with torch.no_grad():
        ref = pol.forward_features(x)[0]                         # [T, d_model] training path (Triton kernels)
    spol = pol
    if cpu_patch:
        # the deploy runtime of real_wolrd/inference/inference_choronos.py:patch_cpu_mamba_kernels():
        # RMSNormGated -> TorchRMSNormGated, selective_state_update / causal_conv1d_update -> None (pure-torch step)
        C2 = load(name + '_rt', path)
        if not hasattr(C2, 'SIGReg'): C2.SIGReg = _StubSIGReg
        C2.RMSNormGated = _torch_rmsnorm_gated_from_inference(); C2.selective_state_update = None; C2.causal_conv1d_update = None
        spol = C2.MambaPolicy(camera_names=cfg.camera_names, embed_dim=cfg.embed_dim, lowdim_dim=cfg.lowdim_dim, d_model=cfg.d_model,
                              action_dim=cfg.action_dim, num_blocks=cfg.num_blocks, future_steps=cfg.future_steps, mamba_cfg=cfg).to(dev).eval()
        miss, unexp = spol.load_state_dict(pol.state_dict(), strict=False)
        print(f'[{name}] deploy-runtime model: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    cap = []
    hk = spol.imle_generator.register_forward_pre_hook(lambda m, args: cap.append(args[0].detach().clone()))
    with torch.no_grad():
        h = spol.init_hidden_states(1)
        for t in range(T):
            _, h = spol.step(x[:, t], h)
    hk.remove(); tvm.resnet18 = _r18
    rec = torch.cat(cap, 0)                                       # [T, d_model] mamba_cond seen by step()
    return report(name, ref.float(), rec.float(), 'mamba_cond per frame: forward_features vs step() (hooked)')


PATHS = {'mtil_fixed': f'{M}/test/mamba_policy_par.py',
         'mtil_pristine': '/data/nas07_new/PersonalData/phuocthien/mtil_original_audit/train/mamba_policy_par.py',
         'chronos3d': f'{M}/test/third_party_audit/chronos_0f2bf8d/mamba_policy_par_3D_IMLE.py',
         'chronos2d': f'{M}/test/third_party_audit/chronos_0f2bf8d/mamba_policy_par_2D_IMLE.py',
         'chronos2d_cpupatch': f'{M}/test/third_party_audit/chronos_0f2bf8d/mamba_policy_par_2D_IMLE.py'}
results = []
for tname in TARGETS:
    try:
        r = (run_mtil if tname.startswith('mtil') else run_chronos)(tname, PATHS[tname])
    except Exception as e:
        import traceback; traceback.print_exc()
        r = {'target': tname, 'error': f'{type(e).__name__}: {e}'}; print(f'[{tname}] ERROR {r["error"]}', flush=True)
    results.append(r); torch.cuda.empty_cache()
out_name = f'{OUT}/scan_step_equiv_T{T}' + ('' if len(sys.argv) < 2 or sys.argv[1] == 'all' else '_' + sys.argv[1].replace(',', '+')) + '.json'
json.dump({'T': T, 'gpu': torch.cuda.get_device_name(0), 'results': results}, open(out_name, 'w'), indent=1)
print('EQUIV DONE', out_name, flush=True)
