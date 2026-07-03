"""
robossm_killtest.py — empirical audit of RoboSSM (youngjuY/RoboSSM @ 07df6df) Longhorn mixer.

Two-part kill-test, mirroring the MTIL step() audit (test/audit_upstream_step.py):

  PART A (no CUDA, no LIBERO needed): the advertised O(1)/step linear-time RECURRENT decode
  path is DEAD CODE. The `Longhorn(nn.Module)` mixer
  (libero/lifelong/models/longhorn.py:416) inherits only nn.Module and defines just
  __init__ + forward; its forward references self._get_states_from_cache(...) and self.step(...)
  for the inference_params branch, but NEITHER method is defined on the class (nor inherited).
  => the moment you pass inference_params (as any recurrent/incremental decode must), forward
     raises AttributeError. We reproduce the EXACT class body verbatim and trigger the crash.

  PART B (needs longhorn_cuda built): the path RoboSSM ACTUALLY uses at rollout
  (forward with inference_params=None -> longhorn_inner_fn, an O(L) parallel scan) is correct.
  We check the fused CUDA parallel kernel vs the repo's own longhorn_ref decomposition
  on random weights: max|Δ| should be ~fp tolerance.

Conclusion: RoboSSM's linear-time recurrence is not merely unused (as a grep would suggest) —
it is UNIMPLEMENTED and would crash if invoked; deployment silently falls back to the O(L)
parallel scan. This is the RoboSSM analogue of MTIL's broken step(): of the released SSM
robot policies we examined, none ships a working AND used recurrent rollout path.
"""
import math, sys, types
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

# ---- longhorn_inner_fn / longhorn_ref: only needed for PART B (graceful if not built) ----
try:
    from mamba_ssm.ops.selective_scan_interface import longhorn_inner_fn, longhorn_ref
    HAVE_CUDA_LONGHORN = True
except Exception as e:
    longhorn_inner_fn = longhorn_ref = None
    HAVE_CUDA_LONGHORN = False
    _IMPORT_ERR = repr(e)

# =====================================================================================
# VERBATIM copy of RoboSSM libero/lifelong/models/longhorn.py:416-515 (the Longhorn mixer).
# Only the heavy robomimic/libero imports at the top of the upstream file are dropped — the
# class body below is byte-for-byte the upstream definition (no step / _get_states_from_cache).
# =====================================================================================
class Longhorn(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 conv_bias=True, bias=False, use_fast_path=False, layer_idx=None,
                 device=None, dtype=None, beta=None, beta_ratio=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model; self.d_state = d_state; self.d_conv = d_conv; self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path; self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
                                kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.activation = "silu"; self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state*2, bias=False, **factory_kwargs)
        self.dt_head = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device)); self.D._no_weight_decay = True
        self.beta = beta; self.beta_ratio = beta_ratio

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)   # <-- undefined
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)                # <-- undefined
                return out
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                       "d (b l) -> b d l", l=seqlen)
        y = longhorn_inner_fn(xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight,
                              self.dt_head.weight, None, D=self.D, delta_bias=self.dt_head.bias.float(),
                              out_proj_weight=self.out_proj.weight, beta=self.beta, beta_ratio=self.beta_ratio)
        return y
# =====================================================================================

def part_A_decode_path_is_dead_code():
    print("="*78); print("PART A — RECURRENT DECODE PATH (the advertised O(1)/step linear-time inference)")
    print("="*78)
    # static introspection: do the methods the forward() calls actually exist?
    for m in ("step", "_get_states_from_cache", "allocate_inference_cache"):
        present = hasattr(Longhorn, m)
        print(f"  Longhorn.{m:<26} defined on class/bases? {present}")
    m = Longhorn(d_model=64, d_state=16, layer_idx=0)   # CPU, random weights — no CUDA needed
    x = torch.randn(1, 4, 64)
    # InferenceParams stand-in: any recurrent decode passes this (seqlen_offset>0 == decoding)
    ip = types.SimpleNamespace(seqlen_offset=1, max_seqlen=8, max_batch_size=1, key_value_memory_dict={})
    try:
        _ = m.forward(x, inference_params=ip)
        print("  UNEXPECTED: decode path returned without error"); return False
    except AttributeError as e:
        print(f"  ✓ decode path CRASHES on first use: AttributeError: {e}")
        print("  => RoboSSM's linear-time recurrent decode is UNIMPLEMENTED dead code,")
        print("     not merely unused. Rollout must use inference_params=None (O(L) parallel scan).")
        return True

def part_B_used_parallel_path_is_correct():
    print("="*78); print("PART B — DEPLOYED PARALLEL PATH (longhorn_inner_fn, what get_action actually runs)")
    print("="*78)
    if not HAVE_CUDA_LONGHORN:
        print(f"  SKIPPED — longhorn_cuda not importable ({_IMPORT_ERR}). PART A stands alone."); return None
    torch.manual_seed(0)
    dev = 'cuda'; B, L, D, N = 1, 64, 256, 16
    m = Longhorn(d_model=D, d_state=N, layer_idx=0, beta=None, beta_ratio=None).to(dev).to(torch.float32)
    x = torch.randn(B, L, D, device=dev)
    xz = rearrange(m.in_proj.weight @ rearrange(x, "b l d -> d (b l)"), "d (b l) -> b d l", l=L)
    args = (xz, m.conv1d.weight, m.conv1d.bias, m.x_proj.weight, m.dt_head.weight, None, m.out_proj.weight)
    kw = dict(D=m.D, delta_bias=m.dt_head.bias.float())
    with torch.no_grad():
        y_par = longhorn_inner_fn(*args, **kw)        # fused CUDA parallel kernel (deployed)
        y_ref = longhorn_ref(*args, **kw)             # repo's own reference decomposition
    d = (y_par.float() - y_ref.float()).abs()
    print(f"  longhorn_inner_fn vs longhorn_ref : max|Δ|={d.max().item():.3e}  mean|Δ|={d.mean().item():.3e}")
    ok = d.max().item() < 1e-2
    print(f"  => deployed parallel scan {'MATCHES' if ok else 'DIVERGES from'} the reference SSM computation")
    return ok

if __name__ == "__main__":
    a = part_A_decode_path_is_dead_code()
    print()
    b = part_B_used_parallel_path_is_correct()
    print("\n" + "="*78)
    print(f"RoboSSM AUDIT VERDICT: decode-path-dead-code={a}  used-parallel-path-correct={b}")
    print("="*78)
