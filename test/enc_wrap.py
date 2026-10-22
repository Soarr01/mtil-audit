"""
enc_wrap.py — run an existing eval/timing script (evaluate_step_fs.py, time_rollout.py, ...) UNCHANGED, but with
MambaPolicy built for a lighter DINOv2 encoder (ENC=vits14|vitb14), via mamba_policy_enc.make_policy.
Usage: ENC=vits14 PYTHONPATH=<MTIL> python enc_wrap.py evaluate_step_fs.py 50 <ckpt_dir>
"""
import os, sys, types, runpy
import mamba_policy_par as MP
import mamba_policy_enc as ME

ENC = os.environ.get('ENC', 'vitl14')
_orig = types.SimpleNamespace(MambaPolicy=MP.MambaPolicy, FrozenDinov2=MP.FrozenDinov2)
MP.MambaPolicy = lambda **kw: ME.make_policy(_orig, ENC, **kw)
print(f'[enc_wrap] MambaPolicy -> DINOv2 {ENC}', flush=True)
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
