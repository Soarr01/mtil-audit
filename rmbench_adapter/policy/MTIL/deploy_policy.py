"""
MTIL adapter for RMBench (runs in the `rmbench` env). The MTIL policy itself runs in the `ssm_vla` env as
MTIL/test/rmbench_policy_server.py; this adapter ships (joint_action.vector, head_camera RGB) each tick and applies the
returned 14-D joint target with take_action(action_type='qpos') — RoboTwin's joint-space convention, matching the
training data (state_t = vector[t], action_t = vector[t+1]).
"""
import os, json, atexit
import numpy as np
from multiprocessing.connection import Client

# Per-episode diagnostic log for put_back_block (memory-specific secondary metric): furthest stage reached
# (0 start, 1 = block centred + button pressed, 2 = returned = success) and, at episode end, which mat the block
# sits on vs its original mat (distinguishes "returned to the WRONG mat" = memory failure from manipulation failure).
DIAG = os.environ.get('MTIL_DIAG_LOG')


class MTILClient:
    def __init__(self, port):
        self.conn = Client(('localhost', int(port)), authkey=b'mtil-rmbench')
        self.summary = None; self.ep = 0
        atexit.register(self.flush)

    def flush(self):
        if DIAG and self.summary is not None:
            try:
                with open(DIAG, 'a') as f: f.write(json.dumps(self.summary, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
            except Exception as e:           # diagnostics must never break the evaluation
                print('[MTIL diag] write failed:', repr(e))
        self.summary = None

    def reset(self):
        self.flush(); self.ep += 1
        self.conn.send(('reset',)); assert self.conn.recv() == 'ok'

    def act(self, qpos, rgb):
        q = np.asarray(qpos, dtype=np.float32).reshape(14)
        im = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., :3])
        self.conn.send(('act', q.tobytes(), im.tobytes(), im.shape[0], im.shape[1]))
        return np.frombuffer(self.conn.recv(), np.float32).astype(np.float64)


def get_model(usr_args):
    return MTILClient(usr_args.get('server_port', 6010))


def _diag(TASK_ENV, model):
    try:
        mats = [np.asarray(m.get_pose().p)[:2] for m in TASK_ENV.mat_lst]
        orig = int(np.argmin([np.linalg.norm(m - np.asarray(TASK_ENV.target_pose)[:2]) for m in mats]))
        b = np.asarray(TASK_ENV.block.get_pose().p)
        d = [float(np.linalg.norm(b[:2] - m)) for m in mats]; j = int(np.argmin(d))
        prev = model.summary or {}
        model.summary = {'episode': model.ep, 'seed': int(getattr(TASK_ENV, 'seed', -1)) if hasattr(TASK_ENV, 'seed') else None,
                         'orig_mat': orig, 'mat_name': TASK_ENV.mat_name,
                         'max_stage': max(int(TASK_ENV.stage_id), prev.get('max_stage', 0)),
                         'final_block_xyz': [round(float(x), 4) for x in b], 'nearest_mat': j,
                         'nearest_mat_dist': round(d[j], 4), 'on_mat': bool(d[j] < 0.03 and b[2] < 0.77),
                         'steps': int(TASK_ENV.take_action_cnt), 'success': bool(TASK_ENV.eval_success)}
    except Exception as e:
        model.summary = {'episode': model.ep, 'diag_error': repr(e)}


def eval(TASK_ENV, model, observation):
    a = model.act(observation['joint_action']['vector'], observation['observation']['head_camera']['rgb'])
    TASK_ENV.take_action(a, action_type='qpos')
    if DIAG and getattr(TASK_ENV, 'task_name', '') == 'put_back_block':
        _diag(TASK_ENV, model)


def reset_model(model):
    model.reset()
