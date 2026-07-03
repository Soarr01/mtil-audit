"""
verify_mem_env.py — sanity-check the delayed-cue memory env WITHOUT a policy.
(a) make_sim_env('sim_transfer_cube_mem') builds
(b) cue (green) is present for step<tau and gone after, for BOTH target bins
(c) env.task.get_reward runs without error and max_reward==4
Run in ssm_vla env after `source eval_render.sh`.
"""
import numpy as np
import sim_env as SE
from sim_env import make_sim_env, BOX_POSE

TAU = 12

def green_cue_pixels(img):
    # green-dominant pixels (cue is rgba 0 1 0): G high, R/B low
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    mask = (g > 120) & (r < 100) & (b < 100)
    return int(mask.sum())

def sample_box_pose():
    rng = np.vstack([[0.0, 0.2], [0.4, 0.6], [0.05, 0.05]])
    p = np.random.uniform(rng[:, 0], rng[:, 1])
    return np.concatenate([p, [1, 0, 0, 0]])

print("(a) building env...")
env = make_sim_env('sim_transfer_cube_mem')
assert env.task.max_reward == 4, env.task.max_reward
print("    OK: env built, max_reward =", env.task.max_reward)

np.random.seed(0)
for target in (0, 1):
    SE.MEM_TARGET[0] = target
    SE.MEM_TAU[0] = TAU
    BOX_POSE[0] = sample_box_pose()
    ts = env.reset()
    # reward machinery at reset
    r0 = env.task.get_reward(env.physics)
    counts = []
    for step in range(30):
        img = ts.observation['images']['top']
        counts.append(green_cue_pixels(img))
        # target_bin should be exposed in obs
        assert 'target_bin' in ts.observation, "target_bin missing from obs"
        assert int(ts.observation['target_bin'][0]) == target
        act = np.zeros(14)  # hold pose (random not needed; reveal/hide is step-driven)
        ts = env.step(act)
        _ = env.task.get_reward(env.physics)  # (c) reward runs every step w/o error
    before = counts[:TAU]
    after = counts[TAU:]
    print(f"\n  target_bin={target} ({'LEFT' if target==0 else 'RIGHT'})")
    print(f"    cue px steps[0:{TAU}]  : {before}")
    print(f"    cue px steps[{TAU}:30] : {after}")
    print(f"    max before tau = {max(before)} (expect >0)")
    print(f"    max after  tau = {max(after)} (expect ==0)")
    assert max(before) > 0, "cue NOT visible before tau!"
    assert max(after) == 0, "cue STILL visible after tau!"
    print(f"    get_reward at reset = {r0} (ran OK)")

print("\nALL CHECKS PASSED: env builds, cue present<tau & gone>=tau (both bins), reward OK, max_reward==4.")
