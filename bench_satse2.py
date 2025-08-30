import time
import numpy as np
from safegym.envs import Satellite_SE2


def bench(name, env_kwargs, steps=100_000):
    env = Satellite_SE2(**env_kwargs)
    obs, info = env.reset()
    # zero action of correct shape
    if env.underactuated:
        action = np.zeros((2,), dtype=np.float32)
    else:
        action = np.zeros((3,), dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    t1 = time.perf_counter()
    env.close()
    dt = t1 - t0
    sps = steps / dt
    print(f"{name}: {steps} steps in {dt:.3f}s -> {sps:.1f} steps/s")


if __name__ == "__main__":
    common = dict(render_mode=None, step=np.float32(0.05), validate_actions=False)

    bench(
        "underactuated,no-history",
        {**common, "underactuated": True, "track_history": False},
    )
    bench(
        "underactuated,history",
        {**common, "underactuated": True, "track_history": True},
    )
    bench(
        "fullyactuated,no-history",
        {**common, "underactuated": False, "track_history": False},
    )

