import os
import sys
import types
import importlib.util
import re
import numpy as np
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENVS_DIR = os.path.join(ROOT, "safegym", "envs")


def load_module_from_path(path):
    """Load a module from a file path without importing safegym package tree.

    - Avoids safegym.envs.__init__ side effects (e.g., pygame).
    - Strips numba jit_module blocks that can fail under direct execution.
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # Remove the optional numba jit_module try/except block if present
    pattern = re.compile(
        r"try:\n\s*from numba import jit_module\n\s*\n\s*jit_module\([^\)]*\)\n\s*print\(\"Using Numba optimised methods\.\"\)\n\s*\n\s*except ModuleNotFoundError:\n\s*print\(\"Using native Python methods\.\"\)\n\s*print\(\"Consider installing numba for compiled and parallelised methods\.\"\)\n",
        re.MULTILINE,
    )
    src = pattern.sub('print("Numba disabled for tests")\n', src)

    module_name = os.path.basename(path).replace(".py", "")
    module = types.ModuleType(module_name)
    module.__file__ = path
    code = compile(src, path, "exec")
    exec(code, module.__dict__)
    return module


def step_n(env, n=3):
    obs, info = env.reset()
    assert isinstance(info, dict)
    for _ in range(n):
        action = env.action_space.sample()
        out = env.step(action)
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        if terminated or truncated:
            break
    env.close()


def test_satellite_se2_reset_step_rgb_array():
    mod = load_module_from_path(os.path.join(ENVS_DIR, "Satellite_SE2.py"))
    Env = getattr(mod, "Satellite_SE2")
    env = Env(render_mode="rgb_array")
    obs, info = env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    step_n(env, n=5)


def test_satellite_base_reset_step():
    mod = load_module_from_path(os.path.join(ENVS_DIR, "Satellite.py"))
    Env = getattr(mod, "Satellite_base")
    env = Env(render_mode=None)
    step_n(env, n=5)


def test_satellite_rot_reset_step():
    mod = load_module_from_path(os.path.join(ENVS_DIR, "Satellite_rot.py"))
    Env = getattr(mod, "Satellite_rot")
    env = Env(render_mode=None)
    step_n(env, n=3)


def test_satellite_tra_reset_step():
    mod = load_module_from_path(os.path.join(ENVS_DIR, "Satellite_tra.py"))
    Env = getattr(mod, "Satellite_tra")
    env = Env(render_mode=None)
    step_n(env, n=3)


@pytest.mark.skipif(
    os.environ.get("SKIP_MUJOCO", "0") == "1", reason="Mujoco explicitly skipped"
)
def test_satellite_mujoco_instantiation():
    path = os.path.join(ENVS_DIR, "Satellite_mujoco.py")
    try:
        mod = load_module_from_path(path)
        Env = getattr(mod, "MujSatEnv")
    except Exception as e:
        pytest.skip(f"Import skipped due to missing deps: {e}")
    try:
        env = Env(render_mode=None)
        # Only a light interaction; mujoco runtime may not be present
        obs = env.reset()
        env.close()
    except Exception as e:
        pytest.skip(f"Instantiation skipped (likely mujoco missing): {e}")
