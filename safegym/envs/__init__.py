from gymnasium.envs.registration import register
import warnings

# Import envs defensively to avoid optional dependency failures (e.g., pygame, mujoco)
try:
    from safegym.envs.Snake_v0 import SnakeEnv  # requires pygame
    _HAS_SNAKE = True
except Exception as _e:
    SnakeEnv = None  # type: ignore
    _HAS_SNAKE = False
    warnings.warn(f"SnakeEnv unavailable: {_e}")

from safegym.envs.Satellite_rot import Satellite_rot
from safegym.envs.Satellite_tra import Satellite_tra

try:
    from safegym.envs.Satellite_mujoco import MujSatEnv  # may require mujoco extras
    _HAS_MUJOCO = True
except Exception as _e:
    MujSatEnv = None  # type: ignore
    _HAS_MUJOCO = False
    warnings.warn(f"Mujoco env unavailable: {_e}")

from safegym.envs.Satellite_SE2 import Satellite_SE2

__al__ = [
    "SnakeEnv",
    "Satellite_rot",
    "Satellite_tra",
    "MujSatEnv",
    "Satellite_SE2",
]

__version__ = "0.1"

if _HAS_SNAKE:
    register(
        id="Snake-v0",
        entry_point="safegym.envs.Snake_v0:SnakeEnv",
    )
register(
    id="Satellite-v0",
    entry_point="safegym.envs.Satellite:Satellite_base",
    max_episode_steps=20000,
    reward_threshold=25000.0,
)
register(
    id="Satellite-discrete-v0",
    entry_point="safegym.envs.Satellite:Satellite_base",
    max_episode_steps=15000,
    reward_threshold=25000.0,
    kwargs={"action_space": "discrete"},
)
register(
    id="Satellite-rot-v0",
    entry_point="safegym.envs.Satellite_rot:Satellite_rot",
    max_episode_steps=5000,  # pretty fast
    reward_threshold=0.0,
)
register(
    id="Satellite-tra-v0",
    entry_point="safegym.envs.Satellite_tra:Satellite_tra",
    max_episode_steps=60_000,
    reward_threshold=0.0,
)

if _HAS_MUJOCO:
    register(
        id="Satellite-mj-v0",
        entry_point="safegym.envs.Satellite_mujoco:MujSatEnv",
        max_episode_steps=60_000,
        reward_threshold=0.0,
    )

register(
    id="Satellite-SE2-v0",
    entry_point="safegym.envs.Satellite_SE2:Satellite_SE2",
    max_episode_steps=100_000,
    reward_threshold=100_000,
)
