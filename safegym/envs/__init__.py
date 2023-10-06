from gymnasium.envs.registration import register

from safegym.envs.Snake_v0 import SnakeEnv
from safegym.envs.Satellite_rot import Satellite_rot
from safegym.envs.Satellite_tra import Satellite_tra
from safegym.envs.Satellite_mujoco import MujSatEnv
from safegym.envs.Satellite_SE2 import Satellite_SE2

__al__ = [
    "SnakeEnv",
    "Satellite_rot",
    "Satellite_tra",
    "MujSatEnv",
    "Satellite_SE2",
]

__version__ = "0.1"

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
