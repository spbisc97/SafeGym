from gymnasium.envs.registration import register

from envs.Snake_v0 import SnakeEnv
from envs.Satellite_rot import Satellite_rot
from envs.Satellite_tra import Satellite_tra
from envs.Satellite_mujoco import MujSatEnv

__al__ = ["SnakeEnv", "Satellite_rot", "Satellite_tra", "MujSatEnv"]

__version__ = "0.0.1"

register(
    id="Snake-v0",
    entry_point="Envs.Snake_v0:SnakeEnv",
)
register(
    id="Satellite-v0",
    entry_point="Envs.Satellite:Satellite_base",
    max_episode_steps=20000,
    reward_threshold=25000.0,
)
register(
    id="Satellite-discrete-v0",
    entry_point="Envs.Satellite:Satellite_base",
    max_episode_steps=15000,
    reward_threshold=25000.0,
    kwargs={"action_space": "discrete"},
)
register(
    id="Satellite-rot-v0",
    entry_point="Envs.Satellite_rot:Satellite_rot",
    max_episode_steps=5000,  # pretty fast
    reward_threshold=0.0,
)
register(
    id="Satellite-tra-v0",
    entry_point="Envs.Satellite_tra:Satellite_tra",
    max_episode_steps=60_000,
    reward_threshold=0.0,
)

register(
    id="Satellite-mj-v0",
    entry_point="Envs.Satellite_mujoco:MujSatEnv",
    max_episode_steps=60_000,
    reward_threshold=0.0,
)
