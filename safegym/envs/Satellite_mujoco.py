from typing import Any, Optional, SupportsFloat
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

import os

PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_XML_PATH = os.path.join(PATH, "assets", "satellite.xml")


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MujSatEnv(MujocoEnv, utils.EzPickle):
    """
    Description:
        A satellite is trying to catch a target.

    Args:
        MujocoEnv (_type_): _description_
        utils (_type_): _description_
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file=MODEL_XML_PATH,
        ctrl_cost_weight=0.1,
        distance_cost_weight=1.0,
        reset_noise_scale=5e-3,
        camera_id=0,
        frame_skip=2,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file=MODEL_XML_PATH,
            ctrl_cost_weight=ctrl_cost_weight,
            distance_cost_weight=distance_cost_weight,
            reset_noise_scale=reset_noise_scale,
            **kwargs,
        )
        self.distance_cost_weight = distance_cost_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.reset_noise_scale = reset_noise_scale
        observation_space = self.observation_space()
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            camera_id=camera_id,
            **kwargs,
        )
        self.reset()
        return

    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)

    def action_space(self):
        return Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)
        rewards = 0
        distance_cost = self.distance_cost()
        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost + distance_cost
        terminated = self.terminated
        observation = self._get_obs()
        info = dict(ctrl_cost=ctrl_cost, distance_cost=distance_cost)
        reward = rewards - costs
        if self.render_mode == "human":
            self.render()
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + 10 * self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    @property
    def terminated(self):
        return False

    def get_body_cvel(self, body_name):
        return self.data.body(body_name).cvel

    def get_body_xquat(self, body_name):
        return self.data.body(body_name).xquat

    def _get_obs(self):
        position = self.get_body_com("chaser")  # 3
        rotation = self.get_body_xquat("chaser")  # 4
        velocity = self.get_body_cvel("chaser")  # 3+3
        observation = np.concatenate((position, rotation, velocity))  # 13
        return observation

    def distance_cost(self):
        return self.distance_cost_weight * np.linalg.norm(
            self.get_body_com("chaser") - self.get_body_com("target")
        )

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def control_cost(self, action):
        control_cost = self.ctrl_cost_weight * np.square(action).sum()
        return control_cost


def double_check_env(env, warn=True):
    """check if the environment has a valid observation and action space"""
    obs, info = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break


if __name__ == "__main__":
    print("Hello World!")
    from gymnasium.utils import env_checker
    import gymnasium as gym
    import time
    from gymnasium.envs.registration import register

    register(
        id="Satellite-mj-v0",
        entry_point="Satellite_mujoco:MujSatEnv",
        max_episode_steps=60_000,
        reward_threshold=0.0,
    )
    env = gym.make(
        "Satellite-mj-v0",
        render_mode="human",
    )

    double_check_env(env)
