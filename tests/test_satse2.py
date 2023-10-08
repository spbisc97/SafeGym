import pytest
import sys
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
k = np.array(
    [
        [
            0.000212278114670,
            -0.000083701859629,
            0.000000000000000,
            0.098840940992828,
            0.018561185733915,
            0.000000000000000,
        ],
        [
            0.000133845264885,
            0.000054717444153,
            0.000000000000000,
            0.018561185733917,
            0.074573364422630,
            0.000000000000000,
        ],
        [
            0.000000000000000,
            0.000000000000000,
            0.000100000000000,
            -0.000000000000000,
            0.000000000000000,
            0.015818032751518,
        ],
    ],
    dtype=np.float32,
)


def test_imports():
    from safegym.envs import Satellite_SE2


def test_env():
    from safegym.envs import Satellite_SE2

    env = Satellite_SE2()
    env.reset()
    env.step(env.action_space.sample())
    env.close()


def test_speed():
    import logging
    from safegym.envs import Satellite_SE2
    import time

    env = Satellite_SE2()
    env.reset()
    start = time.time()
    for i in range(1000):
        env.step(env.action_space.sample())
    end = time.time()
    env.close()
    print("Time taken for 1000 steps: ", end - start)
    print("Time taken for 1 step: ", (end - start) / 1000)


def test_graph():
    from safegym.envs import Satellite_SE2

    env = Satellite_SE2(render_mode="human")
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())
    env.close()


def test_rgb_graph():
    from safegym.envs import Satellite_SE2
    from matplotlib import pyplot as plt
    from PIL import Image

    env = Satellite_SE2(render_mode="rgb_array")
    env.reset()
    for i in range(10):
        env.step(env.action_space.sample())
    frame = env.render()
    pause = input("Press enter to continue")
    env.close()


def test_reward():
    from safegym.envs import Satellite_SE2

    env = Satellite_SE2(
        underactuated=False,
        render_mode="human",
        max_action=np.float32(1),
        step=np.float32(0.1),
    )
    env.reset()
    rewards = []
    done = False
    while not done:
        obs, reward, trunc, term, info = env.step(
            -k @ env.chaser.get_state() * (0.4)
        )
        rewards.append(reward)
        done = term or trunc
    env.close()
    print("Total reward: ", sum(rewards))
    print("total steps: ", len(rewards))


def test_trajectory():
    from safegym.envs import Satellite_SE2
    from matplotlib import pyplot as plt
    from PIL import Image

    env = Satellite_SE2()
    env.reset()
    empty = np.zeros((2,), dtype=np.uint8)
    rewards = []
    for i in range(100_000):
        obs, rew, trunc, term, info = env.step(empty)
        rewards.append(rew)
    env.close()
    print("Total reward: ", sum(rewards))


def profile():
    import cProfile as profile
    import os

    prof = profile.Profile()
    prof.enable()
    test_trajectory()
    prof.disable()
    prof.dump_stats("test_env.prof")
    import os

    os.system("snakeviz test_env.prof")


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    profile()
