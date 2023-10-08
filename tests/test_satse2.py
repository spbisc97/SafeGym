import pytest
import sys
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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


if __name__ == "__main__":
    test_rgb_graph()
    import sys

    sys.path.append("..")
