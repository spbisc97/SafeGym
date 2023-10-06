import pytest
import sys
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def test_imports():
    from safegym.envs import Satellite_SE2
    from safegym.envs import Satellite_rot


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
    for i in range(100000):
        env.step(env.action_space.sample())
    end = time.time()
    env.close()
    print("Time taken for 100000 steps: ", end - start)
    print("Time taken for 1 step: ", (end - start) / 100000)
