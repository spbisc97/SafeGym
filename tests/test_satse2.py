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
    env.close()


def test_reward():
    from safegym.envs import Satellite_SE2
    from gymnasium.wrappers.time_limit import TimeLimit
    import time
    from matplotlib import pyplot as plt

    env = Satellite_SE2(
        underactuated=False,
        render_mode="human",
        max_action=np.float32(1),
        step=np.float32(0.1),
        unconstrained=True,
    )
    TimeLimit(env, max_episode_steps=10000)
    env.reset()
    rewards = []
    done = False
    while not done:
        action = (-k @ env.chaser.get_state()).astype(np.float32)
        np.clip(action, -1, 1, out=action)
        obs, reward, trunc, term, info = env.step(action)
        rewards.append(reward)

        done = term or trunc
    # X = env.render()
    # img = Image.fromarray(X)
    # img.show()
    time.sleep(0.00001)

    env.close()
    # plt.switch_backend("`")
    print("Total reward: ", sum(rewards))
    print("total steps: ", len(rewards))
    # plt.subplot(4, 1, 1)
    # plt.plot([1, 3, 7], [4, 6, -1])
    # plt.show()
    # plt.subplot(4, 1, 2)

    # plt.plot(np.array(rewards))
    # plt.show()
    # plt.subplot(4, 1, 3)
    # time.sleep(5)
    seprew = np.array(env.separate_reward_history)
    print(seprew[:, 1].shape)
    plt.ioff()
    plt.subplot(2, 1, 1)
    plt.plot(seprew[:, 0], label="rel_distance")
    plt.plot(seprew[:, 1], label="distance")
    plt.plot(seprew[:, 2], label="control")
    plt.plot(seprew[:, 3], label="speed")
    plt.plot(seprew[:, 4], label="angle_speed")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(rewards[0:-1], label="total")
    plt.legend()

    plt.show()

    plt.close("all")  # <--- added


def test_reward_doubleint():
    from safegym.envs import Satellite_SE2
    from gymnasium.wrappers.time_limit import TimeLimit
    import time
    from matplotlib import pyplot as plt

    env = Satellite_SE2(
        underactuated=False,
        render_mode="human",
        max_action=np.float32(1),
        step=np.float32(0.1),
        unconstrained=True,
        doubleintegrator=True,
    )
    TimeLimit(env, max_episode_steps=10000)
    env.reset()
    rewards = []
    done = False
    while not done:
        action = (
            -0.0001 * env.chaser.get_state()[0:3]
            - 0.1 * env.chaser.get_state()[3:6]
        ).astype(np.float32)
        np.clip(action, -1, 1, out=action)
        obs, reward, trunc, term, info = env.step(action)
        rewards.append(reward)

        done = term or trunc
    # X = env.render()
    # img = Image.fromarray(X)
    # img.show()
    time.sleep(0.00001)

    env.close()
    # plt.switch_backend("`")
    print("Total reward: ", sum(rewards))
    print("total steps: ", len(rewards))
    # plt.subplot(4, 1, 1)
    # plt.plot([1, 3, 7], [4, 6, -1])
    # plt.show()
    # plt.subplot(4, 1, 2)

    # plt.plot(np.array(rewards))
    # plt.show()
    # plt.subplot(4, 1, 3)
    # time.sleep(5)
    seprew = np.array(env.separate_reward_history)
    print(seprew[:, 1].shape)
    plt.ioff()

    plt.subplot(2, 1, 1)
    plt.plot(seprew[:, 0], label="rel_distance")
    plt.plot(seprew[:, 1], label="distance")
    plt.plot(seprew[:, 2], label="control")
    plt.plot(seprew[:, 3], label="speed")
    plt.plot(seprew[:, 4], label="angle_speed")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(rewards[0:-1], label="total")
    plt.legend()

    plt.show()

    plt.close("all")  # <--- added


def test_trajectory():
    from safegym.envs import Satellite_SE2
    from matplotlib import pyplot as plt
    from PIL import Image

    for i in range(5):
        env = Satellite_SE2(
            render_mode="human",
            initial_integration_steps=np.array([100, 10000]),
            step=np.float32(0.1),
        )
        env.reset()
        empty = np.zeros((2,), dtype=np.uint8)
        rewards = []
        for i in range(4_000):
            obs, rew, trunc, term, info = env.step(empty)
            rewards.append(rew)
        env.close()
        print("Total reward: ", sum(rewards))


def test_underactuated():
    import safegym
    import gymnasium as gym

    env = gym.make("Satellite-SE2-v0", underactuated=True)
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())

    env.close()


def test_fullyactuated():
    import safegym
    import gymnasium as gym

    env = gym.make("Satellite-SE2-v0", underactuated=False)
    env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())
    env.close()


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
    test_reward()
