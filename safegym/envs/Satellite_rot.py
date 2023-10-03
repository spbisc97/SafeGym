import matplotlib as mpl


import matplotlib.pyplot as plt

import matplotlib.style as mplstyle

mplstyle.use("fast")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint, solve_ivp
from numba import jit
from numba.experimental import jitclass
import random
import time
import warnings

# suppress warnings
# warnings.filterwarnings("always", category=RuntimeWarning)


# ? Maybe it's better to normalize the observation space?
# ? Maybe it's better to normalize the action space?
# ? Maybe it's better to use a discrete action space?
# ? Maybe it's better to use a continuous action space?
# ? Maybe it's better to use a continuous action space with a discrete action space?

I = np.diag(np.array([8.33e-2, 1.08e-1, 4.17e-2], dtype=np.float32))
invI = np.diag(
    np.array([12.0048019207683, 9.25925925925926, 23.9808153477218], dtype=np.float32)
)


class Satellite_rot(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "observation_spaces": ["MlpPolicy", "MultiInputPolicy"],
        "control": ["Ml", "ModelPredictiveControl", "LQR", "PID", "Random", "Human"],
        "action_spaces": ["continuous", "discrete"],
        "matplotlib_backend": ["TkAgg", "Qt5Agg", "WXAgg", "GTKAgg", "Qt4Agg", "Agg"],
    }

    def __init__(
        self,
        render_mode=None,
        observation_space="MlpPolicy",
        action_space="continuous",
        control="Ml",
        matplotlib_backend=None,
    ):
        super(Satellite_rot, self).__init__()
        # define action and observation space being gymnasium.spaces
        assert control in self.metadata["control"]
        self.control = control

        assert action_space in self.metadata["action_spaces"]
        if action_space == "continuous":
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.int8)
        #   or action_space in self.metadata['action_spaces']
        #   could be  better to use normalized action space
        assert observation_space in self.metadata["observation_spaces"]
        if observation_space == "MlpPolicy":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "rot": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                    ),
                    "rot_speed": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                }
            )
        # or render_mode in self.metadata['render_modes']
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        assert (
            matplotlib_backend in mpl.rcsetup.all_backends or matplotlib_backend is None
        )
        if matplotlib_backend:
            mpl.use(matplotlib_backend)

        self.qd = np.array([1, 0, 0, 0], dtype=np.float32)
        self.vmax = 30
        self.prev_shaping = None
        self.subplots = None

    def step(self, action):
        reward = 0
        truncated = False
        filtered_action = self._action_filter(action)
        self.chaser.rotate(filtered_action)
        terminated = self._beyond_observational_space()  # or self.chaser.fuel_mass <= 0
        observation = self._get_observation()
        info = self._get_info()
        reward = self._reward_function(action, terminated)

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._remember(observation, filtered_action, reward, info)
            if self.render_mode == "human":
                self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_shaping = None
        self.subplots = None
        self.chaser = Chaser()
        state = np.zeros((7,), dtype=np.float32)
        # state[random.randint(0,2)] = random.randint(-100, 100)
        # q0 = np.random.rand(1, 4) - 0.5
        (eul0,) = (np.random.rand(1, 3) - 0.5) * np.pi * 2
        q0 = eul_to_quat(eul0)
        state[0:4] = q0 / np.linalg.norm(q0)
        state[4:8] = (np.random.rand(1, 3) - 0.5) * 0.5
        self.chaser.set_state(state)
        self.prev_shaping = self._shape_reward()
        info = self._get_info()
        observation = self._get_observation()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.statuses = np.array(self.chaser.state)
            self.norms = np.array([np.linalg.norm(self.chaser.state[0:4])])
            self.euler_angles = np.array(quaternion_to_euler(self.chaser.state[0:4]))
            self.actionss = np.array([]).reshape(0, 3)
            self.rewards = np.array([]).reshape(0, 1)
            self.rewards_sum = np.array([]).reshape(0, 1)
            self.infos = np.array(info)
            self.times = np.array([0])
        return observation, info

    def render(self):
        if not self.render_mode or (
            self.render_mode == "human" and self.times[-1] % 20 != 0
        ):
            return
        if self.subplots == None:
            if self.render_mode == "human":
                plt.ion()
            fig, ax = plt.subplots(4, 3, figsize=(10, 10), layout="constrained")
            lines = np.ma.zeros_like(ax)
            legend = np.array(
                [
                    ["roll", "pitch", "yaw"],
                    ["roll_speed", "pitch_speed", "yaw_speed"],
                    ["ux", "uy", "uz"],
                    ["reward", "reward_sum", "Quat norm"],
                ]
            )
            limits = np.array(
                [
                    [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
                    [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
                    [[-1, 1], [-1, 1], [-1, 1]],
                    [[-2, 0.3], [-100, 1], [0.5, 1.5]],
                ]
            )
            limits[2, :] = limits[2, :] * (Chaser.Tmax * (1.05))
            # rescale = np.array(
            #     [
            #         [False, False, False],
            #         ["roll_speed", "pitch_speed", "yaw_speed"],
            #         [False, False, False],
            #         ["reward", "reward_sum", ""],
            #     ]
            # )
            line_width = 0.6
            (lines[0, 0],) = ax[0, 0].plot(
                self.times, self.euler_angles[:, 0], linewidth=line_width
            )
            (lines[0, 1],) = ax[0, 1].plot(
                self.times, self.euler_angles[:, 1], linewidth=line_width
            )
            (lines[0, 2],) = ax[0, 2].plot(
                self.times, self.euler_angles[:, 2], linewidth=line_width
            )
            (lines[1, 0],) = ax[1, 0].plot(
                self.times, self.statuses[:, 4], linewidth=line_width
            )
            (lines[1, 1],) = ax[1, 1].plot(
                self.times, self.statuses[:, 5], linewidth=line_width
            )
            (lines[1, 2],) = ax[1, 2].plot(
                self.times, self.statuses[:, 6], linewidth=line_width
            )
            (lines[2, 0],) = ax[2, 0].plot(
                self.times[:-1], self.actionss[:, 0], linewidth=line_width
            )
            (lines[2, 1],) = ax[2, 1].plot(
                self.times[:-1], self.actionss[:, 1], linewidth=line_width
            )
            (lines[2, 2],) = ax[2, 2].plot(
                self.times[:-1], self.actionss[:, 2], linewidth=line_width
            )
            (lines[3, 0],) = ax[3, 0].plot(
                self.times[:-1], self.rewards[:], linewidth=line_width
            )
            (lines[3, 1],) = ax[3, 1].plot(
                self.times[:-1], self.rewards_sum[:], linewidth=line_width
            )
            (lines[3, 2],) = ax[3, 2].plot(
                self.times, self.norms[:], linewidth=line_width
            )

            for idx, x in np.ndenumerate(ax):
                # ax[idx[0], idx[1]].set_xlim(0, 100)
                ax[idx[0], idx[1]].set_ylim(limits[idx[0], idx[1], :])
                ax[idx[0], idx[1]].set_title(legend[idx[0], idx[1]])
                ax[idx[0], idx[1]].grid()
                if idx[0] != 3 and idx[1] != 0:
                    ax[idx[0], idx[1]].sharey(ax[idx[0], idx[1] - 1])

            self.subplots = (fig, ax, lines)

        else:
            fig, ax, lines = self.subplots
            lines[0, 0].set_data(self.times, self.euler_angles[:, 0])
            lines[0, 1].set_data(self.times, self.euler_angles[:, 1])
            lines[0, 2].set_data(self.times, self.euler_angles[:, 2])
            lines[1, 0].set_data(self.times, self.statuses[:, 4])
            lines[1, 1].set_data(self.times, self.statuses[:, 5])
            lines[1, 2].set_data(self.times, self.statuses[:, 6])
            lines[2, 0].set_data(self.times[:-1], self.actionss[:, 0])
            lines[2, 1].set_data(self.times[:-1], self.actionss[:, 1])
            lines[2, 2].set_data(self.times[:-1], self.actionss[:, 2])
            lines[3, 0].set_data(self.times[:-1], self.rewards[:])
            lines[3, 1].set_data(self.times[:-1], self.rewards_sum[:])
            lines[3, 2].set_data(self.times, self.norms[:])

        # mplstyle.use("fast")

        for idx, x in np.ndenumerate(ax):
            ax[idx[0], idx[1]].relim(visible_only=True)
            if idx[0] % 2 != 0:
                ax[idx[0], idx[1]].autoscale(enable=True, axis="y", tight=False)
            ax[idx[0], idx[1]].autoscale(enable=True, axis="x", tight=True)

        blit = True
        if blit:
            for idx, x in np.ndenumerate(ax):
                if lines[idx[0], idx[1]]:
                    ax[idx[0], idx[1]].draw_artist(lines[idx[0], idx[1]])
                    # ax[idx[0], idx[1]].draw_artist(ax[idx[0], idx[1]].patch)

            # for x in ax:
            #     for y in x:
            #         fig.canvas.blit(y.bbox)
            #         #memory leaks
        else:
            fig.canvas.draw()
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        # if self.render_mode == "human":

        return

    def close(self):
        super().close()
        if self.subplots:
            plt.close(self.subplots[0])
        return

    def _remember(self, sta, act, rew, inf={}):
        self.statuses = np.vstack((self.statuses, sta))
        self.norms = np.vstack((self.norms, np.linalg.norm(sta[0:4])))
        self.euler_angles = np.vstack((self.euler_angles, quaternion_to_euler(sta[:4])))
        self.actionss = np.vstack((self.actionss, act))
        self.rewards = np.vstack((self.rewards, rew))
        prev_rewards = 0 if len(self.rewards_sum) == 0 else self.rewards_sum[-1]
        self.rewards_sum = np.vstack((self.rewards_sum, prev_rewards + rew))
        self.infos = np.vstack((self.infos, inf))
        self.times = np.vstack((self.times, self.times[-1] + self.chaser.step))
        return

    def _get_observation(self):
        if isinstance(self.observation_space, spaces.Box):
            return np.concatenate(
                (
                    self.chaser.state,
                    # we could also pass just the error(quaternion) instead of the state
                    # till qd = (1 0 0 0), is exactly the same
                ),
                dtype=np.float32,
            )
        else:
            return {
                "rot": self.chaser.state[:4],
                "rot_speed": self.chaser.state[4:8],
            }

    def _reward_function(self, action, terminated=False):
        reward = 0
        term_reward = 0 if not terminated else -100_000

        # # Shaping Reward not used rn
        # shaping = self._shape_reward()
        #
        # if self.prev_shaping is not None:
        # reward = shaping - self.prev_shaping
        # self.prev_shaping = shaping

        # Attitude Error Term
        attitude_error_term = -1e0 * np.linalg.norm(
            self.chaser.quat_track_err(self.chaser.state[:4], self.qd)[1:]
        )

        # Control Effort Term
        control_effort_term = -1e-2 * np.sum(np.abs(action))

        # Stability Term
        stability_term = -1e-1 * np.dot(self.chaser.state[4:8], self.chaser.state[4:8])

        # Smoothness Term
        # smoothness_term = -0.001 * np.linalg.norm(np.gradient(angular_velocity))

        # Total Reward
        reward += (
            term_reward
            + attitude_error_term
            + control_effort_term
            + stability_term
            #  + smoothness_term
        )
        return float(reward)

    def _shape_reward(self):
        shaping = 0
        return shaping

    def _beyond_observational_space(self):
        if np.any(np.abs(self.chaser.state[4:8]) >= self.vmax):
            return True
        return False

    def _action_filter(self, action):
        if self.control == "Ml":
            action = action * Chaser.Tmax
        # action = np.clip(action, -Chaser.Tmax, Chaser.Tmax)

        if np.any(np.abs(action) > Chaser.Tmax):
            # remove in later versions
            print("action out of bounds")

        return np.array(action, dtype=np.float32)

    def _get_info(self):
        return {"qd": self.qd}


class Chaser:
    mass = 30
    initial_battery = 10
    # mu = 3.986004418 * 1e14
    # rt = 6.6 * 1e6
    Tmax = 1e-3  # RW400
    g = 9.81
    dt = 0.05  # lower this when planning to reach higher rotational speeds
    I = np.diag(np.array([8.33e-2, 1.08e-1, 4.17e-2], dtype=np.float32))
    invI = np.diag(
        np.array(
            [12.0048019207683, 9.25925925925926, 23.9808153477218], dtype=np.float32
        )
    )

    def __init__(self, step=1.0):
        # self.position = np.array([0,0,0,0])
        # self.velocity = np.array([0,0,0])
        self.state = np.zeros((7,), dtype=np.float32)
        self.invI = np.linalg.inv(self.I)

        self.step = self.dt if step is None else step

    def set_state(self, state):
        self.state = np.float32(state)
        return self.state

    @staticmethod  # make this static to compile with numba, and prevent cross slow down
    @jit(nopython=True)  # 1/30 of the time when compiled (mostly becaus of cross)
    def _Sat_Rotational_Dyn(t, y, trust):
        def vel_omega(w):
            return np.array(
                [
                    [0, -w[0], -w[1], -w[2]],
                    [w[0], 0, w[2], -w[1]],
                    [w[1], -w[2], 0, w[0]],
                    [w[2], w[1], -w[0], 0],
                ],
                dtype=np.float32,
            )

        dy = np.zeros(7, dtype=np.float32)
        q = y[0:4].astype(np.float32)
        w = y[4:8].astype(np.float32)

        torques = trust  # + noise

        ep = 1 - q @ q

        K = 0.1  # needed for stability at high velocities
        # w0 = np.hstack((0, w), dtype=np.float32)
        # dy[0:4] = 0.5 * Chaser.omega(w0) @ q + K * ep * q

        #! the inlining saves 1 second per episode

        dy[0:4] = 0.5 * (vel_omega(w) @ q) + K * ep * q
        # q = q / np.linalg.norm(q) to keep norm=1
        dy[4:] = invI @ (torques - np.cross(w, I @ w))

        # cross slows down the simulation a lot

        return dy

    @staticmethod
    @jit(nopython=True)
    def omega(q):
        mat = np.array(
            [
                [q[0], -q[1], -q[2], -q[3]],
                [q[1], q[0], q[3], -q[2]],
                [q[2], -q[3], q[0], q[1]],
                [q[3], q[2], -q[1], q[0]],
            ],
            dtype=np.float32,
        )
        return mat

    def rotate(self, trust):
        trust = np.clip(trust, -Chaser.Tmax, Chaser.Tmax)
        # self.state = self._rk4(
        # self._Sat_Rotational_Dyn, self.step, self.state, args=(trust,)
        # )
        sol = solve_ivp(
            Chaser._Sat_Rotational_Dyn,
            [0, self.step],
            self.state,
            args=(trust,),
            method="RK45",  # method="RK45",#method="LSODA",
        )
        # print(k)
        self.state = sol.y[:, -1]
        pass

    def _rk4(self, f, step, y0, args=()):
        N = int(self.step / self.dt)
        t = np.linspace(0, self.step, N + 1, dtype=np.float32)
        n = len(t)
        y = np.zeros((n, len(y0)), dtype=np.float32)
        y[0] = y0
        for i in range(n - 1):
            h = t[i + 1] - t[i]
            k1 = f(t[i], y[i], *args)
            k2 = f(t[i] + h / 2.0, y[i] + k1 * h / 2.0, *args)
            k3 = f(t[i] + h / 2.0, y[i] + k2 * h / 2.0, *args)
            k4 = f(t[i] + h, y[i] + k3 * h, *args)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y[-1]

    @staticmethod
    def quaternion_err_rate(q1, qd, w=np.array([0, 0, 0]), wd=np.array([0, 0, 0])):
        q_e = Chaser.quat_track_err(q1, qd)
        p = -q_e[1:4] #/ (1.5 - np.dot(q_e[1:4], q_e[1:4]))
        d = wd - w
        kp = q_e[0] * 1e-4 * invI
        kd = 1 * 1e-2
        u = kp @ p + kd * d
        return u

    @staticmethod
    def quat_track_err(q1, qd):
        q_inv = Chaser.quat_inv(qd)
        e = Chaser.omega(q_inv) @ q1  # equal to quaternion_multiply(q1,qd_inv)):
        # e_norm = np.linalg.norm(e)
        # if e_norm > 1.2 or e_norm < 0.8:
        # print("error_norm_error")
        # e = e / np.linalg.norm(e)
        return e

    @staticmethod
    @jit(nopython=True)
    def quat_inv(q):
        # shouldnt be needed the denominator is always 1
        den = np.dot(q, q)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32) / den


# i could use abr_control
@jit(nopython=True)
def eul_to_quat(eul):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qw,qx, qy, qz: The orientation in quaternion [x,y,z,w] format
    """
    yaw, pitch, roll = eul
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


@jit(nopython=True)
def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [roll, pitch, yaw]


def test_env():
    env = gym.make(
        "Satellite-rot-v0",
        render_mode=None,
        control="PID",
        matplotlib_backend=None,
    )
    print("env checked")
    term = False
    rwds = 0
    t_start = time.time()
    steps = 1000
    ep = 1
    for j in range(ep):
        print("ep", j)
        obs, info = env.reset()
        while True:
            action = Chaser.quaternion_err_rate(obs[0:4], info["qd"], w=obs[4:8])
            # action = env.action_space.sample()
            # action = np.array([0, 1, 0])
            obs, reward, term, trunc, info = env.step(action)
            # print(obs)

            # if (
            #     type(obs) != np.ndarray
            #     or type(info) != dict
            #     or type(action) != np.ndarray
            #     or type(reward) != float
            #     or type(term) != bool
            #     or type(trunc) != bool
            # ):
            #     print(type(obs))
            #     print(type(info))
            #     print(type(action))
            #     print(type(reward))
            #     print(type(term))
            #     print(type(trunc))
            #     print("type error")
            #     time.sleep(5)
            rwds += reward
            if term or trunc:
                print(term, trunc)
                x = env.render()
                # plt.imsave(f"next {j+1}.jpg",x)
                break
    print(time.time() - t_start)
    env.close()


def snakeviz_profiler(fun):
    import cProfile as profile
    import os

    prof = profile.Profile()
    prof.enable()
    fun()
    prof.disable()
    prof.dump_stats("test_env.prof")
    import os

    os.system("snakeviz test_env.prof")


def pstats_profiler(fun):
    import cProfile as profile
    import pstats

    prof = profile.Profile()
    prof.enable()
    fun()
    prof.disable()
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(100)  # top 100 rows

def scalene_profiler(fun):
    from scalene import scalene_profiler
    scalene_profiler.start()
    fun()
    scalene_profiler.stop()

    

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from gymnasium.envs.registration import register

    register(
        id="Satellite-rot-v0",
        entry_point="Satellite_rot:Satellite_rot",
        max_episode_steps=1000,
        reward_threshold=0.0,
    )
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_rot(), warn=True)
    print("env checked")
    #test_env()
    #snakeviz_profiler(test_env)
    
    #scalene_profiler(test_env)
