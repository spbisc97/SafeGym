import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.style as mplstyle

# mplstyle.use("fast")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint, solve_ivp
from numba import jit  # or try codonpy instead


# ? Maybe it's better to normalize the observation space?
# ? Maybe it's better to normalize the action space?
# ? Maybe it's better to use a discrete action space?
# ? Maybe it's better to use a continuous action space?
class Satellite_tra(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "observation_spaces": ["MlpPolicy", "MultiInputPolicy"],
        "control": [
            "Ml",
            "ModelPredictiveControl",
            "LQR",
            "PID",
            "Random",
            "Human",
        ],
        "action_spaces": ["continuous", "discrete"],
        "matplotlib_backend": [
            "TkAgg",
            "Qt5Agg",
            "Qt4Agg",
            "MacOSX",
            "WX",
            "gtk3agg",
            "Agg",  # non gui backend
        ],
    }

    def __init__(
        self,
        render_mode: str = None,
        observation_space: str = "MlpPolicy",
        action_space: str = "continuous",
        control: str = "Ml",
        matplotlib_backend: str = None,
        starting_state: np.ndarray = np.array([0, 0, 0, 0, 0, 0]),
        starting_noise: np.ndarray = np.array(
            [80, 80, 80, 0.0008, 0.0008, 0.0008]
        ),
    ):
        super(Satellite_tra, self).__init__()
        # define action and observation space being gymnasium.spaces
        assert control in self.metadata["control"]
        self.control = control

        assert action_space in self.metadata["action_spaces"]
        if action_space == "continuous":
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.int8
            )
        #   or action_space in self.metadata['action_spaces']
        #   could be  better to use normalized action space
        assert observation_space in self.metadata["observation_spaces"]
        if observation_space == "MlpPolicy":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        if observation_space == "MultiInputPolicy":
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                    ),
                    "fuel": spaces.Box(
                        low=0,
                        high=Chaser.initial_fuel_mass,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
        # or render_mode in self.metadata['render_modes']
        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode
        assert (matplotlib_backend in mpl.rcsetup.all_backends) or (
            matplotlib_backend is None
        )
        if matplotlib_backend:
            mpl.use(matplotlib_backend)
        self.subplots = None

        self.dmax = 40000
        self.vmax = 2000
        self.prev_shaping = None
        self.starting_state = starting_state
        self.starting_noise = starting_noise

    def step(self, action):
        reward = 0
        info = {}
        truncated = False
        filtered_action = self._action_filter(action)
        self.chaser.move(filtered_action)
        terminated = (
            self._beyond_observational_space()
        )  # or self.chaser.fuel_mass <= 0
        observation = self._get_observation()
        reward = self._reward_function(action, terminated)
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._remember(self.chaser.state, filtered_action, reward, info)
            if self.render_mode == "human":
                self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_shaping = None
        self.subplots = None
        self.chaser = Chaser()
        state = np.zeros((6,), dtype=np.float32)
        # state[random.randint(0,2)] = random.randint(-100, 100)
        # (np.random.rand(1, 3) - 0.5) * 0.01
        state = np.random.normal(self.starting_state, self.starting_noise)
        self.chaser.set_state(state)
        self.prev_shaping = self._shape_reward()
        info = {}
        observation = self._get_observation()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.statuses = np.array(self.chaser.state)
            self.actions = np.array([]).reshape(0, 3)
            self.rewards = np.array([]).reshape(0, 1)
            self.rewards_sum = np.array([]).reshape(0, 1)
            self.infos = np.array(info)
            self.fuels = np.array([self.chaser.fuel_mass])
            self.times = np.array([0])
        return observation, info

    def render(self):
        if not self.render_mode or (
            self.render_mode == "human" and (self.times[-1] - 1) % 2000 != 0
        ):
            return
        if self.subplots == None:
            fig, ax = plt.subplots(
                4, 3, figsize=(10, 10), layout="constrained"
            )
            fig.add_gridspec(hspace=30)
            lines = np.ma.zeros_like(ax)
            legend = np.array(
                [
                    ["x", "y", "z"],
                    ["vx", "vy", "vz"],
                    ["ux", "uy", "uz"],
                    ["reward", "reward_sum", "Fuel"],
                ]
            )
            limits = np.array(
                [
                    [[-100, 100], [-100, 100], [-100, 100]],
                    [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]],
                    [[-1, 1], [-1, 1], [-1, 1]],
                    [[-2, 0.3], [-100, 1], [-1, 1]],
                ]
            )
            limits[2, :] = limits[2, :] * Chaser.Tmax
            line_width = 0.6
            (lines[0, 0],) = ax[0, 0].plot(
                self.times, self.statuses[:, 0], linewidth=line_width
            )
            (lines[0, 1],) = ax[0, 1].plot(
                self.times, self.statuses[:, 1], linewidth=line_width
            )
            (lines[0, 2],) = ax[0, 2].plot(
                self.times, self.statuses[:, 2], linewidth=line_width
            )
            (lines[1, 0],) = ax[1, 0].plot(
                self.times, self.statuses[:, 3], linewidth=line_width
            )
            (lines[1, 1],) = ax[1, 1].plot(
                self.times, self.statuses[:, 4], linewidth=line_width
            )
            (lines[1, 2],) = ax[1, 2].plot(
                self.times, self.statuses[:, 5], linewidth=line_width
            )
            (lines[2, 0],) = ax[2, 0].plot(
                self.times[:-1], self.actions[:, 0], linewidth=line_width
            )
            (lines[2, 1],) = ax[2, 1].plot(
                self.times[:-1], self.actions[:, 1], linewidth=line_width
            )
            (lines[2, 2],) = ax[2, 2].plot(
                self.times[:-1], self.actions[:, 2], linewidth=line_width
            )
            (lines[3, 0],) = ax[3, 0].plot(
                self.times[:-1], self.rewards[:], linewidth=line_width
            )
            (lines[3, 1],) = ax[3, 1].plot(
                self.times[:-1], self.rewards_sum[:], linewidth=line_width
            )
            (lines[3, 2],) = ax[3, 2].plot(
                self.times, self.fuels, linewidth=line_width
            )

            for idx, x in np.ndenumerate(ax):
                # ax[idx[0], idx[1]].set_xlim(0, 100)
                ax[idx[0], idx[1]].set_ylim(limits[idx[0], idx[1], :])
                ax[idx[0], idx[1]].set_title(legend[idx[0], idx[1]])
                ax[idx[0], idx[1]].grid()
                if idx[0] != 3 and idx[1] != 0:
                    ax[idx[0], idx[1]].sharey(ax[idx[0], idx[1] - 1])
            plt.grid()
            if self.render_mode == "human":
                plt.ion()
                plt.show(block=False)
            else:
                plt.ioff()
            self.subplots = (fig, ax, lines)

        else:
            fig, ax, lines = self.subplots
            lines[0, 0].set_data(self.times, self.statuses[:, 0])
            lines[0, 1].set_data(self.times, self.statuses[:, 1])
            lines[0, 2].set_data(self.times, self.statuses[:, 2])
            lines[1, 0].set_data(self.times, self.statuses[:, 3])
            lines[1, 1].set_data(self.times, self.statuses[:, 4])
            lines[1, 2].set_data(self.times, self.statuses[:, 5])
            lines[2, 0].set_data(self.times[:-1], self.actions[:, 0])
            lines[2, 1].set_data(self.times[:-1], self.actions[:, 1])
            lines[2, 2].set_data(self.times[:-1], self.actions[:, 2])
            lines[3, 0].set_data(self.times[:-1], self.rewards[:])
            lines[3, 1].set_data(self.times[:-1], self.rewards_sum[:])
            lines[3, 2].set_data(self.times, self.fuels)

        # mplstyle.use("fast")

        for idx, x in np.ndenumerate(ax):
            ax[idx[0], idx[1]].relim(visible_only=True)
            ax[idx[0], idx[1]].autoscale(enable=True, axis="both", tight=False)

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
        # fig.show()

        return

    def close(self):
        super().close()
        if self.subplots:
            plt.close(self.subplots[0])
        return

    def _remember(self, sta, act, rew, inf={}):
        self.statuses = np.vstack((self.statuses, sta))
        self.actions = np.vstack((self.actions, act))
        self.rewards = np.vstack((self.rewards, rew))
        prev_rewards = (
            0 if len(self.rewards_sum) == 0 else self.rewards_sum[-1]
        )
        self.rewards_sum = np.vstack((self.rewards_sum, prev_rewards + rew))
        self.infos = np.vstack((self.infos, inf))
        self.times = np.vstack((self.times, self.times[-1] + self.chaser.step))
        self.fuels = np.vstack((self.fuels, self.chaser.fuel_mass))
        return

    def _get_observation(self):
        if isinstance(self.observation_space, spaces.Box):
            return np.concatenate(
                (
                    self.chaser.state,
                    np.asarray((self.chaser.fuel_mass,), dtype=np.float32),
                )
            )
        else:
            return {
                "state": self.chaser.state,
                "fuel": np.asarray((self.chaser.fuel_mass,), dtype=np.float32),
            }

    def _reward_function(self, action=0, terminated=False):
        reward = 0
        term_reward = 0 if not terminated else -100_000

        # Shaping Term (Not Used)
        # shaping = self._shape_reward()
        #
        # if self.prev_shaping is not None:
        # reward = shaping - self.prev_shaping
        # self.prev_shaping = shaping

        # Position Error Term
        log_position_error_term = np.log(
            np.linalg.norm(self.chaser.state[:3]) + 1e-1
        )

        # Control Effort Term
        control_effort_term = np.linalg.norm(action)

        reward += term_reward - log_position_error_term - control_effort_term

        return float(reward)

    def _shape_reward(self):
        shaping = 0
        return shaping

    def _beyond_observational_space(self):
        if np.any(np.abs(self.chaser.state[0:3]) >= self.dmax) or np.any(
            np.abs(self.chaser.state[3:6]) >= self.vmax
        ):
            return True
        return False

    def _action_filter(self, action):
        if self.control == "Ml":
            action = action * Chaser.Tmax
        # action = np.clip(action, -Chaser.Tmax, Chaser.Tmax)

        if np.any(np.abs(action) > Chaser.Tmax):
            # remove in later versions
            print("action out of bounds")

        return action


class Chaser:
    mass = 30
    initial_fuel_mass = 10
    mu = 3.986004418 * 1e14
    rt = 6.6 * 1e6
    Isp = 2250
    Tmax = 1.05e-3
    g = 9.81
    dt = 1

    def __init__(self, step=None):
        # self.position = np.array([0,0,0])
        # self.velocity = np.array([0,0,0])
        self.state = np.zeros((6,), dtype=np.float32)
        # self.inertia = np.array([8.33e-2,1.08e-1,4.17e-2])
        self.fuel_mass = self.initial_fuel_mass
        self.w = np.sqrt((self.mu / (self.rt**3)), dtype=np.float32)
        self.step = self.dt if step is None else step

    def set_state(self, state):
        self.state = np.float32(state)
        return self.state

    def _Sat_Translational_Dyn(self, t, y, trust):
        dy = np.zeros(
            6,
            dtype=np.float32,
        )
        torques = trust.copy()

        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]

        rc = ((self.rt + y[0]) ** 2 + y[1] ** 2 + y[2] ** 2) ** (1 / 2)
        eta = self.mu / (rc**3)
        total_mass = self.mass + self.fuel_mass

        dy[3] = (
            self.w**2 * y[0]
            + 2 * self.w * y[4]
            + self.mu / (self.rt**2)
            - (eta) * (self.rt + y[0])
            + torques[0] / total_mass
        )
        dy[4] = (
            -2 * self.w * y[3]
            + self.w**2 * y[1]
            - (eta) * y[1]
            + torques[1] / total_mass
        )
        dy[5] = -(eta) * y[2] + torques[2] / total_mass

        return dy

    def _Sat_Translational_Linear_Dyn(self, t, y, trust):
        dy = np.zeros(
            6,
            dtype=np.float32,
        )
        torques = trust.copy()

        dy[0] = y[3]
        dy[1] = y[4]
        dy[2] = y[5]

        rc = ((self.rt + y[0]) ** 2 + y[1] ** 2 + y[2] ** 2) ** (1 / 2)
        eta = self.mu / (rc**3)
        total_mass = self.mass + self.fuel_mass

        dy[3] = (
            self.w**2 * y[0]
            + 2 * self.w * y[4]
            + self.mu / (self.rt**2)
            - (eta) * (self.rt + y[0])
            + torques[0] / total_mass
        )
        dy[4] = (
            -2 * self.w * y[3]
            + self.w**2 * y[1]
            - (eta) * y[1]
            + torques[1] / total_mass
        )
        dy[5] = -(eta) * y[2] + torques[2] / total_mass

        return dy

    def move(self, trust):
        trust = np.clip(trust, -Chaser.Tmax, Chaser.Tmax)
        self.state = self._rk4(
            self._Sat_Translational_Dyn, self.step, self.state, args=(trust,)
        )

        #! there is no need to use a more precise integrator
        # sol = solve_ivp(
        # self._Sat_Translational_Dyn,
        # (0, self.step),
        # self.state,
        # args=(trust,),
        # method="RK45",
        # )
        # self.state = sol.y[:, -1].astype(np.float32)

        self.fuel_mass = (
            self.fuel_mass
            - np.sum(np.abs(trust)) / (self.g * self.Isp) * self.dt
        )
        pass

    def distance_to_target(self, target=np.zeros((3,), dtype=np.float32)):
        return np.linalg.norm(self.state[0:3] - target)

    def velocity_norm(self):
        return np.linalg.norm(self.state[3:6])

    # @jit
    def _rk4(self, f, t_len, y0, args=()):
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


def test_env(render_mode=None, control="PID", matplotlib_backend=None):
    env = gym.make(
        "Satellite-tra-v0",
        render_mode=render_mode,
        control=control,
        matplotlib_backend=matplotlib_backend,
    )
    K = np.array(
        [
            [
                4.37827913830195e-06,
                -4.55805760795127e-08,
                9.13530034828882e-19,
                0.000416745113065582,
                0.00185764340773104,
                3.96890134028692e-16,
            ],
            [
                6.92271669334668e-06,
                -3.55539079353476e-08,
                -1.47059095426734e-20,
                6.19214469243926e-05,
                0.00295066889892318,
                5.56097720476582e-17,
            ],
            [
                2.83885307788195e-17,
                6.23667199640684e-21,
                9.01572205276646e-10,
                4.02876583128802e-15,
                1.61053760602795e-14,
                0.000268562427265422,
            ],
        ]
    )
    import time

    ep = 1
    imgs = []
    for j in range(ep):
        obs, info = env.reset()
        while True:
            action = -np.dot(K, obs[0:6])
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                x = env.render()
                break
        imgs.append(x)
        print(f"next {j+1}")
    plt.imshow(np.concatenate(imgs, axis=1))
    plt.show()
    #
    env.close()


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from gymnasium.envs.registration import register

    check_env(Satellite_tra(), warn=True)
    print("env checked")
    register(
        id="Satellite-tra-v0",
        entry_point="Satellite_tra:Satellite_tra",
        max_episode_steps=300_000,
        reward_threshold=0.0,
    )
    test_env(render_mode="rgb_array", matplotlib_backend="Qt5Agg")

    # snakeviz_profiler(test_env)
