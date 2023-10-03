import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import jit


# ? Maybe it's better to normalize the observation space?
# ? Maybe it's better to normalize the action space?
# ? Maybe it's better to use a discrete action space?
# ? Maybe it's better to use a continuous action space?
class Satellite_base(gym.Env):
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
    }

    def __init__(
        self,
        render_mode=None,
        observation_space="MlpPolicy",
        action_space="continuous",
        control="Ml",
    ):
        super(Satellite_base, self).__init__()
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
        else:
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

        self.dmax = 20000
        self.vmax = 20000
        self.prev_shaping = None

    def step(self, action):
        reward = 0
        info = {}
        truncated = False
        self.chaser.move(self._action_filter(action))
        terminated = (
            self._beyond_observational_space()
        )  # or self.chaser.fuel_mass <= 0
        observation = self._get_observation()
        reward = self._reward_function()
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.chaser = Chaser()
        state = np.zeros((6,), dtype=np.float32)
        # state[random.randint(0,2)] = random.randint(-100, 100)
        state[0:3] = (np.random.rand(1, 3) - 0.5) * 10
        state[3:] = (np.random.rand(1, 3) - 0.5) * 0.1
        self.chaser.set_state(state)
        self.prev_shaping = self._shape_reward()
        info = {}
        return self._get_observation(), info

    def render(self):
        pass

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

    def _reward_function(self):
        reward = 0
        # distance_reward = prev_distance - distance
        # fuel_reward = prev_fuel - fuel
        shaping = self._shape_reward()

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # reward -=(self.chaser.distance_to_target()+0.1)
        reward += 10 / (self.chaser.distance_to_target() + 0.01)
        reward -= (self.dmax / 100) / (
            1.74 * self.dmax - self.chaser.distance_to_target() + 0.01
        )
        # something similar to -tanh(distance)
        return float(reward)

    def _shape_reward(self):
        shaping = (
            -self.chaser.distance_to_target()  # +prev distance
            # + 10/(self.chaser.velocity_norm()+1)
            + self.chaser.fuel_mass  # - prev fuel
        )
        return shaping

    def _beyond_observational_space(self):
        if np.any(np.abs(self.chaser.state[:3]) >= self.dmax) or np.any(
            np.abs(self.chaser.state[3:]) >= self.vmax
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
    Tmax = 1.05e-2
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

    def _Sat_Translational_Dyn(self, y, t, trust):
        dy = np.zeros(
            6,
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

    def _Sat_Translational_Linear_Dyn(self, y, t, trust):
        dy = np.zeros(
            6,
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
            self._Sat_Translational_Dyn, self.state, self.dt, args=(trust,)
        )
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
    def _rk4(self, f, y0, t, args=()):
        N = int(self.step / self.dt)
        t = np.linspace(0, self.step, N + 1, dtype=np.float32)
        n = len(t)
        y = np.zeros((n, len(y0)), dtype=np.float32)
        y[0] = y0
        for i in range(n - 1):
            h = t[i + 1] - t[i]
            k1 = f(y[i], t[i], *args)
            k2 = f(y[i] + k1 * h / 2.0, t[i] + h / 2.0, *args)
            k3 = f(y[i] + k2 * h / 2.0, t[i] + h / 2.0, *args)
            k4 = f(y[i] + k3 * h, t[i] + h, *args)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y[-1]


if __name__ == "__main__":
    # Q_main()
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_base(), warn=True)
