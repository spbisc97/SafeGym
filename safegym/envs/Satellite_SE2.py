import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import (
    Any,
    Optional,
    SupportsFloat,
)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.raise_window"] = False

INERTIA = 4.17e-2  # [kg*m^2]
INERTIA_INV = 23.9808  # 1/INERTIA
MASS = 30 + 10  # [kg]
MU = 3.986004418 * 10**14  # [m^3/s^2]
RT = 6.6 * 1e6  # [m]
NU = np.sqrt(MU / RT**3)
FMAX = 1.05e-3  # [N]
TMAX = 6e-3  # [Nm]
FTMAX = 6e-3  # just to clip with the same value for
STEP = 0.05  # [s]

VROT_MAX = 1  # [rad/s]
VTRANS_MAX = 10  # [m/s]

XY_MAX = 1000  # [m]

XY_PLOT_MAX = 1000  # [m]


STARTING_STATE = np.array([0, 1000, 0, 0, 0, 0, 0, 0], dtype=np.float32)

STARTING_NOISE = np.array([10, 10, np.pi / 2, 1e-5, 1e-5, 1e-4, 0, 0])

EULER_SPEEDUP = True


class Satellite_SE2(gym.Env):
    """
    A gym environment for simulating the motion of a satellite in 2D space.

    The Chaser is modeled as a rigid body with three degrees of freedom,
    and is controlled by action.
    The goal of the chaser is to approach and dock with a target spacecraft,
    while avoiding collisions and minimizing fuel usage.

    The observation of the system is represented by a 10-dimensional vector,
    consisting of:
    - the position of the chaser in R2
    - the direction vector of the chaser in R2
    - the orientation vector of the target in R2
    - the velocity of the chaser in R2
    - the angular velocity of the chaser in R1
    - the angular velocity of the target in R1

    The action space is either 2-dimensional or 3-dimensional, depending
    on whether the environment is underactuated or not.
    The action represents the thrust vector applied by the chaser spacecraft.

    The reward function is a combination of the negative logarithm of the
    distance between the chaser and target spacecrafts,
    and the squared magnitude of the thrust vector.
    """

    metadata = {
        "render_modes": ["human", "graph", "rgb_array", None],
        "render_fps": 1000,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        underactuated: Optional[bool] = True,
        starting_state: Optional[np.ndarray] = STARTING_STATE,
        starting_noise: Optional[np.ndarray] = STARTING_NOISE,
        unit_action_space: Optional[bool] = True,
        max_action=FTMAX,
    ):
        super(Satellite_SE2, self).__init__()
        assert isinstance(underactuated, bool)
        assert isinstance(starting_state, np.ndarray)
        assert isinstance(starting_noise, np.ndarray)
        assert isinstance(unit_action_space, bool)
        self.underactuated = underactuated
        self.unit_action_space = unit_action_space
        self.max_action = max_action
        self.starting_state = starting_state
        self.starting_noise = starting_noise
        self.render_mode = render_mode
        # Added fix and axes for rendering
        self.fig = None
        self.ax = None
        self.xylim = XY_PLOT_MAX
        # Added lists for storing historical data plotting
        self.state_history = []
        self.action_history = []
        self.time_step = 0

        self.build_action_space()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.steps_beyond_done = None
        self.chaser = self.Chaser(underactuated=underactuated)
        self.target = self.Target()
        self.reset()

        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        random_state = self.__generate_random_state()
        # set a not really stable initial traejctory
        chaser_stable_random = np.hstack(
            (
                0,
                random_state[1],
                random_state[2],
                random_state[1] / 2000,
                0,  # random_state[0] * 2e-3,
                random_state[5],
            )
        )

        self.chaser.reset(chaser_stable_random)
        for _ in range(np.random.randint(10, 1000)):
            self.chaser.step(0.5)  # roughly integration to move the object

        self.target.reset(random_state[6:8])
        observation = self.__get_observation()
        self.xylim = self.chaser.radius() * 2
        info = {}
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise f"{action!r} ({type(action)}) invalid"
        self.chaser.set_control(self.__action_filter(action))
        self.chaser.step()
        self.time_step += 1  # Increment the time_step at each step.
        terminated = False  # self.__termination()
        truncated = False
        reward = self.__reward()
        observation = self.__get_observation()
        info = {}
        if self.render_mode in ["human", "graph"]:
            self.render()  # Update the rendering after every action

        return observation, reward, terminated, truncated, info

    def render(self, observationlist: np.ndarray = None):
        """Render the environment."""
        if self.render_mode == "human":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots()
                self.ax.set_xlim(-self.xylim, self.xylim)
                self.ax.set_ylim(-self.xylim, self.xylim)
                self.ax.set_title("Satellite SE2 Environment")
                plt.ion()  # Turn on interactive mode

            if self.time_step % 100 != 0:
                return

            self.__draw_satellite()

            # Display the plot
            plt.pause(0.00001)  # Pause to show frame

        if self.render_mode == "graph":
            self.state_history.append(self.__get_state())
            self.action_history.append(self.chaser.get_control())

            if False:  # if np.size(self.action_history, 1) % 2 == 0:
                return

            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 15))
                plt.ion()  # Turn on interactive mode

            for ax in self.axs:
                ax.clear()

            # State history to numpy array
            state_history = np.array(self.state_history)

            # Plot positions
            self.axs[0].plot(state_history[:, 0], label="x position")
            self.axs[0].plot(state_history[:, 1], label="y position")
            self.axs[0].set_ylabel("Position")
            self.axs[0].legend(loc="upper right")

            # Plot velocities
            self.axs[1].plot(state_history[:, 2], label="x velocity")
            self.axs[1].plot(state_history[:, 3], label="y velocity")
            self.axs[1].set_ylabel("Velocity")
            self.axs[1].legend(loc="upper right")

            # Plot angle
            self.axs[2].plot(state_history[:, 4], label="Chaser Angle")
            self.axs[2].plot(state_history[:, 6], label="Target Angle")
            self.axs[2].set_ylabel("Angle (radians)")
            self.axs[2].legend(loc="upper right")

            # Plot angle speed
            self.axs[3].plot(
                state_history[:, 5], label="Chaser Angular Velocity"
            )
            self.axs[3].plot(
                state_history[:, 7], label="Target Angular Velocity"
            )
            self.axs[3].set_ylabel("Angular velocity (radians/s)")
            self.axs[3].legend(loc="upper right")

            # Plot actions
            self.axs[4].plot(
                self.action_history, label="Actions", linestyle="-", marker="o"
            )
            self.axs[4].set_ylabel("Action")
            self.axs[4].set_xlabel("Timesteps")
            self.axs[4].legend(loc="upper right")

            # Display timestamp relative to the plot time

            plt.pause(0.00001)  # Pause to show frame

        if self.render_mode == "rgb_array":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots()
                self.ax.set_xlim(-self.xylim, self.xylim)
                self.ax.set_ylim(-self.xylim, self.xylim)
                self.ax.set_title("Satellite SE2 Environment")

            self.__draw_satellite()
            self.fig.canvas.draw()
            img = np.array(
                self.fig.canvas.renderer.buffer_rgba()
            )  # Get the RGBA buffer
            data = img[:, :, :3]  # Drop the alpha channel to get RGB

            return data
        return

    def __draw_satellite(self):
        # Clear the axis for new drawings
        self.ax.clear()
        scale = self.xylim / 2

        # Draw the target (stationary at the center)
        self.ax.scatter(0, 0, color="red", s=100, label="Target")

        # Draw the target orientation
        theta = self.target.get_state()
        length = 5.0
        end_x = length * np.sin(theta[0])
        end_y = length * np.cos(theta[0])
        self.ax.plot(
            [0, end_x],
            [0, end_y],
            color="red",
        )

        # Draw the chaser satellite
        chaser_state = self.chaser.get_state()
        self.ax.scatter(
            chaser_state[0],
            chaser_state[1],
            color="blue",
            s=50,
            label="Chaser",
        )

        # Draw velocity vector of chaser
        end_x = chaser_state[0] + chaser_state[3] * 1e3
        end_y = chaser_state[1] + chaser_state[4] * 1e3
        self.ax.plot(
            [chaser_state[0], end_x],
            [chaser_state[1], end_y],
            color="green",
        )

        # Draw orientation of chaser as a short line
        length = 5.0
        self.ax.arrow(
            chaser_state[0],
            chaser_state[1],
            np.sin(chaser_state[2]),
            np.cos(chaser_state[2]),
            color="blue",
            head_width=1,
            head_length=2,
        )

        # Draw the chaser fore vector
        chaser_control = self.chaser.get_control() * 2e2 * scale
        if self.underactuated:
            length = chaser_control[0]
            end_x = chaser_state[0] + length * np.sin(chaser_state[2])
            end_y = chaser_state[1] + length * np.cos(chaser_state[2])
        else:
            length = np.linalg.norm(chaser_control[0:2])
            end_x = chaser_state[0] + chaser_control[0]
            end_y = chaser_state[1] + chaser_control[1]
        self.ax.plot(
            [chaser_state[0], end_x],
            [chaser_state[1], end_y],
            color="red",
        )

        # Legend, grid, and title
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlim(-self.xylim, self.xylim)
        self.ax.set_ylim(-self.xylim, self.xylim)
        self.ax.set_title("Satellite SE2 Environment")

        # Display timestamp relative to the plot time
        self.fig.suptitle(f"Time: {self.time_step*STEP:.2f} s", fontsize=12)

    def close(self) -> None:
        super().close()
        if self.fig:
            plt.close(self.fig)
        return

    def __get_observation(self) -> np.ndarray:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        observation = np.zeros((10,), dtype=np.float32)
        observation[0] = w[0]
        observation[1] = w[1]
        observation[2] = np.sin(w[2])
        observation[3] = np.cos(w[2])
        observation[4] = np.sin(theta[0])
        observation[5] = np.cos(theta[0])

        observation[6] = w[3]
        observation[7] = w[4]
        observation[8] = w[5]
        observation[9] = theta[1]

        return observation

    def __get_state(self) -> np.ndarray:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        state = np.zeros((10,), dtype=np.float32)
        state = np.concatenate((w, theta))
        return state

    def __generate_random_state(self):
        state = np.random.normal(self.starting_state, self.starting_noise)
        return state

    def __reward(self):
        reward = 0.5
        ch_radius = self.chaser.radius()
        ch_control = self.chaser.get_control()
        ch_speed = self.chaser.speed()

        reward += (
            (-np.log10(ch_radius + 1e-3))
            - (ch_control[0] ** 2 + ch_control[1] ** 2)
            + (-np.log10(ch_speed + 1e-3))
        )
        return reward

    def __action_filter(self, action):
        max_action = self.max_action
        action = action * max_action
        return action

    def __termination(self):
        if self.chaser.radius() < 1:
            return True
        else:
            return False

    def crash(self):
        if (self.chaser.radius() < 1) and (self.chaser.speed() > 1):
            return True
        else:
            return False

    def success(self):
        if self.chaser.radius() < 1 and self.chaser.speed() < 1:
            return True
        else:
            return False

    def build_action_space(self):
        if self.unit_action_space:
            max_action = 1
        else:
            max_action = self.max_action
        if self.underactuated:
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(2,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(3,), dtype=np.float32
            )

    class Chaser:
        """Chaser class for the satellite environment."""

        def __init__(self, step=STEP, state=None, underactuated=True):
            self.state = None

            if state is None:
                self.set_state()
            else:
                self.set_state(state)
            assert underactuated in [True, False]
            self.underactuated = underactuated

            if underactuated:
                self.__sat_dyn = self.__sat_dyn_underactuated

                self.control = np.zeros((2,), dtype=np.float32)
                self.control_space = 2  # avoid gym space check
            else:
                self.__sat_dyn = self.__sat_dyn_fullyactuated

                self.control = np.zeros((3,), dtype=np.float32)
                self.control_space = 3  # avoid gym space check
                # would be nice to have a check control space each time but
                # would slow down the code
            return

        def set_state(self, state=np.zeros((6,), dtype=np.float32)):
            self.state = np.float32(state)
            return

        def set_control(self, control: np.ndarray = None):
            self.control = np.float32(control)
            return

        def get_state(self):
            return self.state

        def get_control(self):
            return self.control

        def __sat_dyn_underactuated(
            self,
            t: SupportsFloat,
            w: np.ndarray,
            u: np.ndarray,
        ):
            # u = [fx, tau]

            dw = np.zeros((6,), dtype=np.float32)
            #  if symbolic:
            #    dw = [t, t, t, t, t, t]
            # else:
            dw[0] = w[3]
            dw[1] = w[4]
            dw[2] = w[5]
            dw[3] = (
                (3 * (NU**2) * w[0])
                + (2 * NU * w[4])
                + (np.sin(w[2]) * u[0] / MASS)
            )

            dw[4] = (-2 * NU * w[3]) + (np.cos(w[2]) * u[0] / MASS)
            dw[5] = INERTIA_INV * u[1]

            return dw

        def __sat_dyn_fullyactuated(
            self,
            t: SupportsFloat,
            w: np.ndarray,
            u: np.ndarray,
        ):
            # u = [fx, fu, tau]

            dw = np.zeros((6,), dtype=np.float32)
            # if symbolic:
            #     dw = [t, t, t, t, t, t]

            dw[0] = w[3]
            dw[1] = w[4]
            dw[2] = w[5]
            dw[3] = (3 * (NU**2) * w[0]) + (2 * NU * w[4]) + (u[0] / MASS)

            dw[4] = (-2 * NU * w[3]) + (u[1] / MASS)
            dw[5] = INERTIA_INV * u[2]
            return dw

        def step(self, ts=STEP):
            t = np.zeros((1,), dtype=np.float32)
            w = self.get_state()
            u = self.get_control()
            k1 = self.__sat_dyn(t, w, u)
            self.set_state(w + ts * (k1))

            if EULER_SPEEDUP:
                return self.state

            k2 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k1, u)
            k3 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k2, u)
            k4 = self.__sat_dyn(t + ts, w + ts * k3, u)
            self.set_state(w + ts * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            # Maybe np Vectorized calculation of k1, k2, k3, and k4
            # could speedup the code
            return self.state

        def reset(self, state=np.zeros((6,), dtype=np.float32)):
            self.set_state(state)
            if self.underactuated is True:
                self.set_control(np.zeros((2,), dtype=np.float32))
            else:
                self.set_control(np.zeros((3,), dtype=np.float32))
            return self.state

        def render(self):
            pass

        def radius(self):
            return np.linalg.norm(self.state[0:2])

        def speed(self):
            return np.linalg.norm(self.state[3:5])

        def symbolic_dynamics(self):
            pass

    class Target:
        # add dynamics later
        def __init__(self, step=STEP, state=np.zeros((2,), dtype=np.float32)):
            # state = [theta,theta_dot]
            self.state = state
            self.set_state(state)
            return

        def set_state(
            self, state: np.array = np.zeros((2,), dtype=np.float32)
        ):
            self.state = np.float32(state)
            return

        def get_state(self):
            return self.state

        def reset(self, state=np.zeros((2,), dtype=np.float32)):
            self.set_state(state)
            return self.state

        def __remember(self):
            pass

        # add dynamics later when speed is needed


def _test():
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_SE2(), warn=True)
    print("env checked")


def _test2(underactuated=True):
    env = Satellite_SE2(
        underactuated=underactuated,
        render_mode="human",
        starting_state=np.array([0, 10, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        starting_noise=np.zeros((8,)),
    )
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    actions = []
    print(env.action_space.sample())
    for _ in range(10000):
        actions.append(env.action_space.sample())
        observation, reward, term, trunc, info = env.step(actions[-1])
        render = env.render()
        observations.append(observation)
        rewards.append(reward)
    env.close()
    plt.plot(observations)
    plt.show()
    plt.plot(actions)
    plt.show()
    plt.plot(rewards)
    plt.show()

    return


def _test3(underactuated=True):
    from stable_baselines3 import DDPG, PPO
    from moviepy.editor import ImageSequenceClip

    register(
        id="Satellite_SE2-v0",
        entry_point="Satellite_SE2:Satellite_SE2",
        max_episode_steps=200000,
        reward_threshold=0.0,
    )

    starting_state = np.array([0, 10, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    starting_noise = np.zeros((8,))
    model = PPO.load("Satellite_SE2")
    env = gym.make(
        "Satellite_SE2-v0",
        underactuated=underactuated,
        starting_state=starting_state,
        starting_noise=starting_noise,
    )
    model.set_env(env)
    # model = PPO("MlpPolicy", env=env, verbose=1)
    for _ in range(4):
        model.learn(total_timesteps=200000)
        model.save("Satellite_SE2")

    observation, info = env.reset()
    observations = [observation]
    rewards = []
    actions = []

    env.reset()

    env = gym.make(
        "Satellite_SE2-v0",
        underactuated=underactuated,
        starting_state=starting_state,
        starting_noise=starting_noise,
        render_mode="rgb_array",
    )
    env.reset()
    frames = [env.render()]
    for _ in range(100000):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, trun, term, info = env.step(action)
        if _ % 50 == 0:
            frames.append(env.render())
        observations.append(observation)
        rewards.append(reward)
        actions.append(action)
    env.close()
    clip = ImageSequenceClip(frames, fps=100)
    save_path = "./video.mp4"
    clip.write_videofile(save_path, fps=100)

    plt.plot(observations)
    plt.show()
    plt.plot(actions)
    plt.show()
    plt.plot(rewards)
    plt.show()


def _test4():
    # check just the dynamics
    starting_state = np.zeros((8,))
    starting_state[1] = 30
    env = Satellite_SE2(
        render_mode=None,
        starting_noise=np.zeros((8,)),
        starting_state=starting_state,
    )
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    action = np.array([0, 0], dtype=np.float32)
    for _ in range(1000000):
        observation, reward, term, trunc, info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
    env.close()
    observations = np.array(observations)
    print(observations.shape)
    print(observations[:, 0])
    plt.plot(observations[:, 0], observations[:, 1])
    plt.show()
    plt.plot(rewards)
    plt.show()


def _test5():
    from gymnasium.experimental.wrappers import HumanRenderingV0, RecordVideoV0
    from moviepy.editor import ImageSequenceClip

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

    # check just the dynamicsstarting_state = np.zeros((8,))
    starting_state = np.zeros((8,))
    starting_state[1] = 30

    env = Satellite_SE2(
        underactuated=False,
        render_mode="rgb_array",
        max_action=1,  # set to 1 for nomal control
        starting_state=starting_state,
        starting_noise=np.zeros((8,)),
    )
    # env = HumanRenderingV0(env)
    # env = RecordVideoV0(env, video_folder=".", video_length=0)
    observation, info = env.reset()
    observations = [observation]
    frames = []
    rewards = []
    action = np.array([0, 0, 0], dtype=np.float32)
    env.action_space.sample()
    print(env.action_space.sample())

    for _ in range(100000):
        action = -k @ env.chaser.get_state()

        if np.linalg.norm(action) > 1:
            action = action / np.norm(action)
        observation, reward, term, trunc, info = env.step(action)
        if _ % 50 == 0:
            frames.append(env.render())
        observations.append(observation)
        rewards.append(reward)

    clip = ImageSequenceClip(frames, fps=100)
    save_path = "./video.mp4"
    clip.write_videofile(save_path, fps=100)

    env.close()

    observations = np.array(observations)
    print(observations.shape)
    plt.plot(observations[:, 0], observations[:, 1])
    plt.show()


def _test6():
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
    starting_state = np.zeros((8,))
    starting_state[1] = 30
    env = Satellite_SE2(
        underactuated=False,
        render_mode=None,
        max_action=1,  # set to 1 for nomal control
        starting_state=starting_state,
        starting_noise=np.zeros((8,)),
    )
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    action = np.array([0, 0, 0], dtype=np.float32)
    env.action_space.sample()
    print(env.action_space.sample())
    for _ in range(10000):
        action = -k @ env.chaser.get_state()
        if np.linalg.norm(action) > 1:
            action = action / np.norm(action)
        observation, reward, term, trunc, info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
    env.close()


def _scalene_profiler():
    from scalene import scalene_profiler

    scalene_profiler.start()
    _test6()
    scalene_profiler.stop()


if __name__ == "__main__":
    from gymnasium.envs.registration import register

    _test3()
