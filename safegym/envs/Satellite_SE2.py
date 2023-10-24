import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Optional, SupportsFloat, Tuple
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.patches import Circle
import warnings

# matplotlib.use(
#     "TKAgg"
# )  # Or 'Qt5Cairo' or 'Qt4Cairo' if you're using PyQt/PySide


matplotlib.rcParams["patch.force_edgecolor"] = True
matplotlib.rcParams["figure.raise_window"] = False
mplstyle.use("fast")
matplotlib.rcParams["path.simplify"] = True
# # pltstyle = ["ggplot"]


# pltstyle = [
#     style
#     for style in mplstyle.available
#     if (("seaborn" and "paper") in style)
#     or (("seaborn" and "colorblind") in style)
#     or (("seaborn" and "whitegrid") in style)
# ]
# pltstyle.append("fast")
@staticmethod
def generate_stable_trajectory(
    radius: np.float32, theta: np.float32, theta_dot: np.float32
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    state = np.zeros((6,), dtype=np.float32)
    state[0] = 0  # x
    state[1] = radius  # y
    state[2] = theta  # theta
    state[3] = radius / 2000  # vx
    state[4] = 0  # vy
    state[5] = theta_dot  # omega
    return state


# mplstyle.use(pltstyle)


INERTIA = 4.17e-2  # [kg*m^2]
INERTIA_INV = 23.9808  # 1/INERTIA
MASS = 30 + 10  # [kg]
MU = 3.986004418 * 10**14  # [m^3/s^2]
RT = 6.6 * 1e6  # [m]
NU = np.sqrt(MU / RT**3)
FMAX = 1.05e-3  # [N]
TMAX: np.float32 = np.float32(1e-4)  # [Nm]
FTMAX: np.float32 = np.float32(1e-3)  # just to clip with the same value for
STEP: np.float32 = np.float32(0.05)  # [s]

VROT_MAX = np.float32(2 * np.pi)  # [rad/s]
VTRANS_MAX = np.float32(50)  # [m/s]

XY_MAX = np.float32(10)  # [m]

XY_PLOT_MAX = np.float32(10)  # [m]

y0: np.float32 = np.float32(5)  # [m]
# STARTING_STATE=
radius: np.float32 = y0  # [m],
theta: np.float32 = np.float32(0)  # [rad],
theta_dot: np.float32 = np.float32(0)  # [rad/s],
phi: np.float32 = np.float32(0)  # [rad]
phi_dot: np.float32 = np.float32(0)  # [rad/s]


STARTING_STATE_CHASER = generate_stable_trajectory(
    radius=y0, theta=theta, theta_dot=theta_dot
)
STARTING_STATE_TARGET = np.array([phi, phi_dot], dtype=np.float32)

STARTING_STATE = np.concatenate(
    [STARTING_STATE_CHASER, STARTING_STATE_TARGET], dtype=np.float32
)
INITIAL_INTEGRATION_STEPS = np.array(object=[0, 0], dtype=np.int32)

STARTING_NOISE = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.float32,
)
# REWARD_WEIGHTS = distance_decrease,distance,action,speed,angle_speed
REWARD_WEIGHTS = np.array([100, 0.1, 0.3, 1, 10], dtype=np.float32)
EULER_SPEEDUP = True


class Satellite_SE2(gym.Env):  # type: ignore
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
        "render_modes": [
            "human",
            "graph",
            "rgb_array",
            "rgb_array_graph",
        ],
        "render_fps": 10,
        "render_steps": 200,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        underactuated: Optional[bool] = True,
        starting_state: Optional[
            np.ndarray[Tuple[int], np.dtype[np.float32]]
        ] = STARTING_STATE,
        starting_noise: Optional[
            np.ndarray[Tuple[int], np.dtype[np.float32]]
        ] = STARTING_NOISE,
        unit_action_space: Optional[bool] = True,
        max_action: np.float32 = FTMAX,
        step: np.float32 = STEP,
        xy_max: np.float32 = XY_MAX,
        xy_plot_max: np.float32 = XY_PLOT_MAX,
        vtrans_max: np.float32 = VTRANS_MAX,
        vrot_max: np.float32 = VROT_MAX,
        normalized_obs: bool = True,
        unconstrained: bool = False,
        initial_integration_steps: np.ndarray[
            tuple[int], np.dtype[np.int32]
        ] = INITIAL_INTEGRATION_STEPS,
        reward_weights: np.ndarray[
            tuple[int], np.dtype[np.float32]
        ] = REWARD_WEIGHTS,
        doubleintegrator: bool = False,
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
        self.unconstrained = unconstrained
        self.is_success = False
        self.initial_integration_steps = initial_integration_steps
        self.reward_weights = reward_weights
        assert (
            render_mode in self.metadata["render_modes"] or render_mode is None
        )
        self.render_mode = render_mode
        self.__step = step
        # Added fix and axes for rendering
        self.fig = None
        self.ax = None
        self.xy_plot_lim = xy_plot_max
        self.xy_max = xy_max
        self.vrot_max = vrot_max
        self.vtrans_max = vtrans_max
        # Added lists for storing historical data plotting
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.separate_reward_history = []
        self.time_step = -1

        self.build_action_space()
        self.normalized_obs = normalized_obs
        self.build_observation_space()  # must be after self.normalized

        self.steps_beyond_done = None
        self.chaser = self.Chaser(
            underactuated=underactuated,
            step=self.__step,
            doubleintegrator=doubleintegrator,
        )
        self.target = self.Target()
        self.reset()

        return

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        random_state = np.random.normal(
            self.starting_state, self.starting_noise
        )

        chaser_starting_state = np.array(random_state[0:6], dtype=np.float32)
        target_starting_state = np.array(random_state[6:8], dtype=np.float32)

        self.chaser.reset(chaser_starting_state)
        if (
            self.initial_integration_steps[0]
            == self.initial_integration_steps[1]
        ):
            self.initial_integration_steps[1] = (
                self.initial_integration_steps[0] + 1
            )
        action_free_steps = np.random.randint(
            self.initial_integration_steps[0],
            self.initial_integration_steps[1],
        )
        for i in range(0, action_free_steps):
            self.chaser.step()
        self.target.reset(target_starting_state)
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.separate_reward_history = []
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        self.is_success = False
        self.is_unsafe = False
        observation = self.__get_observation()
        self.xy_plot_lim = self.chaser.radius() * 2
        info = {
            "is_success": self.is_success,
            "time_step": self.time_step,
            "is_unsafe": self.is_unsafe,
        }
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, np.float32, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise Exception(f"{action!r} invalid")
        assert self.time_step != -1, "Must reset environment first"
        self.chaser.set_control(self.__action_filter(action))
        self.chaser.step()
        observation = self.__get_observation()
        reward = self._reward_function()  # Compute the reward
        self.time_step += 1  # Increment the time_step at each step.
        info = {
            "is_success": self.is_success,
            "time_step": self.time_step,
            "is_unsafe": self.is_unsafe,
        }
        # "action": action,
        # "observation": observation,
        # "reward": reward,}
        if self.render_mode in ["human"]:
            self.render()  # Update the rendering after every action
        self.action_history.append(self.chaser.get_control())
        self.state_history.append(self.__get_state())
        self.reward_history.append(reward)

        if not self.terminated:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                warnings.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. "
                    "You should always call 'reset()' once you "
                    "receive 'done = True' -- any further steps "
                    "are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = np.float32(0.0)

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(layout="constrained")
                self.ax.set_xlim(-self.xy_plot_lim, self.xy_plot_lim)
                self.ax.set_ylim(-self.xy_plot_lim, self.xy_plot_lim)
                self.ax.set_title("Satellite SE2 Environment")
                plt.ion()  # Turn on interactive mode

            if (
                self.time_step % self.metadata["render_steps"] != 0
            ):  # Plot with fps
                return

            self.__draw_satellite()

            # Display the plot
            plt.pause(0.00001)  # Pause to show frame

        if self.render_mode == "graph":
            # if False:  # if np.size(self.action_history, 1) % 2 == 0:
            # return

            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(
                    6, 1, figsize=(10, 15), layout="constrained"
                )
                plt.ion()  # Turn on interactive mode

            self.__draw_satellite_graph()

            plt.pause(0.00001)  # Pause to show frame

        if self.render_mode == "rgb_array":
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(layout="constrained")
                self.ax.set_xlim(-self.xy_plot_lim, self.xy_plot_lim)
                self.ax.set_ylim(-self.xy_plot_lim, self.xy_plot_lim)
                self.ax.set_title("Satellite SE2 Environment")

            self.__draw_satellite()
            self.fig.canvas.draw()
            img = np.array(
                self.fig.canvas.renderer.buffer_rgba()  # type: ignore
            )  # Get the RGBA buffer
            data = img[:, :, :3]  # Drop the alpha channel to get RGB

            return data
        if self.render_mode == "rgb_array_graph":
            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(
                    6, 1, figsize=(10, 15), layout="constrained"
                )

            self.__draw_satellite_graph()
            self.fig.canvas.draw()
            img = np.array(
                self.fig.canvas.renderer.buffer_rgba()  # type: ignore
            )
            data = img[:, :, :3]
            return data

        return

    def __draw_satellite_graph(self):
        if self.fig is None or self.axs is None:
            return
        for ax in self.axs:
            ax.clear()
        # State history to numpy array
        state_history = np.array(self.state_history)
        if self.time_step < 1:
            return
        # Plot positions
        self.axs[0].plot(state_history[:, 0], label="x position")
        self.axs[0].plot(state_history[:, 1], label="y position")
        self.axs[0].set_ylabel("Position")
        self.axs[0].legend(loc="upper right")
        # Plot velocities
        self.axs[1].plot(state_history[:, 3], label="x velocity")
        self.axs[1].plot(state_history[:, 4], label="y velocity")
        self.axs[1].set_ylabel("Velocity")
        self.axs[1].legend(loc="upper right")
        # Plot angle
        self.axs[2].plot(state_history[:, 2], label="Chaser Angle")
        self.axs[2].plot(state_history[:, 6], label="Target Angle")
        self.axs[2].set_ylabel("Angle (rad)")
        self.axs[2].legend(loc="upper right")
        # Plot angle speed
        self.axs[3].plot(state_history[:, 5], label="Chaser Angular Velocity")
        self.axs[3].plot(state_history[:, 7], label="Target Angular Velocity")
        self.axs[3].set_ylabel("Angular velocity (rad/s)")
        self.axs[3].legend(loc="upper right")
        # Plot actions
        self.axs[4].plot(
            self.action_history,
            label="Actions",
            linestyle=None,
            marker=".",
            markersize=1,
        )
        self.axs[4].set_ylabel("Action")
        self.axs[4].set_xlabel("Timesteps")
        self.axs[4].legend(loc="upper right")

        self.axs[5].plot(
            self.reward_history,
            label=f"Reward{sum(self.reward_history):.2e}",
            linestyle=None,
            marker=".",
            markersize=1,
        )
        self.axs[5].set_yscale("symlog", linthresh=1e-3, linscale=1e-3)
        self.axs[5].set_ylabel("Reward")
        self.axs[5].set_xlabel("Timesteps")
        self.axs[5].legend(loc="upper right")
        # Legend, grid, and title

        # Display timestamp relative to the plot time
        return

    def __draw_satellite(self):
        if self.fig is None or self.ax is None:
            return
        # Clear the axis for new drawings
        self.ax.clear()
        scale = self.xy_plot_lim / 10

        # Draw the target (stationary at the center)
        self.ax.scatter(0, 0, color="red", s=100, label="Target")

        # Draw the target orientation
        theta = self.target.get_state()
        length = scale / 3
        end_x = length * np.cos(theta[0])
        end_y = length * np.sin(theta[0])
        self.ax.arrow(
            0,
            0,
            end_x,
            end_y,
            head_width=scale / 3,
            head_length=scale / 2,
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
        end_x = chaser_state[0] + chaser_state[3] * (scale * 15e1)
        end_y = chaser_state[1] + chaser_state[4] * (scale * 15e1)
        self.ax.plot(
            [chaser_state[0], end_x],
            [chaser_state[1], end_y],
            color="green",
            label=f"V:{np.linalg.norm(chaser_state[3:5]):.2e}"
            + "\n"
            + f"Vrot:{(chaser_state[5]):.2e}",
        )

        # Draw orientation of chaser as a short line
        length = scale / 4
        self.ax.arrow(
            chaser_state[0],
            chaser_state[1],
            length * np.cos(chaser_state[2]),
            length * np.sin(chaser_state[2]),
            color="blue",
            head_width=scale / 3,
            head_length=scale / 3,
        )

        # Draw the chaser force vector
        chaser_control = self.chaser.get_control()
        if self.underactuated:
            length = chaser_control[0] * 3e4 * scale
            end_x = chaser_state[0] + length * np.cos(chaser_state[2])
            end_y = chaser_state[1] + length * np.sin(chaser_state[2])
            self.ax.plot(
                [chaser_state[0], end_x],
                [chaser_state[1], end_y],
                color="red",
                label=f"F:{(chaser_control[0]):.2e}"
                + "\n"
                + f"T:{(chaser_control[1]):.2e}",
            )
        else:
            end_x = chaser_state[0] + 6e2 * scale * chaser_control[0]
            end_y = chaser_state[1] + 6e2 * scale * chaser_control[1]
            self.ax.plot(
                [chaser_state[0], end_x],
                [chaser_state[1], chaser_state[1]],
                color="red",
            )
            self.ax.plot(
                [chaser_state[0], chaser_state[0]],
                [chaser_state[1], end_y],
                color="red",
                label=f"Fx:{(chaser_control[0]):.2e}"
                + "\n"
                + f"Fy:{(chaser_control[1]):.2e}"
                + "\n"
                + f"T:{(chaser_control[2]):.2e}",
            )
        CIRCLE = Circle(  # create patch outside of the func gives errors?!
            (0, 0), 1, color="green", fill=False, linewidth=1, alpha=0.5
        )
        self.ax.add_patch(CIRCLE)
        self.ax.plot(
            [], color="green", label=f"Rew: {self._reward_function():.2e}"
        )

        # Legend, grid, and title
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlim(-self.xy_plot_lim, self.xy_plot_lim)
        self.ax.set_ylim(-self.xy_plot_lim, self.xy_plot_lim)
        self.ax.set_title("Satellite SE2 Environment")

        # Display timestamp relative to the plot time
        self.fig.suptitle(f"Time: {self.time_step*STEP:.2f} s", fontsize=12)

    def close(self) -> None:
        super().close()
        if self.fig:
            plt.close(self.fig)
        return

    def __get_normalized_observation(
        self,
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        observation = np.zeros((10,), dtype=np.float32)
        observation[0] = w[0] / self.xy_max
        observation[1] = w[1] / self.xy_max
        observation[2] = np.cos(w[2])  # already btw -1 and 1
        observation[3] = np.sin(w[2])  # already btw -1 and 1
        observation[4] = np.cos(theta[0])  # already btw -1 and 1
        observation[5] = np.sin(theta[0])  # already btw -1 and 1

        observation[6] = w[3] / self.vtrans_max
        observation[7] = w[4] / self.vtrans_max
        observation[8] = w[5] / self.vrot_max
        observation[9] = theta[1] / self.vrot_max
        if any((np.abs(observation)) > 1):
            warnings.warn("Observation is not normalized")
            print(observation)
        # i could just add here term or trunc conditions

        return observation

    def __get_absolute_observation(
        self,
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        observation = np.zeros((10,), dtype=np.float32)
        observation[0] = w[0]
        observation[1] = w[1]
        observation[2] = np.cos(w[2])
        observation[3] = np.sin(w[2])
        observation[4] = np.cos(theta[0])
        observation[5] = np.sin(theta[0])

        observation[6] = w[3]
        observation[7] = w[4]
        observation[8] = w[5]
        observation[9] = theta[1]
        return observation

    def __get_state(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        w = self.chaser.get_state()
        theta = self.target.get_state()
        state = np.concatenate((w, theta))
        return state

    def _reward_function(self) -> np.float32:
        ch_radius = self.chaser.radius()
        ch_control = self.chaser.get_control()
        ch_speed = self.chaser.speed()
        ch_state = self.chaser.get_state()

        if self.termination():
            self.terminated = True
            if self.unconstrained or self.success():
                self.is_success = True
                return np.float32(
                    1000
                    + (1_000_000 / ((self.time_step + 1) * self.__step))
                    * (1 / (ch_speed + 1e-4))
                )
            if self.crash():
                self.is_unsafe = True
                return np.float32(-5000 * (ch_speed + 1e-4))

        if self.out_of_bounds():
            self.terminated = True
            return np.float32(-1000)

        # for shaping i could add a reward for the radius lowering

        # Encourage the agent to minimize the distance
        if self.time_step != 0:
            reward_decreased_distance = (
                np.linalg.norm(self.state_history[-1][0:2])
            ) - ch_radius
        else:
            reward_decreased_distance = 0
        reward_decreased_distance = reward_decreased_distance / self.xy_max

        normalized_distance = ch_radius / self.xy_max
        reward_distance = -normalized_distance

        # Encourage the agent to minimize control effort, with normalization
        reward_control = -np.linalg.norm(ch_control / self.max_action)

        # Encourage the agent to maintain low speed
        reward_speed = -ch_speed / (VTRANS_MAX)

        # penalize high angular velocity
        reward_angular_velocity = -np.abs(ch_state[5]) / (VROT_MAX)

        # i could add a reward for the angle between the chaser and the target
        # i coudl add reward_weights to init function
        # give steps rewards
        extra_reward = 0
        if normalized_distance < 0.7:
            extra_reward += 1
            if normalized_distance < 0.4:
                extra_reward += 1
                if normalized_distance < 0.2:
                    extra_reward += 1
        extra_reward_weight = 0.5
        extra_reward = extra_reward * extra_reward_weight

        # Combine
        reward = (
            (self.reward_weights[0] * reward_decreased_distance)
            + (self.reward_weights[1] * reward_distance)
            + (self.reward_weights[2] * reward_control)
            + (self.reward_weights[3] * reward_speed)
            + (self.reward_weights[4] * reward_angular_velocity)
            + extra_reward
        )
        # print(
        #     reward_decrease_distance,
        #     reward_distance,
        #     reward_control,
        #     reward_speed,
        #     reward_angular_velocity,
        # )
        # print(
        #     (self.reward_weights[0] * reward_decrease_distance),
        #     (self.reward_weights[1] * reward_distance),
        #     (self.reward_weights[2] * reward_control),
        #     (self.reward_weights[3] * reward_speed),
        #     (self.reward_weights[4] * reward_angular_velocity),
        # )

        self.separate_reward_history.append(
            [
                (self.reward_weights[0] * reward_decreased_distance),
                (self.reward_weights[1] * reward_distance),
                (self.reward_weights[2] * reward_control),
                (self.reward_weights[3] * reward_speed),
                (self.reward_weights[4] * reward_angular_velocity),
                extra_reward,
            ]
        )
        return np.float32(reward)

    def _reward_shaping(self):
        pass

    def out_of_bounds(self):
        chaser_state = self.chaser.get_state()
        if np.abs(chaser_state[0]) > self.xy_max:
            self.is_unsafe = True
            return True
        if np.abs(chaser_state[1]) > self.xy_max:
            self.is_unsafe = True
            return True

        if np.abs(chaser_state[3]) > self.vtrans_max:
            return True
        if np.abs(chaser_state[4]) > self.vtrans_max:
            return True
        if np.abs(chaser_state[5]) > self.vrot_max:
            return True

        return False

    def __action_filter(self, action):
        max_action = self.max_action
        action = action * max_action
        return action

    """Terminated (bool) 
    – Whether the agent reaches the terminal state (as defined under the MDP of 
    the task) which can be positive or negative. An example is reaching the 
    goal state or moving into the lava from the Sutton and Barton, Gridworld. 
    If true, the user needs to call reset().

    Truncated (bool) 
    – Whether the truncation condition outside the scope of the MDP is 
    satisfied. Typically, this is a timelimit, but could also be used to 
    indicate an agent physically going out of bounds. 
    Can be used to end the episode prematurely before a terminal state is 
    reached. If true, the user needs to call reset().
    """

    def termination(self):
        if self.chaser.radius() < 1:
            return True
        else:
            return False

    def crash(self):
        if self.chaser.radius() < 1:  # just keep the double check
            if self.chaser.speed() > 0.01:  # 0.01m/s = 1cm/s
                return True
        return False

    def success(self):
        if self.chaser.radius() < 1:  # just keep the double check
            if self.chaser.speed() < 0.01:  # 0.01m/s = 1cm/s
                return True
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

    def build_observation_space(self):
        if self.normalized_obs == True:
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(10,), dtype=np.float32
            )
            self.__get_observation = self.__get_normalized_observation
        else:
            abs_lim = np.array(
                [
                    self.xy_max,
                    self.xy_max,
                    1,
                    1,
                    1,
                    1,
                    self.vtrans_max,
                    self.vtrans_max,
                    self.vrot_max,
                    self.vrot_max,
                ],
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-abs_lim, high=abs_lim, shape=(10,), dtype=np.float32
            )
            self.__get_observation = self.__get_absolute_observation

    @staticmethod
    def generate_stable_trajectory(
        radius: np.float32, theta: np.float32, theta_dot: float
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        state = np.zeros((6,), dtype=np.float32)
        state[0] = 0  # x
        state[1] = radius  # y
        state[2] = theta  # theta
        state[3] = radius / 2000  # vx
        state[4] = 0  # vy
        state[5] = theta_dot  # omega
        return state

    class Chaser:
        """Chaser class for the satellite environment."""

        def __init__(
            self,
            step: np.float32,
            state: Optional[
                np.ndarray[tuple[int], np.dtype[np.float32]]
            ] = None,
            underactuated: bool = True,
            doubleintegrator: bool = False,
        ):
            self.__step = step
            if state is None:
                self.set_state()
            else:
                self.set_state(state)
            assert underactuated in [True, False]
            self.underactuated = underactuated

            if underactuated:
                if doubleintegrator:
                    self.__sat_dyn = (
                        self.__sat_dyn_underactuated_doubleintegrator
                    )
                else:
                    self.__sat_dyn = self.__sat_dyn_underactuated

                self.control = np.zeros((2,), dtype=np.float32)
                self.control_space = 2  # avoid gym space check
            else:
                if doubleintegrator:
                    self.__sat_dyn = (
                        self.__sat_dyn_fullyactuated_doubleintegrator
                    )
                else:
                    self.__sat_dyn = self.__sat_dyn_fullyactuated

                self.control = np.zeros((3,), dtype=np.float32)
                self.control_space = 3  # avoid gym space check
                # would be nice to have a check control space each time but
                # would slow down the code

            if EULER_SPEEDUP and self.__step < 0.1:
                self.step = self.euler_step
            else:
                self.step = self.rk4_step

            if self.__step > 0.5:
                warnings.warn(
                    "Step size is too large for the dynamics to be stable."
                    " Consider using a smaller step size."
                )

            return

        def set_state(
            self,
            state: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros(
                (6,), dtype=np.float32
            ),
        ):
            self.state = state
            return

        def set_control(
            self,
            control: np.ndarray[tuple[int], np.dtype[np.float32]],
        ):
            self.control = np.float32(control)
            return

        def get_state(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
            return np.array(self.state)

        def get_control(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
            return np.array(self.control)

        def __sat_dyn_underactuated(
            self,
            t: SupportsFloat,
            w: np.ndarray[tuple[int], np.dtype[np.float32]],
            u: np.ndarray[tuple[int], np.dtype[np.float32]],
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
                + (np.cos(w[2]) * u[0] / MASS)
            )

            dw[4] = (-2 * NU * w[3]) + (np.sin(w[2]) * u[0] / MASS)
            dw[5] = INERTIA_INV * u[1]

            return dw

        def __sat_dyn_underactuated_doubleintegrator(
            self,
            t: SupportsFloat,
            w: np.ndarray[tuple[int], np.dtype[np.float32]],
            u: np.ndarray[tuple[int], np.dtype[np.float32]],
        ):
            dw = np.zeros((6,), dtype=np.float32)
            dw[0] = w[3]
            dw[1] = w[4]
            dw[2] = w[5]
            dw[3] = (np.cos(w[2])) * u[0] / MASS
            dw[4] = (np.sin(w[2])) * u[0] / MASS
            dw[5] = INERTIA_INV * u[1]
            return dw

        def __sat_dyn_fullyactuated(
            self,
            t: SupportsFloat,
            w: np.ndarray[tuple[int], np.dtype[np.float32]],
            u: np.ndarray[tuple[int], np.dtype[np.float32]],
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

        def __sat_dyn_fullyactuated_doubleintegrator(
            self,
            t: SupportsFloat,
            w: np.ndarray[tuple[int], np.dtype[np.float32]],
            u: np.ndarray[tuple[int], np.dtype[np.float32]],
        ):
            dw = np.zeros((6,), dtype=np.float32)
            dw[0] = w[3]
            dw[1] = w[4]
            dw[2] = w[5]
            dw[3] = u[0] / MASS
            dw[4] = u[1] / MASS
            dw[5] = INERTIA_INV * u[2]
            return dw

        def euler_step(self):
            ts: np.float32 = self.__step
            t = np.zeros((1,), dtype=np.float32)
            w: np.ndarray[tuple[int], np.dtype[np.float32]] = self.get_state()
            u: np.ndarray[tuple[int], np.dtype[np.float32]] = (
                self.get_control()
            )
            k1 = self.__sat_dyn(t, w, u)
            self.set_state(w + ts * (k1))
            return self.state

        def rk4_step(self):
            ts: np.float32 = self.__step
            t = np.zeros((1,), dtype=np.float32)
            w: np.ndarray[tuple[int], np.dtype[np.float32]] = self.get_state()
            u: np.ndarray[tuple[int], np.dtype[np.float32]] = (
                self.get_control()
            )
            k1 = self.__sat_dyn(t, w, u)
            k2 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k1, u)
            k3 = self.__sat_dyn(t + 0.5 * ts, w + 0.5 * ts * k2, u)
            k4 = self.__sat_dyn(t + ts, w + ts * k3, u)
            self.set_state(w + ts * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            return self.state

        def reset(
            self,
            state: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros(
                (6,), dtype=np.float32
            ),
        ):
            self.set_state(state)
            if self.underactuated is True:
                self.set_control(np.zeros((2,), dtype=np.float32))
            else:
                self.set_control(np.zeros((3,), dtype=np.float32))
            return self.state

        def render(self):
            pass

        def radius(self) -> np.float32:
            state = np.array(self.state)
            return np.linalg.norm(state[0:2]).astype(np.float32)

        def speed(self) -> np.float32:
            state = np.array(self.state)
            return np.linalg.norm(state[3:5]).astype(np.float32)

        def symbolic_dynamics(self):
            pass

    class Target:
        # add dynamics later
        def __init__(
            self,
            step: np.float32 = STEP,
            state: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros(
                (2,), dtype=np.float32
            ),
        ):
            # state = [theta,theta_dot]
            self.state = state
            self.set_state(state)
            return

        def set_state(
            self,
            state: np.ndarray[Tuple[int], np.dtype[np.float32]] = np.zeros(
                (2,), dtype=np.float32
            ),
        ):
            self.state = np.float32(state)
            return

        def get_state(self) -> np.ndarray[Tuple[int], np.dtype[np.float32]]:
            return np.array(self.state, dtype=np.float32)

        def reset(
            self,
            state: np.ndarray[tuple[int], np.dtype[np.float32]] = np.zeros(
                (2,), dtype=np.float32
            ),
        ):
            self.set_state(state)
            return self.state

        def __remember(self):
            pass

        # add dynamics later when speed is needed


try:
    from numba import jit_module

    jit_module(nopython=True, error_model="numpy")
    print("Using Numba optimised methods.")

except ModuleNotFoundError:
    print("Using native Python methods.")
    print("Consider installing numba for compiled and parallelised methods.")


def _test():
    from stable_baselines3.common.env_checker import check_env

    check_env(Satellite_SE2(), warn=True)
    print("env checked")


def _test2(underactuated=True):
    env = Satellite_SE2(
        underactuated=underactuated,
        render_mode="human",
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

    model = PPO.load("Satellite_SE2")
    env = gym.make(
        "Satellite_SE2-v0",
        underactuated=underactuated,
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

    env = Satellite_SE2(
        render_mode=None,
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

    env = Satellite_SE2(
        underactuated=False,
        render_mode="rgb_array",
        max_action=np.float32(1),  # set to 1 for nomal control
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

    for _ in range(20000):
        action = -k @ env.chaser.get_state()

        if np.linalg.norm(action) > 1:
            action = action / np.linalg.norm(action)
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

    env = Satellite_SE2(
        underactuated=False,
        render_mode="human",
        max_action=np.float32(1),  # set to 1 for nomal control
        normalized_obs=False,
        step=np.float32(0.1),
    )
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    frames = [env.render()]
    action = np.array([0, 0, 0], dtype=np.float32)
    env.action_space.sample()
    print(env.action_space.sample())
    for _ in range(8000):
        action = -k @ env.chaser.get_state()
        act_norm = np.linalg.norm(action[0:2])

        if act_norm > FTMAX:
            action[0:2] = action[0:2] / act_norm * FTMAX
        if _ < 0:
            action = np.array([0, 0, 0], dtype=np.float32)
        observation, reward, term, trunc, info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        # if _ % 100 == 0:
        # frames.append(env.render())
    env.close()
    plt.plot(rewards)
    plt.show()
    clip = ImageSequenceClip(frames, fps=50)
    save_path = "./video_fully_lqr.mp4"
    clip.write_videofile(save_path, fps=50)


def _test8():
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

    env = Satellite_SE2(
        underactuated=True,
        render_mode="rgb_array",
        step=np.float32(0.05),
    )
    observation, info = env.reset()
    observations = [observation]
    rewards = []
    frames = [env.render()]
    actions = []
    action_full = np.array([0, 0, 0], dtype=np.float32)
    action_under = np.array([0, 0], dtype=np.float32)
    env.action_space.sample()
    print(env.action_space.sample())
    for _ in range(200000):
        state = env.chaser.get_state()
        action_full = -k @ env.chaser.get_state()
        act_norm = np.linalg.norm(action_full[0:2])
        # if act_norm > FTMAX:
        # action_full[0:2] = action_full[0:2] / act_norm * FTMAX
        ref_state = np.array(
            [0, 0, np.arctan2(action_full[1], action_full[0]), 0, 0, 0],
            dtype=np.float32,
        )
        print(ref_state)
        error = ref_state - env.chaser.get_state()
        action_under = np.array(
            [act_norm / (1 + np.power(10 * error[2], 2)), k[2, :] @ (error)],
            dtype=np.float32,
        )
        if _ < 4000:
            action_under = np.array([0, 0], dtype=np.float32)
        observation, reward, term, trunc, info = env.step(action_under)
        actions.append(action_under)
        observations.append(observation)
        rewards.append(reward)
        # if _ % 200 == 0:
        # frames.append(env.render())

    env.close()

    clip = ImageSequenceClip(frames, fps=50)
    save_path = "./video_under_lqr.mp4"
    clip.write_videofile(save_path, fps=50)
    plt.plot(actions)
    plt.show()


def _scalene_profiler():
    from scalene import scalene_profiler

    scalene_profiler.start()
    _test6()
    scalene_profiler.stop()


if __name__ == "__main__":
    from gymnasium.envs.registration import register

    _test6()
