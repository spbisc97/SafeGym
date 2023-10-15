import gymnasium as gym
from gymnasium import spaces
from typing import Any

SNAKE_LEN_GOAL = 30
from collections import deque
import numpy as np
import random
import time
import pygame

SCREEN_WIDTH = 500
INITIAL_SNAKE_POSITION = [[250, 250]]
GRID_SIZE = 10


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}
    # Game Vars

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.INITIAL_SNAKE_POSITION = INITIAL_SNAKE_POSITION
        self.GRID_SIZE = GRID_SIZE
        self.RATIO = self.SCREEN_WIDTH // self.GRID_SIZE
        # define action and observation space being gymnaium.spaces

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=np.hstack(
                (
                    np.array([0, 0, -500, -500, 0], dtype=int),
                    (np.ones(SNAKE_LEN_GOAL, dtype=int) * -1),
                )
            ),
            high=np.hstack(
                (
                    np.array([500, 500, 500, 500, SNAKE_LEN_GOAL], dtype=int),
                    (np.ones(SNAKE_LEN_GOAL, dtype=int) * 3),
                )
            ),
            shape=(5 + SNAKE_LEN_GOAL,),
            dtype=np.int64,
        )

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode
        if self.render_mode is not None:
            self._init_render()

        return

    def step(self, action):
        reward = 0
        terminated = False
        self.prev_actions.append(action)

        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down
        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += self.RATIO
        elif action == 0:
            self.snake_head[0] -= self.RATIO
        elif action == 2:
            self.snake_head[1] += self.RATIO
        elif action == 3:
            self.snake_head[1] -= self.RATIO

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            reward += 1000
            self.apple_position = self.collision_with_apple()
            self.snake_position.insert(0, list(self.snake_head))
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if (
            self.collision_with_boundaries() == 1
            or self.collision_with_self() == 1
        ):
            terminated = True
        # reward proposal
        euclidian_distance = np.sqrt(
            (self.snake_head[0] - self.apple_position[0]) ** 2
            + (self.snake_head[1] - self.apple_position[1]) ** 2
        )
        reward += -euclidian_distance / 500 if not terminated else -1000

        if self.render_mode == "human":
            self.render()

        if self.snake_position.__len__() >= SNAKE_LEN_GOAL:
            info = {"is_success": True}
            terminated = True
            reward = 1000

        else:
            info = {"is_success": False}

        # observation proposal
        # head_x, head_y, apple_delta_x, apple_delta_y, snake_length, previous_moves
        observation = self._get_obs()
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )  # observation, reward, terminated,truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        # self.img = np.zeros((500,500,3),dtype='uint8')
        ## Initial Snake and Apple position
        self.snake_position = self.INITIAL_SNAKE_POSITION
        self.apple_position = [
            random.randrange(1, self.GRID_SIZE) * self.RATIO,
            random.randrange(1, self.GRID_SIZE) * self.RATIO,
        ]
        self.score = 0
        self.action = 1
        self.snake_head = [250, 250]
        self.snake_position = [[250, 250]]
        ...
        # observation proposal
        # head_x, head_y, apple_delta_x, apple_delta_y, snake_length, previous_moves
        # head_x = self.snake_head[0]
        # head_y = self.snake_head[1]
        # apple_delta_x = head_x -self.apple_position[0]
        # apple_delta_y =  head_y -self.apple_position[1]
        # snake_length = len(self.snake_position)
        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = self._get_obs()
        info = {"is_success": False}

        return observation, info  # observation, info

    ## not needed
    def render(self):
        self.screen.fill((0, 0, 0))
        counter = 0
        color = [255, 255, 255]
        head_color = [0, 255, 0]
        for position in self.snake_position:
            pygame.draw.rect(
                self.screen,
                color if counter == 0 else head_color,
                pygame.Rect(
                    position[0],
                    position[1],
                    self.RATIO,
                    self.RATIO,
                ),
            )
            counter += 1
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(
                self.apple_position[0],
                self.apple_position[1],
                self.RATIO,
                self.RATIO,
            ),
        )
        pygame.display.flip()

        self.clock.tick(10)
        return

    def _init_render(self):
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_WIDTH)
            )
            self.clock = pygame.time.Clock()
        return

    def close(self):
        # if it is necessary to shut down the environment properly
        if self.render_mode == "human":
            pygame.quit()
        pass

    def _get_obs(self):
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        apple_delta_x = head_x - self.apple_position[0]
        apple_delta_y = head_y - self.apple_position[1]
        snake_length = len(self.snake_position)

        observation = np.array(
            [head_x, head_y, apple_delta_x, apple_delta_y, snake_length]
            + list(self.prev_actions)
        )
        return observation

    def collision_with_apple(self):
        apple_position = [
            random.randrange(1, self.GRID_SIZE) * self.RATIO,
            random.randrange(1, self.GRID_SIZE) * self.RATIO,
        ]
        self.score += 1
        return apple_position

    def collision_with_boundaries(self):
        if (
            self.snake_head[0] >= 500
            or self.snake_head[0] < 0
            or self.snake_head[1] >= 500
            or self.snake_head[1] < 0
        ):
            return 1
        else:
            return 0

    def collision_with_self(self):
        if self.snake_head in self.snake_position[1:]:
            return 1
        else:
            return 0


def main():
    env = SnakeEnv("human")
    for i in range(10):
        env.reset()
        while True:
            env.render()
            time.sleep(0.1)
            obs, reward, term, trunc, info = env.step(
                env.action_space.sample()
            )
            print(obs, reward, term, trunc, info)
            if term or trunc:
                break
    env.close()


def key_press():
    gym.register(
        id="Snake-v0",
        entry_point="Snake_v0:SnakeEnv",
        kwargs={"render_mode": "human"},
        max_episode_steps=500,
    )
    env = gym.make("Snake-v0", render_mode="human")

    episodes = 0
    term = False
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        counter = 0
        switcher = {
            "1": 1,
            "2": 2,
            "3": 0,
            "4": 3,
        }
        while not term:
            counter += 1
            env.render()
            key = int(1 + np.floor(counter / 4) % 4)
            action = switcher.get(str(key), 1)
            time.sleep(0.1)
            obs, reward, term, trunc, info = env.step(action)
            print(obs)
            print(len(obs))
            if term or trunc:
                term = False
                break
    episodes = 3

    term = False
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        print(obs)
        print(len(obs))
        while not term:
            action_button = input("Press Enter to continue...")
            switcher = {
                "a": 0,
                "d": 1,
                "w": 3,
                "s": 2,
            }
            action = switcher.get(action_button, 1)
            env.render()
            obs, reward, term, trunc, info = env.step(action)
            print(obs)
            print(reward)
            print(len(obs))

            if term or trunc:
                term = False
                break
    env.close()


def train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    env = SnakeEnv()
    env = make_vec_env(lambda: SnakeEnv(), n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=1,
    )
    model.load("snake_model")

    model.learn(total_timesteps=100_000)
    model.save("snake_model")

    env = SnakeEnv(render_mode="human")
    obs, info = env.reset()
    while not info["is_success"]:
        time.sleep(0.5)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, term, trunc, info = env.step(action)
        print(obs, rewards, term, trunc, info)
        if term or trunc:
            term = False
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    train()
