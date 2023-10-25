# @title ReplayBuffer
from typing import Tuple
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self, state, action, reward, next_state, done, unsafe=False
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[
            self.position
        ] = (  # needs some fixes for random sampling substitution
            state,
            action,
            reward,
            next_state,
            done,
            unsafe,
        )
        self.position = self.position + 1

    def sample(
        self, batch_size
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, unsafe = map(
            np.stack, zip(*batch)
        )
        return state, action, reward, next_state, done

    def safety_sample(
        self, batch_size
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, unsafe = map(
            np.stack, zip(*batch)
        )
        return state, action, reward, next_state, done, unsafe

    def __len__(self) -> int:
        return len(self.buffer)
