import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity: int, state_shape: tuple):
        self.capacity = capacity
        self.states = np.empty((capacity,) + state_shape, dtype=np.float32)
        self.next_state = np.empty((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.empty((capacity,), dtype=int)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)
        self.i = 0
        self.buffer_full = False

    def add_transition(self, state: np.ndarray, action: int, reward: float,
                       done: bool, next_state: np.ndarray):
        self.states[self.i] = state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.dones[self.i] = done
        self.next_state[self.i] = next_state

        self.i += 1
        if self.i == self.capacity:
            self.buffer_full = True
            self.i = 0

    def __len__(self):
        return self.i if not self.buffer_full else self.capacity

    def sample(self, n: int = 1):
        indices = np.random.randint(0, len(self), n)
        return self.states[indices], self.actions[indices], self.rewards[indices], \
               self.dones[indices], self.next_state[indices]
               