"""
replay_buffer.py
"""

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    A ReplayBuffer object stores a set of experiences under the form of:
    (state, action, reward, is_done, next_state)
    """

    def __init__(self, size):
        self.size = size
        self.count = 0
        self.buffer = deque()

    def store(self, s, a, r, done, ns):
        elem = (s, a, r, done, ns)
        if self.count < self.size:
            self.buffer.append(elem)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.appendleft(elem)

    def sample(self, batch_size):
        _batch = random.sample(self.buffer, batch_size)
        s_batch = [_[0] for _ in _batch]
        a_batch = [_[1] for _ in _batch]
        r_batch = [_[2] for _ in _batch]
        d_batch = [_[3] for _ in _batch]
        ns_batch = [_[4] for _ in _batch]
        return s_batch, a_batch, r_batch, d_batch, ns_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count
