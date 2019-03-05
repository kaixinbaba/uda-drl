import numpy as np


class ReplayBuffer(object):

    def __init__(self, n_s, n_a, memory_size=10000):
        self.n_s = n_s
        self.n_a = n_a
        self.memory_size = memory_size
        self.memory = np.zeros([memory_size, n_s * 2 + n_a + 2])
        self.memory_count = 0
        self.pointer = 0

    def store(self, s, a, r, s_, done):
        transaction = np.hstack([s, a, r, done, s_])
        self.memory[self.pointer, :] = transaction
        self.pointer = 0 if (self.pointer == self.memory_size) else (self.pointer + 1)
        self.memory_count += 1

    def sample(self, batch_size):
        batch_index = np.random.choice(range(self.memory_size), batch_size)
        return self.memory[batch_index]
