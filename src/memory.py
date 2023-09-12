import numpy as np


class ReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing_steps=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.index = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
        self.priorities[self.index] = self.max_priority
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta + (1 - self.beta) * \
            min(1.0, self.index / self.beta_annealing_steps)

        max_weight = (len(self.buffer) * probs.min()) ** (-beta)
        weights = (len(self.buffer) * probs[indices]) ** (-beta) / max_weight

        return indices, samples, weights

    def update_priority(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            self.priorities[i] = (abs(error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[i])

    def size(self):
        return len(self.buffer)
