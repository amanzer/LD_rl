import math

import numpy as np
import random


def create_table(nrow, ncol):
    a = np.zeros((nrow, ncol), dtype=np.float64)
    a.fill(0.0001)
    return a


class Omniscient_HyperQAgent:
    def __init__(self, N, alpha, gamma, mixed_strategies):
        self.alpha = alpha
        self.gamma = gamma
        self.mixed_strategies = mixed_strategies
        self.Q = create_table(len(self.mixed_strategies), len(self.mixed_strategies))
        self.N = N

    def act(self, opponent_strategy, random_move=False):

        if random_move:
            res = random.randint(0, len(self.mixed_strategies) - 1)
            return [res, self.mixed_strategies[res]]
        else:


            line = self.Q[opponent_strategy]

            res = line.argmax()

            return [res, self.mixed_strategies[res]]

    def learn(self, x, y, reward, opponent_strat):

        next_best_strat = self.act(opponent_strat)

        self.Q[y, x] += self.alpha * (reward + self.gamma * self.Q[opponent_strat, next_best_strat[0]] - self.Q[y, x])
