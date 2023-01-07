import numpy as np
import random

import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


def build_array(n):
    a = [0]
    b = [0 for i in range(1, n)]
    c = a + b

    return np.array(c, dtype=np.float64)


def create_table(nrow, ncol):
    a = np.zeros((nrow, ncol), dtype=np.float64)
    a.fill(0.0001)
    return a


class Bayes_HyperQAgent:
    def __init__(self, N, alpha, gamma, mu, mixed_strategies):
        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.mixed_strategies = mixed_strategies
        self.len = len(self.mixed_strategies)

        self.Q = create_table(self.len+1, self.len+1)
        self.N = N
        self.Bayes_values = build_array(self.len+1)
        self.P_H_Y_values = build_array(self.len+1)
        self.frequencies = build_array(self.len+1)

    def act(self, random_move=False):

        if random_move:
            res = random.randint(0, self.len- 1)
            return [res, self.mixed_strategies[res]]
        else:

            op_strat = self.estimate_opponent_strategy()

            line = self.Q[op_strat]

            res = line.argmax()

            return [res, self.mixed_strategies[res]]

    def compute_bayes_values(self, observed_states):

        for i in range(len(observed_states)):
            self.Bayes_values[observed_states[i]] = self.Bayes_formulation(observed_states[i])

    def compute_phy_values(self, observed_states):

        for i in range(1, self.len+1):
            self.P_H_Y_values[i] = self.P_H_Y(i, observed_states)

    def estimate_opponent_strategy(self):
        """
        Returns the most probable opponent strategy using the bayes formulation estimation.
        """

        op_strat = self.Bayes_values
        res = op_strat.argmax()

        return res

    def compute_prior_proba(self, observed_states):
        elem = np.arange(0,self.len+1)

        counts = np.bincount(observed_states[np.isin(observed_states, elem)], minlength=len(elem))

        self.frequencies =  counts / np.sum(counts)



    def Bayes_formulation(self, state):


        numerator = self.P_H_Y_values[state]*self.frequencies[state]

        denominator = np.sum(self.P_H_Y_values*self.frequencies)

        return numerator/denominator


    def P_H_Y(self, state ,observed_states):

        if state not in observed_states:
            return 0


        n = len(observed_states)

        prod = 1
        for i in range(n):
            if state == observed_states[i]:
                prod *= (state/(self.len+1))**(1- self.mu*(n-i))

        return prod

    def learn(self, x, y, reward, observed_actions):

        self.compute_prior_proba(observed_actions)
        self.compute_phy_values(observed_actions)
        self.compute_bayes_values(observed_actions)

        esti_op_next_strat = self.estimate_opponent_strategy()
        next_best_strat = self.act()

        a = self.alpha * self.Bayes_values[y]
        b = reward + self.gamma * self.Q[esti_op_next_strat, next_best_strat[0]] - self.Q[y, x]
        self.Q[y, x] += a * b
