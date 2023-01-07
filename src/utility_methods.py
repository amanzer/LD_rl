import numpy as np
import random


# def choose_action(probs):
#     # Choose action according to the current mixed policy
#     action = np.random.choice(len(probs),
#                               p=probs)  # p defines the probability, otherwise it assumes uniform distribution
#     return action

def choose_action(probs):
    # Choses the action according to the 3 probabilites of the mixed strategy
    rand = random.random()
    cum_prob = 0
    for i, prob in enumerate(probs):
        cum_prob += prob
        if rand < cum_prob:
            return i
    return len(probs) - 1


def build_simplex_grid(N):
    """
    Returns a simplex grid containing all the possible mixed strategies built with a
    uniform grid discretization of size N.
    :param N:
    """
    points = [[0, 0, 0]]
    for i in range(N + 1):
        for j in range(N - i):
            k = N - i - j
            points.append([i / N, j / N, k / N])
    return np.array(points[1:], dtype=np.float64)


