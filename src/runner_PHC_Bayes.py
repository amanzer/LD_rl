import pickle
import time

import tqdm
import matplotlib.pyplot as plt

from Agents.Hyper_Q_Bayes import Bayes_HyperQAgent
from Agents.PolicyHillClimbing import PolicyHillClimbingAgent
from utility_methods import *

payoff = np.array([[(0, 0), (-1, 1), (1, -1)],
                   [(1, -1), (0, 0), (-1, 1)],
                   [(-1, 1), (1, -1), (0, 0)]],
                  dtype=object)

def most_frequent(arr):
    # Find the unique elements and their counts
    unique, counts = np.unique(arr, return_counts=True)
    # Find the index of the maximum count
    max_index = np.argmax(counts)
    # Return the most frequent element
    return unique[max_index]


def hyper_q_learning(nb_iterations, N, alpha, gamma, mu):
    """
    Training loop for two agents : Hyper-Q (Omniscient) vs PHC
    """
    mixed_strategies = build_simplex_grid(N)

    avg_returns = np.zeros(nb_iterations)
    agent1 = PolicyHillClimbingAgent(mixed_strategies)
    agent2 = Bayes_HyperQAgent(N, alpha, gamma, mu,mixed_strategies)

    scores = [0, 0]

    action1 = agent1.act()
    action2 = agent2.act(True)      # First move is random

    a1 = choose_action(action1[1])
    a2 = choose_action(action2[1])

    rew1 = payoff[a1, a2, 0]
    rew2 = payoff[a1, a2, 1]

    obs_state = np.array([action1[0]+1])


    agent2.learn(action2[0], action1[0], rew2, obs_state)

    agent1.learn(a1, rew1)

    scores[0] += rew1
    scores[1] += rew2
    for i in tqdm.trange(nb_iterations):

        # Act randomly every 1000 steps

        if i % 1000 == 0:
            action2 = agent2.act(True)
        else:
            action2 = agent2.act()

        action1 = agent1.act()
        a1 = choose_action(action1[1])
        a2 = choose_action(action2[1])

        rew1 = payoff[a1, a2, 0]
        rew2 = payoff[a1, a2, 1]

        obs_state = np.append(obs_state, action1[0]+1)

        if (len(obs_state) > 500):
            obs_state = obs_state[(len(obs_state)-1)//2:]
        agent2.learn(action2[0], action1[0]+1, rew2, obs_state)

        agent1.learn(a1, rew1)

        scores[0] += rew1
        scores[1] += rew2

        avg_returns[i] = rew2

    return avg_returns

if __name__ == '__main__':
    t1 = time.time()
    nb_iterations = 1600000
    nb_sim = 10
    N = 25
    alpha = 0.01
    gamma = 0.9
    mu = 0.005

    rewards = np.zeros((nb_iterations, nb_sim))
    for i in range(nb_sim):
        results = hyper_q_learning(nb_iterations,  N, alpha, gamma, mu)
        rewards[:, i] = results[0]

    
    #with open('avg_results_PHCvsBAYES.pkl', 'wb') as f:
    #    pickle.dump(np.mean(rewards, axis=1), f)

    reduced_rewards = np.mean(rewards.reshape(-1, 10000), axis=1)

    plt.plot(reduced_rewards)
    plt.show()
    t2 = time.time()
    print(t2 - t1)
