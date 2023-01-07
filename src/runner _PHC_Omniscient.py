import time
import tqdm
import matplotlib.pyplot as plt

import pickle


from Agents.Hyper_Q_Omniscient import Omniscient_HyperQAgent
from Agents.PolicyHillClimbing import PolicyHillClimbingAgent
from utility_methods import *

payoff = np.array([[(0, 0), (-1, 1), (1, -1)],
                   [(1, -1), (0, 0), (-1, 1)],
                   [(-1, 1), (1, -1), (0, 0)]],
                  dtype=object)


def hyper_q_learning(nb_iterations, N, alpha, gamma):
    """
    Training loop for two agents : Hyper-Q (Omniscient) vs PHC
    """
    # used for bellman error
    n_states = 3  # There are 3 possible states       (rock, paper or scissors)
    V = np.zeros(n_states)
    errors = np.zeros(nb_iterations)

    mixed_strategies = build_simplex_grid(N)

    avg_returns = np.zeros(nb_iterations)
    agent1 = Omniscient_HyperQAgent(N, alpha, gamma, mixed_strategies)
    agent2 = PolicyHillClimbingAgent(mixed_strategies)

    scores = [0, 0]

    action1 = agent1.act(0, True)
    action2 = agent2.act()

    a1 = choose_action(action1[1])
    a2 = choose_action(action2[1])

    current_state = a1

    rew1 = payoff[a1, a2, 0]
    rew2 = payoff[a1, a2, 1]
    agent2.learn(a2, rew2)

    agent1.learn(action1[0], action2[0], rew1, agent2.closest_mixed_strategy(agent2.policy)[0])

    scores[0] += rew1
    scores[1] += rew2
    for i in tqdm.trange(nb_iterations):

        action2 = agent2.act()

        # Act randomly every 1000 steps
        if i % 1000 == 0:
            action1 = agent1.act(action2[0], True)
        else:
            action1 = agent1.act(action2[0])

        a1 = choose_action(action1[1])
        a2 = choose_action(action2[1])

        rew1 = payoff[a1, a2, 0]
        rew2 = payoff[a1, a2, 1]

        agent2.learn(a2, rew2)

        agent1.learn(action1[0], action2[0], rew1, agent2.closest_mixed_strategy(agent2.policy)[0])

        scores[0] += rew1
        scores[1] += rew2

        avg_returns[i] = rew1

        # Update the value function (bellman backup ?)
        V[current_state] = (alpha * (rew1 + V[a1])) + (1 - alpha * V[current_state])

        # Store the error in an array
        errors[i] = (V[current_state] - (rew1 + V[a1])) ** 2  # error = current state value - value at the next state

        current_state = a1  # Update the current state for the next time

    return avg_returns, errors


if __name__ == '__main__':
    t1 = time.time()
    nb_iterations = 16000000
    nb_sim = 10
    N = 25
    alpha = 0.01
    gamma = 0.9

    rewards = np.zeros((nb_iterations, nb_sim))
    bellman_error = np.zeros((nb_iterations, nb_sim))

    for i in range(nb_sim):
        results = hyper_q_learning(nb_iterations, N, alpha, gamma)

        rewards[:, i] = results[0]
        bellman_error[:, i] = results[1]


    """
    # Open a file in write mode
    with open('avg_results_PHCvsOMNISCIENT.pkl', 'wb') as f:
        # Use pickle.dump to serialize the array and write it to the file
        pickle.dump(np.mean(rewards, axis=1), f)
        pickle.dump(np.mean(bellman_error, axis=1), f)
    """

    reduced_rewards = np.mean(rewards.reshape(-1, 10000), axis=1)
    reduced_bellman_error = np.mean(bellman_error.reshape(-1, 10000), axis=1)

    plt.plot(reduced_rewards)
    plt.show()
    t2 = time.time()
    print(t2 - t1)

    plt.clf()
    plt.plot(reduced_bellman_error)
    plt.title("Bellman error HyperQ omniscient vs PHC")
    plt.ylabel("Bellman error")
    plt.show()


