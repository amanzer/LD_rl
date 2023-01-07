import copy

from src.utility_methods import *


class PolicyHillClimbingAgent:
    def __init__(self,
                 mixed_strategies,
                 learning_rate=0.01,
                 delta=0.01,
                 gamma=0.9,
                 epsilon=0.5):
        """
        :param mixed_strategies : list of policies used by the hyper-Q agent
        :param learning_rate: The learning rate for the Q-table update
        :param delta : learning rate for the policy update
        :param gamma: The discount factor.
        :param epsilon : exploration parameter
        """
        self.n_actions = 3
        self.learning_rate = learning_rate
        self.delta = delta
        self.gamma = gamma
        self.q_table = np.zeros(self.n_actions)
        self.mixed_strategies = mixed_strategies
        self.policy = np.array([0.3333333, 0.3333333, 0.3333333])
        self.epsilon = epsilon

    def act(self) :
        """
        Return the action according to the current policy.

        :param training: Boolean flag for training, when not training agent
        should act greedily.
        :return: The action.
        """
        return self.closest_mixed_strategy(self.policy)

    def greedy_action(self) -> int:
        """
        Returns the greedy action.
        """
        return self.q_table.argmax()

    def learn(self, act: int, rew: float) -> None:
        """
        Update the Q-Value and Policy according to the formula from M. Bowling and M. Veloso.
        """
        greedy_act = self.greedy_action()

        # Updating the Qtable
        self.q_table[act] = ((1 - self.learning_rate) * self.q_table[act]) + self.learning_rate * (
                rew + self.gamma * self.q_table[greedy_act])

        # # Updating the policy
        current_policy = copy.deepcopy(self.policy)

        if current_policy[greedy_act] + self.delta > 1:
            current_policy[greedy_act] = 1
        else:
            current_policy[greedy_act] = current_policy[greedy_act] + self.delta

        for i in range(self.n_actions):
            if i != greedy_act:
                if current_policy[i] - self.delta < 0:
                    current_policy[i] = 0.0
                else:
                    current_policy[i] -= self.delta / (self.n_actions - 1)

        #current_policy = np.true_divide(current_policy, np.sum(current_policy))
        #print(f"after learn : {current_policy}")
        self.policy = current_policy
        #print(f"closest ms : {self.policy[1]}")

    def closest_mixed_strategy(self, strategy):
        """
        For the training against Hyper-Q agent we want to use the same mixed strategies,
        this method will find the closest to the policy to the one created after the learn formula
        is applied.
        :param strategy: policy given after the learn formula
        :return: policy from mixed strategies closest to it.
        """
        min_distance = float("inf")
        closest_array = None
        res = 1
        for i in range(1, len(self.mixed_strategies)):
            distance = np.linalg.norm(self.mixed_strategies[i] - strategy)
            # If distance is smaller than current minimum, update minimum and closest array
            if distance < min_distance:
                min_distance = distance
                closest_array = self.mixed_strategies[i]
                res = i
        return [res, closest_array]
