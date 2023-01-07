"""
IGA (infinitesimal gradient ascent) agent implementation for 2x2 matrix games.

Implemented according to the original paper :
    Satinder Singh, Michael Kearns, and Yishay Mansour. 2000.
    "Nash convergence of gradient dynamics in general-sum games."
    In Proceedings of the Sixteenth conference on Uncertainty in artificial intelligence (UAI'00).
    Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 541–548.
"""


class IGAAgent:
    """
    The agent class for IGA (infinitesimal gradient ascent)
    """

    def __init__(self, mixed_strategy, row_player_payoff_matrix, learning_rate=0.01):
        """
        :param learning_rate: Learning rate = alpha (0.01 as specified in the article)
        :param mixed_strategy: mixed strategy of the agent (probability of playing first action)
        """
        self.learning_rate = learning_rate          # step size in the paper (lowercase Eta)
        self.mixed_strategy = mixed_strategy        # alpha in the original paper
        self.payoffs = row_player_payoff_matrix     # matrix of payoffs for the row player

    def calculate_gradient(self, opponent_mixed_strategy):
        """
        Calculate the gradient based on the formula from the original paper
        :param opponent_mixed_strategy: mixed strategy of the opponent (their probability of playing first action)
        :return: computed gradient
        """
        u = (self.payoffs[0][0] + self.payoffs[1][1]) - (self.payoffs[1][0] + self.payoffs[0][1])
        gradient = (opponent_mixed_strategy * u) - (self.payoffs[1][1] - self.payoffs[0][1])

        return gradient

    def update(self, opponent_mixed_strategy):
        """
        Update the mixed strategy according to the opponent's mixed strategy
            (learning rate is called the step size in the papers)
        :param opponent_mixed_strategy: mixed strategy of the opponent (their probability of playing action 1
        """
        update_by = self.learning_rate * self.calculate_gradient(opponent_mixed_strategy)

        # Update the value only if it does not go over 1.0 or under 0.0
        if (self.mixed_strategy + update_by <= 1.0) and (self.mixed_strategy + update_by >= 0.0):
            self.mixed_strategy += update_by

    def act(self):
        """
        Act by choosing a greedy move
        """
        pass
