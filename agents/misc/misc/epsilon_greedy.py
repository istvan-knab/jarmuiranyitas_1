import numpy as np


class EGreedy:
    """
    This class is responsible for choosing between exploring or exploiting
    """
    def __init__(self, epsilon_start: float, epsilon_decay: float, seed: int, output_dim) -> None:
        """
        The initial parameters of the discount
        :param epsilon_start: starting value of epsilon
        :param epsilon_decay: Discount factor of the value decent
        """
        self.current_epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.output_dim = output_dim

        self.seed(seed)

    def choose_action(self):
        """
        The choosing of the action of the time step
        :return: action(type not defined yet)
        """
        action = None
        if np.random.random() <= self.current_epsilon:
            action = np.random.randint(0, self.output_dim)

        return action

    def update_epsilon(self):
        self.current_epsilon = self.current_epsilon * self.epsilon_decay

    @staticmethod
    def seed(seed) -> None:
        np.random.seed(seed)
