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
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.current_epsilon = self.epsilon_start
        self.action_type = str()

    def calculate_epsilon_decay(self)->None:
        """
        To compare exponential discount of epsilon value
        :return: None
        """
        self.current_epsilon = self.current_epsilon * self.epsilon_decay

    def generate_reference_value(self)->None:
        """
        By choosing an action there is a value to compare with epsilon between 0 and 1 random choosen in the
        domain 0-1 .
        :return:None
        """
        self.reference_value = np.random()

    def choose_action_type(self)->None:
        """
        By comparsion of the two float values between 0 and 1 we can get the type of the action . By discounting
        epsilon we will get initially more exploring actions , in the end after several time steps by low epsilon
        value the reference will be in the most cases higher than the epsilon value .
        :return:None
        """
        self.generate_reference_value()
        self.current_epsilon()

        if self.reference_value > self.current_epsilon:
            self.action_type = "exploit"
        else:
            self.action_type = "explore"


    def choosing_action(self):
        """
        The choosing of the action of the time step
        :return: action(type not defined yet)
        """
        self.choose_action_type()
        if self.action_type == "exploit":
            #argmax
            pass
        elif self.action_type == "explore":
            #random
            pass
        else:
            #default case
            pass

        #TODO:return parameter with the action value(not defined yet)
