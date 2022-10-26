import random
import numpy as np
from collections import deque
from jarmuiranyitas_1.agents.misc.memory import Memory

class RMemory(Memory):
    """
    This class is responsible for storing the experiments batched in a tuple
    The deque represents the memory , with 2 dimensional list
    state : not implemented yet
    reward : float
    action : not implemented yet
    next state :not implemented yet
    done : bool
    """

    def __init__(self, size: int) -> None:
        """

        :param size: size of the memory buffer
        :return: None
        """
        pass

    def save(self, experience: np.ndarray) -> None:
        """

        :param experience:
        :return: None
        """
        pass

    def sample(self):
        """

        :return: sampled experience
        """
        pass

