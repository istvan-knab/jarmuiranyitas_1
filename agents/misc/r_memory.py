import random
import numpy as np
from jarmuiranyitas_1.agents.misc.memory import Memory
from collections import namedtuple, deque

Experiences = namedtuple('Experiences', ('state', 'action', 'next_state', 'reward', 'done'))


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
        Initializing the size of memory
        :param size: size of the memory buffer
        :return: None
        """
        self.capacity = size
        self.memory = deque([], maxlen=self.capacity)
        pass

    def save(self, *args: tuple) -> None:
        """

        :param experience:a named tuple which stores the states, action and reward , done
        foc comparsion and optimization
        :return: None
        """
        self.memory.append(Experiences(*args))

    def sample(self, batch_size) -> tuple:
        """
        Select random lines of the sample batch , in size of batch
        :return: the reduced strucure in range of batch
        """
        return random.sample(self.memory, batch_size)

