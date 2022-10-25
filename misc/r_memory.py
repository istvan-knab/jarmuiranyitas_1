import random
from collections import deque


class RMemory(object):
    """
    This class is responsible for storing the experiments batched in a tuple
    The deque represents the memory , with 2 dimensional list
    state : not implemented yet
    reward : float
    action : not implemented yet
    next state :not implemented yet
    done : bool
    """

    def __init__(self, BUFFER_SIZE):

        self.memory = deque([[],[],[],[],[]], maxlen=BUFFER_SIZE)

    def push(self):
        """
        Save a transition
        :return:
        """
        #define tuple outside
        pass

    def sample(self, batch_size):
        """
        RAndom sample for optimization
        :param batch_size: integer
        :return: sample
        """
        return random.sample(self.memory, batch_size)

