import torch
import numpy
from collections import namedtuple

from jarmuiranyitas_1.agents.misc.memory.memory import Memory
from jarmuiranyitas_1.agents.misc.memory.r_memory import RMemory
from jarmuiranyitas_1.agents.misc.memory.per_memory import PERMemory
from jarmuiranyitas_1.agents.dqn_agent_with_per import DQNAgentWithPER


class Optimizer:
    """
    This class is responsible for optimize the modell with gradient descent
    """

    MEMORY_ITEM = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
    def __init__(self, memory,BATCH_SIZE):
        self.memory = memory
        self.BATCH_SIZE = BATCH_SIZE

    def get_sample(self):
        pass

    def calc_q_value(self):
        pass

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0