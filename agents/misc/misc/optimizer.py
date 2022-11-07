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
    def __init__(self, memory, BATCH_SIZE):
        self.memory = memory
        self.BATCH_SIZE = BATCH_SIZE
        self.size = min(self.batch_size, len(self.memory))

    def get_sample(self):
        sample = self.memory.sample(self.size)
        mini_batch = self.memory.MEMORY_ITEM(*zip(*sample))

        state_batch = torch.FloatTensor(mini_batch.state)
        action_batch = torch.reshape(torch.LongTensor(mini_batch.action), (self.batch_size, 1))
        next_state_batch = torch.FloatTensor(mini_batch.next_state)
        reward_batch = torch.reshape(torch.FloatTensor(mini_batch.reward), (self.batch_size, 1))

    def calc_q_value(self):
        pass

    def optimizer_step(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0