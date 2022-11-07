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
    def __init__(self, memory, BATCH_SIZE , optimizer):
        self.memory = memory
        self.BATCH_SIZE = BATCH_SIZE
        self.size = min(self.batch_size, len(self.memory))
        self.optimizer = optimizer

    def get_sample(self):
        sample = self.memory.sample(self.size)
        mini_batch = self.memory.MEMORY_ITEM(*zip(*sample))

        self.state_batch = torch.FloatTensor(mini_batch.state)
        self.action_batch = torch.reshape(torch.LongTensor(mini_batch.action), (self.batch_size, 1))
        self.next_state_batch = torch.FloatTensor(mini_batch.next_state)
        self.reward_batch = torch.reshape(torch.FloatTensor(mini_batch.reward), (self.batch_size, 1))

        with torch.no_grad():
            output_next_state_batch = torch.reshape(self.target_network(self.next_state_batch), (self.size, -1))

    def calc_q_value(self):

        y_batch = []
        for i in range(self.size):
            if self.mini_batch.done[i]:
                y_batch.append(self.reward_batch[i])
            else:
                y_batch.append(self.reward_batch[i] + self.discount_factor * torch.max(self.output_next_state_batch[i]))

        y_batch = torch.cat(y_batch)
        output_batch = torch.reshape(self.action_network(self.state_batch), (self.size, -1))
        q_values = (torch.gather(output_batch, 1, self.action_batch)).squeeze()

        errors = torch.abs(q_values - y_batch).data.numpy()

    def optimizer_step(self):

        self.get_sample()
        self.calc_q_value()

        self.optimizer.zero_grad()
        #Todo : add loss function as argument------->finish def