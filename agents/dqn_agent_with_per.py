from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.tensorboard import SummaryWriter

from jarmuiranyitas_1.agents.agent import Agent
from jarmuiranyitas_1.agents.misc.memory.per_memory import PERMemory
from jarmuiranyitas_1.agents.misc.memory.r_memory import RMemory
from jarmuiranyitas_1.agents.misc.misc.epsilon_greedy import EGreedy
from jarmuiranyitas_1.agents.misc.neural_network.nn_initializer import NNInitializer


class DQNAgentWithPER(Agent):
    def __init__(self,
                 learning_rate: float = 1e-4,
                 discount_factor: float = 0.99,
                 batch_size: int = 128,
                 network_size: list = None,
                 network_type: str = "mlp",
                 seed: int = 0,
                 epsilon_start: float = 1,
                 epsilon_decay: float = 0.99,
                 memory_size: int = 1e20,
                 memory_alpha: float = 0.6,
                 memory_beta: float = 0.4,
                 memory_beta_increment: float = 0.0005,
                 memory_epsilon: float = 0.01,
                 per: bool = False,
                 ) -> None:

        self.discount_factor = discount_factor
        self.batch_size = batch_size

        nn_initializer = NNInitializer(network_type, network_size, seed)
        self.action_network = nn_initializer.generate_nn()
        self.target_network = nn_initializer.generate_nn()
        self.update_networks()

        self.optimizer = optimizer.Adam(self.action_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.summary_writer = SummaryWriter(log_dir="../experiments/logs")
        self.e_greedy = EGreedy(epsilon_start, epsilon_decay, seed, network_size[-1])
        if per:
            self.memory = PERMemory(memory_size, memory_alpha, memory_beta, memory_beta_increment, memory_epsilon, seed)
        else:
            self.memory = RMemory(memory_size, seed)

        self.seed(seed)

    def seed(self, seed) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def inference(self, state: np.ndarray) -> np.ndarray:
        action = self.e_greedy.choose_action()
        if action is None:
            state = torch.FloatTensor(state)
            action = self.action_network(state)
            action = torch.argmax(action)

        return action

    def fit(self) -> None:
        size = min(self.batch_size, len(self.memory))

        sample = self.memory.sample(size)
        mini_batch = self.memory.MEMORY_ITEM(*zip(*sample))

        state_batch = torch.FloatTensor(mini_batch.state)
        action_batch = torch.reshape(torch.LongTensor(mini_batch.action), (self.batch_size, 1))
        next_state_batch = torch.FloatTensor(mini_batch.next_state)
        reward_batch = torch.reshape(torch.FloatTensor(mini_batch.reward), (self.batch_size, 1))

        with torch.no_grad():
            output_next_state_batch = torch.reshape(self.target_network(next_state_batch), (size, -1))

        y_batch = []
        for i in range(size):
            if mini_batch.done[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.discount_factor * torch.max(output_next_state_batch[i]))

        y_batch = torch.cat(y_batch)
        output_batch = torch.reshape(self.action_network(state_batch), (size, -1))

        q_values = (torch.gather(output_batch, 1, action_batch)).squeeze()

        errors = torch.abs(q_values - y_batch).data.numpy()

        self.optimizer.zero_grad()
        loss = torch.mean(self.criterion(q_values, y_batch))
        loss.backward()
        for param in self.action_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_networks(self):
        self.target_network.load_state_dict(OrderedDict(self.action_network.state_dict()))

    def save_experience(self, **kwargs):
        self.memory.save(**kwargs)
