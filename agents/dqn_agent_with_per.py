from collections import OrderedDict
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.tensorboard import SummaryWriter

from jarmuiranyitas_1.agents.agent import Agent
from jarmuiranyitas_1.agents.misc.per_memory import PERMemory
from jarmuiranyitas_1.agents.misc.r_memory import RMemory
from jarmuiranyitas_1.agents.misc.epsilon_greedy import EGreedy
from jarmuiranyitas_1.agents.misc.nn_initializer import NNInitializer


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
        pass

    def fit(self) -> None:
        pass
