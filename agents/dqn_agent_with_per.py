from collections import OrderedDict
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer

from jarmuiranyitas_1.agents.agent import Agent
from jarmuiranyitas_1.agents.misc.per_memory import PERMemory
from jarmuiranyitas_1.agents.misc.er_memory import ERMemory
from jarmuiranyitas_1.agents.misc.e_greedy import EGreedy
from jarmuiranyitas_1.agents.misc.neural_network import NNInitializer


class DQNAgentWithPER(Agent):
    def __init__(self,
                 learning_rate: float = 1e-4,
                 discount_factor: float = 0.99,
                 batch_size: int = 128,
                 network_size: list = None,
                 network_type: str = "cnn",
                 seed: int = 0,
                 epsilon_start: float = 1,
                 epsilon_decay: float = 0.99,
                 memory_size: int = 1e20,
                 memory_alpha: float = 0.6,
                 memory_beta: float = 0.4,
                 memory_beta_increment: float = 0.0005,
                 memory_epsilon: float = 0.01,
                 per: bool = False) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        if per:
            self.memory = PERMemory(memory_size, memory_alpha, memory_beta, memory_beta_increment, memory_epsilon)
        else:
            self.memory = ERMemory(memory_size)
        self.action_network = NNInitializer(network_type, network_size)
        self.target_network = NNInitializer(network_type, network_size)
        self.e_greedy = EGreedy(epsilon_start, epsilon_decay, self.action_network)
        self.seed = seed

    def seed(self, seed) -> None:
        pass

    def inference(self, state: np.ndarray) -> np.ndarray:
        pass

    def fit(self) -> None:
        pass
