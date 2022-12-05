from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn

from jarmuiranyitas_1.agents.agent import Agent
from torch.utils.tensorboard import SummaryWriter
from jarmuiranyitas_1.agents.misc.misc.optimizer import Optimizer
from jarmuiranyitas_1.agents.misc.misc.epsilon_greedy import EGreedy
from jarmuiranyitas_1.agents.misc.memory.r_memory import RMemory


class LateralAgent(Agent):
    """
    ------------------------
         ***abstract***
           Agent Class
    ------------------------

    Methods:
        __init__(self,
                 learning_rate: float,
                 discount_factor: float,
                 batch_size: int,
                 network_size: list,
                 seed: int) -> None
        reset(self) -> numpy.ndarray
        fit(self) -> None
        inference(self, state: np.ndarray) -> np.ndarray
        seed(self, seed) -> None
    """
    @abstractmethod
    def __init__(self,
                 learning_rate: float = 1e-4,
                 discount_factor: float = 0.99,
                 batch_size: int = 128,
                 network_size: list = None,
                 network_type: str = "mlp",
                 seed: int = 0,
                 epsilon_start: float = 1,
                 epsilon_decay: float = 0.99,
                 BUFFER_SIZE = 1e20) -> None:
        """
        This function initializes the agent corresponding to its type and the given parameters.
        :param learning_rate: alpha parameter
        :param discount_factor: gamma parameter
        :param batch_size: number of training examples used in the estimate of the error gradient
        :param network_size: size of the different layers in the neural network
        :param seed: seed number
        """
        #net and bellman parameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.network_size = network_size
        self.network_type = network_type
        self.BUFFER_SIZE = BUFFER_SIZE

        #logging results
        self.summary_writer = SummaryWriter(log_dir="../experiments/logs")

        #objects
        self.e_greedy = EGreedy(epsilon_start, epsilon_decay, seed, network_size[-1])
        self.memory = RMemory(self.BUFFER_SIZE, seed)
        self.loss = nn.MSELoss()
        self.optimizer = optimizer.Adam(self.action_network.nn_parameters, lr=learning_rate)


    @abstractmethod
    def fit(self) -> None:
        """
        This function realizes the update of the model by fitting the training data.
        :return: None
        """
        pass

    @abstractmethod
    def inference(self, state: np.ndarray) -> np.ndarray:
        """
        This function generates an output for the given state passed as an argument.
        :param state: input matrix
        :return action: output matrix
        """
        pass

    @abstractmethod
    def seed(self, seed) -> None:
        """
        This function seeds the agent and all its dependencies.
        :param seed: int value for seeding the agent
        :return: None
        """

        pass

    @abstractmethod
    def save_experience(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        pass
