from abc import ABC, abstractmethod
import numpy as np
import torch
import random

from jarmuiranyitas_1.agents.misc.misc.optimizer import Optimizer
from jarmuiranyitas_1.agents.misc.misc.epsilon_greedy import EGreedy


class LateralAgent(ABC):
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
                 BUFFER_SIZE: int = 1e20,
                 )->None:
        """
        This function initializes the agent corresponding to its type and the given parameters.
        :param learning_rate: alpha parameter
        :param discount_factor: gamma parameter
        :param batch_size: number of training examples used in the estimate of the error gradient
        :param network_size: size of the different layers in the neural network
        :param seed: seed number
        """
        pass

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
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        pass

    @abstractmethod
    def save_experience(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        self.memory.save(**kwargs)
