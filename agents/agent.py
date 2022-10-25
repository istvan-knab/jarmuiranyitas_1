from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
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
                 learning_rate: float,
                 discount_factor: float,
                 batch_size: int,
                 network_size: list,
                 seed: int) -> None:
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
        pass
