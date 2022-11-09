from abc import ABC, abstractmethod
import numpy as np


class Save(ABC):
    """
    Abstract class to define the format of saving parameters
    """
    @abstractmethod
    def save(self, data: np.array):
        pass


class SaveModel(Save):
    """

    """
    def save(self, data: list) -> None:
        pass


class SaveReward(Save):
    """

    """
    def save(self, data: list) -> None:
        pass


class SaveTrainingResult(Save):
    """

    """
    def save(self, data: list) -> None:
        pass