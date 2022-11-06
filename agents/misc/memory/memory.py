from abc import ABC, abstractmethod
from collections import namedtuple


class Memory(ABC):
    """
    ------------------------
         ***abstract***
           Memory Class
    ------------------------

    Methods:
        __init__(self, size: int) -> None
        save(self, experience: np.ndarray) -> None
        sample(self)
    """
    MEMORY_ITEM = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    @abstractmethod
    def __init__(self, size: int, seed: int) -> None:
        """

        :param size: size of the memory buffer
        :param seed:
        :return: None
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """

        :return: None
        """
        pass

    @abstractmethod
    def save(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def sample(self, size: int) -> list:
        """

        :return: sampled experience
        """
        pass

    @abstractmethod
    def seed(self, seed) -> None:
        """

        :param seed:
        :return:
        """
        pass
