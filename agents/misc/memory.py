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
    def __init__(self, size: int) -> None:
        """

        :param size: size of the memory buffer
        :return: None
        """
        pass

    @abstractmethod
    def save(self, experience: np.ndarray) -> None:
        """

        :param experience:
        :return: None
        """
        pass

    @abstractmethod
    def sample(self):
        """

        :return: sampled experience
        """
        pass
