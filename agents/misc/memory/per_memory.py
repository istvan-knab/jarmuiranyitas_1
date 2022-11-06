from collections import namedtuple

from jarmuiranyitas_1.agents.misc.memory.memory import Memory


class PERMemory(Memory):
    MEMORY_ITEM = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def __init__(self, size: int, alpha, beta, beta_increment, epsilon, seed) -> None:
        pass

    def __len__(self) -> int:
        pass

    def save(self, **kwargs) -> None:
        pass

    def sample(self, size: int) -> list:
        pass

    def seed(self, seed) -> None:
        pass
