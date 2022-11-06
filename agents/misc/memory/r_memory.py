import random
from collections import deque, namedtuple

from jarmuiranyitas_1.agents.misc.memory.memory import Memory


class RMemory(Memory):

    MEMORY_ITEM = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def __init__(self, size: int, seed: int) -> None:
        self.memory = deque(maxlen=size)
        self.seed(seed)

    def __len__(self) -> int:
        return len(self.memory)

    def save(self, **kwargs) -> None:
        experience = self.MEMORY_ITEM(kwargs["state"], kwargs["action"], kwargs["next_state"], kwargs["reward"],
                                      kwargs["done"])

        self.memory.append(experience)

    def sample(self, size: int) -> list:
        return random.sample(self.memory, size)

    def seed(self, seed: int) -> None:
        random.seed(seed)
