from collections import OrderedDict
from typing import Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer

from jarmuiranyitas_1.agents.agent import Agent
from jarmuiranyitas_1.agents.misc.per_memory import PERMemory
from jarmuiranyitas_1.agents.misc.per_memory import EGreedy
from jarmuiranyitas_1.agents.misc.per_memory import NNInitializer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

    def inference(self, state: np.ndarray) -> np.ndarray:
        pass

    def fit(self) -> None:
        pass
