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

LEARNING_RATE = 1e-4
GAMMA = 0.99
MEMORY_SIZE = 2e20
EPSILON_START = 1
EPSILON_DECAY = 0.99
EPSILON_MEMORY = 0.01
BATCH_SIZE = 128
ALPHA = 0.6
BETA = 0.4
BETA_INCREMENT = 0.01
NN_SIZES = [1, 1, 1, 1]
NN_TYPE = "cnn"             # cnn or mlp
