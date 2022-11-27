from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.tensorboard import SummaryWriter


from jarmuiranyitas_1.agents.agent import Agent
from jarmuiranyitas_1.agents.misc.memory.per_memory import PERMemory
from jarmuiranyitas_1.agents.misc.memory.r_memory import RMemory
from jarmuiranyitas_1.agents.misc.misc.epsilon_greedy import EGreedy
from jarmuiranyitas_1.agents.misc.neural_network.nn_initializer import NNInitializer


class DQNAgentWithPER(Agent):
    def __init__(self,
                 learning_rate: float = 1e-4,
                 discount_factor: float = 0.99,
                 batch_size: int = 128,
                 network_size: list = None,
                 network_type: str = "mlp",
                 seed: int = 0,
                 epsilon_start: float = 1,
                 epsilon_decay: float = 0.99,
                 memory_size: int = 1e20,
                 memory_alpha: float = 0.6,
                 memory_beta: float = 0.4,
                 memory_beta_increment: float = 0.0005,
                 memory_epsilon: float = 0.01,
                 per: bool = False,
                 ) -> None:

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = 0

        nn_initializer = NNInitializer(network_type, network_size, seed)
        # self.action_network = nn_initializer.generate_nn()
        # self.target_network = nn_initializer.generate_nn()
        self.action_network = torch.load("../experiments/models/f110_2.pth")
        self.target_network = torch.load("../experiments/models/f110_2.pth")
        self.update_networks()

        self.optimizer = optimizer.Adam(self.action_network.nn_parameters, lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.summary_writer = SummaryWriter(log_dir="../experiments/logs")
        self.e_greedy = EGreedy(epsilon_start, epsilon_decay, seed, network_size[-1])
        if per:
            self.memory = PERMemory(memory_size, memory_alpha, memory_beta, memory_beta_increment, memory_epsilon, seed)
        else:
            self.memory = RMemory(memory_size, seed)

        self.seed(seed)

    def seed(self, seed) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def inference(self, state: np.ndarray) -> np.ndarray:
        action = self.e_greedy.choose_action()
        if action is None:
            state = torch.FloatTensor(state).detach()
            self.action_network.eval()
            with torch.no_grad():
                action = self.action_network(state)
            self.action_network.train()
            action = torch.argmax(action)

        return action

    def fit(self) -> None:
        size = min(self.batch_size, len(self.memory))

        sample = self.memory.sample(size)
        mini_batch = self.memory.MEMORY_ITEM(*zip(*sample))

        state_batch = torch.reshape(torch.FloatTensor(np.array(mini_batch.state)), (size, -1)).detach()
        action_batch = torch.reshape(torch.LongTensor(np.array(mini_batch.action)), (size, -1)).detach()
        next_state_batch = torch.reshape(torch.FloatTensor(np.array(mini_batch.next_state)), (size, -1)).detach()
        reward_batch = torch.reshape(torch.FloatTensor(np.array(mini_batch.reward)), (size, -1)).detach()
        done_batch = torch.reshape(torch.FloatTensor(np.array(mini_batch.done)), (size, -1)).detach()

        with torch.no_grad():
            output_next_state_batch = self.target_network(next_state_batch).detach()
            output_next_state_batch = torch.max(output_next_state_batch, 1)[0].detach()
            output_next_state_batch = torch.reshape(output_next_state_batch, (size, -1)).detach()

        y_batch = reward_batch + self.discount_factor * output_next_state_batch * (1 - done_batch)

        output = torch.reshape(self.action_network(state_batch), (size, -1))
        q_values = torch.gather(output, 1, action_batch)

        loss = self.criterion(q_values, y_batch)
        self.loss = float(loss)

        self.optimizer.zero_grad()

        loss.backward()

        for param in self.action_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()



    def update_networks(self):
        self.target_network.load_state_dict(OrderedDict(self.action_network.state_dict()))

    def save_experience(self, **kwargs):
        self.memory.save(**kwargs)

    def write(self, subject: str, value: float, e):
        self.summary_writer.add_scalar(subject, value, e)
