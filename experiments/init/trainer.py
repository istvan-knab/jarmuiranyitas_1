import sys
import numpy as np
import torch


class Trainer:
    def __init__(self, agent, env, name):
        self.env = env
        self.name = name
        self.agent = agent

    def train(self, episodes):

        for e in range(episodes):
            state = self.env.reset(np.array([0.0, 0.0, 1.5]))
            state.reshape((1, -1))

            while True:
                self.env.render()

                action = self.agent.inference(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state.reshape((1, -1))

                self.agent.save_transition(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state

                if done:
                    self.agent.fit()
                    break

        torch.save(self.agent.action_net, "../models/f110_" + ".pth")
        sys.exit()
