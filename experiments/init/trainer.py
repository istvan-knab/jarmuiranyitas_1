import sys
import numpy as np
import torch


class Trainer:
    def __init__(self, agent, env, name):
        self.env = env
        self.name = name
        self.agent = agent

    def train(self, episodes):
        steering = [-0.4, -0.2, 0.0, 0.2, 0.4]

        for e in range(episodes):
            state, _, _, _ = self.env.reset(np.array([[0.0, 0.0, 2.8]]))

            while True:
                self.env.render()

                action = self.agent.inference(state['scans'][0])

                next_state, reward, done, _ = self.env.step(np.array([[steering[action], 1.0]]))

                self.agent.save_experience(state=state['scans'][0], action=action, next_state=next_state['scans'][0],
                                           reward=reward, done=done)
                state = next_state

                if done:
                    self.agent.fit()
                    print("Episode: {0}/{1}\nCurrent epsilon: {2}".format(e, episodes,
                                                                          self.agent.e_greedy.current_epsilon))

                    if e % 1000 == 0:
                        torch.save(self.agent.action_network, "models/f110_" + ".pth")
                    break

        sys.exit()
