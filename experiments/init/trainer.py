import sys
import numpy as np
import torch
from collections import deque


class Trainer:
    def __init__(self, agent, env, name):
        self.env = env
        self.name = name
        self.agent = agent

    def train(self, episodes):
        torch.set_flush_denormal(True)
        steering = [-0.4, 0.0, 0.4]
        reward_over_100_episode = deque(maxlen=100)

        for e in range(episodes):
            state, _, _, _ = self.env.reset(np.array([[0, 0, 3.0]]))
            state = state['scans'][0][180:900:24]
            state /= 4.7
            reward_sum = 0
            current_lap = 0
            time = 0

            while True:

                action = self.agent.inference(state)
                # self.env.render()

                next_state, reward, done, _ = self.env.step(np.array([[steering[action], 1.0]]))
                lap_count = next_state['lap_counts']
                next_state = next_state['scans'][0][180:900:24]
                next_state /= 4.7

                if done:
                    reward = - 1

                else:
                    min_dist = min(next_state)
                    min_dist_index = np.argmin(next_state)
                    if 0 < min_dist_index < len(next_state) - 1:
                        min_plus_1_angle = np.deg2rad((min_dist_index + 1) * 6)
                        min_minus_1_angle = np.deg2rad((min_dist_index - 1) * 6)
                        x_1 = np.cos(min_plus_1_angle) * next_state[min_dist_index + 1]
                        x_2 = np.cos(min_minus_1_angle) * next_state[min_dist_index - 1]
                        y_1 = np.sin(min_plus_1_angle) * next_state[min_dist_index + 1]
                        y_2 = np.sin(min_minus_1_angle) * next_state[min_dist_index - 1]
                    elif min_dist_index == 0:
                        min_plus_1_angle = np.deg2rad(6)
                        x_1 = np.cos(min_plus_1_angle) * next_state[1]
                        x_2 = next_state[0]
                        y_1 = np.sin(min_plus_1_angle) * next_state[1]
                        y_2 = 0
                    else:
                        min_minus_1_angle = np.deg2rad(174)
                        x_1 = - next_state[min_dist_index]
                        x_2 = np.cos(min_minus_1_angle) * next_state[min_dist_index - 1]
                        y_1 = 0
                        y_2 = np.sin(min_minus_1_angle) * next_state[min_dist_index - 1]

                    delta_x = abs(x_1 - x_2)
                    delta_y = abs(y_1 - y_2)
                    relative_yaw_angle = np.pi / 2 - np.arctan(delta_y / delta_x)
                    reward1 = np.interp(np.cos(relative_yaw_angle), (0.95, 1.0), (- 0.5, 0.5))
                    reward2 = np.interp(min_dist, (0.08, 0.1025), (-0.5, 0.5))

                    reward = reward1 + reward2

                if lap_count > current_lap_count:
                    reward += 100
                    current_lap_count += 1

                reward_sum += reward
                time += 1

                self.agent.save_experience(state=state, action=action, next_state=next_state, reward=reward, done=done)
                state = next_state

                if time % 5 == 0:
                    self.agent.fit()

                if done:
                    self.agent.e_greedy.update_epsilon()
                    reward_over_100_episode.append(reward_sum)
                    if e % 1 == 0:
                        self.agent.update_networks()

                    self.agent.write('Reward per episode', reward_sum, e+1)
                    self.agent.write('Epsilon', self.agent.e_greedy.current_epsilon, e+1)
                    self.agent.write('Loss', self.agent.loss, e+1)

                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(reward_over_100_episode)), end="")
                    if e % 100 == 0:
                        print('\rEpisode {}\tAverage Score: {:.2f}\tCurrent epsilon: {:.8f}'.format
                              (e, np.mean(reward_over_100_episode), self.agent.e_greedy.current_epsilon))
                        torch.save(self.agent.action_network, "models/f110_" + ".pth")
                    break

        sys.exit()
