import torch
import numpy as np

from jarmuiranyitas_1.experiments.init.environment_initializer import EnvironmentInitializer


action_net = torch.load("models/f110_2.pth").eval().float()
env_initializer = EnvironmentInitializer(env_name='f110', seed=0, map_name='xyz_palya', map_ext='.png')
env = env_initializer.env

STEERING = [- 1.0, 0.0, 1.0]

for i in range(10):
    done = False
    score = 0

    state, _, _, _ = env.reset(np.array([[0, 1, 3]]))
    state = torch.from_numpy(state['scans'][0][180:900:24])
    state /= 4.7

    while not done:
        env.render()
        action = action_net(state.float())
        action = int(torch.argmax(action))

        next_state, _, done, _ = env.step(np.array([[STEERING[action], 2]]))

        state = next_state['scans'][0][180:900:24]
        state /= 4.7
        state = torch.from_numpy(state)
