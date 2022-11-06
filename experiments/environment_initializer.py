import gym
import os


class EnvironmentInitializer:

    def __init__(self, seed, map_name, map_ext):
        self._env = None

        self._create_env(seed, map_name, map_ext)

    @property
    def env(self):
        return self._env

    def _create_env(self, seed, map_name, map_ext):
        path = os.path.abspath("../misc/maps")
        path = ''.join([path, "/", map_name])
        self._env = gym.make('f110_gym:f110-v0', seed=seed, map=path, map_ext=map_ext, num_agents=1)
