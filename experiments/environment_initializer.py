import gym
import os


class EnvironmentInitializer:

    def __init__(self, seed, map_name, map_ext):
        self._env = None

        self.create_env()

    @property
    def env(self):
        return self._env

    def create_env(self):
        self._env = gym.make('f110_gym:f110-v0')
