import gym
import os


class EnvironmentInitializer:

    def __init__(self, env_name, **kwargs):
        self._env = None

        if env_name == 'f110':
            self._create_env = self._create_env_f110
        else:
            raise NotImplementedError

        self._create_env(**kwargs)

    @property
    def env(self):
        return self._env

    @abstractmethod
    def _create_env_abstract(self, **kwargs):
        pass

    def _create_env_f110(self, **kwargs):
        path = os.path.abspath("../misc/maps")
        path = ''.join([path, "/", kwargs["map_name"]])
        self._env = gym.make('f110_gym:f110-v0', seed=kwargs["seed"], map=path, map_ext=kwargs["map_ext"],
                             num_agents=1, timestep=0.01)
