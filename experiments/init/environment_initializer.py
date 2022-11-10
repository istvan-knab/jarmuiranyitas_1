from jarmuiranyitas_1.f110_env.f110_gym.envs.f110_env import F110Env
import os
from abc import abstractmethod
import yaml
from argparse import Namespace

class EnvironmentInitializer:

    def __init__(self, env_name, **kwargs):
        self._env = None
        self._load_environment_config()

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

    def _load_environment_config(self) -> None:
        config_path = os.path.abspath('init/environment_config.yml')
        with open(config_path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self._config = Namespace(**conf_dict)

    def _create_env_f110(self, **kwargs):


        path = os.path.abspath("../misc/maps")
        path = ''.join([path, "/", kwargs["map_name"]])
        '''
        self._env =  F110Env(self._config.seed,
                             self._config.path,
                             self._config.map_ext,
                             self._config.num_agents,
                             self._config.time_step)
        '''
        self._env = F110Env(seed=kwargs["seed"], map=path, map_ext=kwargs["map_ext"], num_agents=1, timestep=0.01)
