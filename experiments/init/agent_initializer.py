import yaml
from argparse import Namespace

from jarmuiranyitas_1.agents.dqn_agent_with_per import DQNAgentWithPER


class AgentInitializer:
    """
    ------------------------
    Agent Initializer Class
    ------------------------


    Attributes:
        _agent : None -> custom agent class
        _config : None -> Namespace
        _agent_type : str
        _seed : int


    Methods:
        __init__(self, seed, agent_type, *args) -> None
        _load_agent_config(self) -> None
        _choose_agent(self) -> None
    """
    def __init__(self, seed, agent_type):
        self._agent = None

        self._config = None
        self._agent_type = agent_type
        self._seed = seed

        self._load_agent_config()
        self._create_agent()

    @property
    def agent(self):
        return self._agent

    def _load_agent_config(self) -> None:
        """

        :return: None
        """
        with open('agent_config.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self._config = Namespace(**conf_dict)

    def _create_agent(self) -> None:
        """

        :return: None
        """
        if self._agent_type == 'dqn':
            self._agent = DQNAgentWithPER(self._config.dqn_learning_rate,
                                          self._config.dqn_discount_factor,
                                          self._config.dqn_batch_size,
                                          self._config.dqn_network_size,
                                          self._config.dqn_network_type,
                                          self._seed,
                                          self._config.dqn_epsilon_start,
                                          self._config.dqn_epsilon_decay,
                                          self._config.dqn_memory_size,
                                          self._config.dqn_memory_alpha,
                                          self._config.dqn_memory_beta,
                                          self._config.dqn_memory_beta_increment,
                                          self._config.dqn_memory_epsilon,
                                          self._config.dqn_per)
        else:
            raise NotImplementedError("This agent is not yet implemented, please select another one via the 'agent'"
                                      "option!")
