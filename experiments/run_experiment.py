from argparse import ArgumentParser

from jarmuiranyitas_1.experiments.trainer import Trainer
from jarmuiranyitas_1.experiments.agent_initializer import AgentInitializer
from jarmuiranyitas_1.experiments.environment_initializer import EnvironmentInitializer


# Defaults
SEED = None
MAP_NAME = 'xyz_palya'
TRAINING_NAME = 'Training_results_' + MAP_NAME + '_' + str(SEED)
AGENT = 'dqn'
MAX_EPISODES = 1e6
MAP_EXT = '.png'


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--seed", default=None, type=int)
    arg_parser.add_argument("--map_name", default='xyz_palya', type=str)
    arg_parser.add_argument("--training_name", default='Training_results', type=str)
    arg_parser.add_argument("--agent", default='dqn', type=str)
    arg_parser.add_argument("--max_episodes", default=1e6, type=int)
    args = arg_parser.parse_args()

    env_initializer = EnvironmentInitializer(args.seed, args.map_name)
    agent_initializer = AgentInitializer(args.seed, args.agent)
    trainer = Trainer(args.training_name, env_initializer.env, agent_initializer.agent, args.max_episodes)

    trainer.train()


if __name__ == "__main__":
    main()
