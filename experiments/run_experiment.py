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
    arg_parser.add_argument("--seed", default=SEED, type=int)
    arg_parser.add_argument("--map_name", default=MAP_NAME, type=str)
    arg_parser.add_argument("--training_name", default=TRAINING_NAME, type=str)
    arg_parser.add_argument("--agent", default=AGENT, type=str)
    arg_parser.add_argument("--max_episodes", default=MAX_EPISODES, type=int)
    arg_parser.add_argument("--map_ext", default=MAP_EXT, type=str)
    args = arg_parser.parse_args()

    env_initializer = EnvironmentInitializer(args.seed, args.map_name, args.map_ext)
    agent_initializer = AgentInitializer(args.seed, args.agent)
    trainer = Trainer(args.training_name, env_initializer.env, agent_initializer.agent, args.max_episodes)

    trainer.train()


if __name__ == "__main__":
    main()
