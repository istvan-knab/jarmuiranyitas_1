from argparse import ArgumentParser

from jarmuiranyitas_1.experiments.init.trainer import Trainer
from jarmuiranyitas_1.experiments.init.agent_initializer import AgentInitializer
from jarmuiranyitas_1.experiments.init.environment_initializer import EnvironmentInitializer


# Defaults
SEED = 0
MAP_NAME = 'xyz_palya'
TRAINING_NAME = 'Training_results_' + MAP_NAME + '_' + str(SEED)
AGENT = 'dqn'
MAX_EPISODES = 1000000
MAP_EXT = '.png'
ENV_NAME = 'f110'


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--seed", default=SEED, type=int)
    arg_parser.add_argument("--map_name", default=MAP_NAME, type=str)
    arg_parser.add_argument("--training_name", default=TRAINING_NAME, type=str)
    arg_parser.add_argument("--agent", default=AGENT, type=str)
    arg_parser.add_argument("--max_episodes", default=MAX_EPISODES, type=int)
    arg_parser.add_argument("--map_ext", default=MAP_EXT, type=str)
    arg_parser.add_argument("--env_name", default=ENV_NAME, type=str)
    args = arg_parser.parse_args()

    env_initializer = EnvironmentInitializer(env_name=args.env_name, seed=args.seed, map_name=args.map_name,
                                             map_ext=args.map_ext)
    agent_initializer = AgentInitializer(args.seed, args.agent)
    trainer = Trainer(agent_initializer.agent, env_initializer.env, args.training_name)

    trainer.train(args.max_episodes)


if __name__ == "__main__":
    main()
