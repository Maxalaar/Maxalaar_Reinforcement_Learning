from pathlib import Path
import pickle
from ray.rllib.algorithms import Algorithm, PPOConfig, DQNConfig, PPO, DQN
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config, DreamerV3


def get_checkpoint_algorithm_configuration(checkpoint_path: Path):
    with open(checkpoint_path / 'class_and_ctor_args.pkl', 'rb') as file:
        ctor_info = pickle.load(file)
        algorithm_class = ctor_info['class']
        algorithm_configuration_dictionary = ctor_info['ctor_args_and_kwargs'][0][0]

    if algorithm_class.__name__ == PPO.__name__:
        algorithm_configuration: PPOConfig = PPOConfig.from_dict(algorithm_configuration_dictionary)
    elif algorithm_class.__name__ == DQN.__name__:
        algorithm_configuration: DQNConfig = DQNConfig.from_dict(algorithm_configuration_dictionary)
    elif algorithm_class.__name__ == DreamerV3.__name__:
        algorithm_configuration: DreamerV3Config = DreamerV3Config.from_dict(algorithm_configuration_dictionary)
    else:
        raise ValueError('Unsupported algorithm ' + str(algorithm_class) + '.')
    algorithm_configuration.algo_class = algorithm_class

    return algorithm_configuration
