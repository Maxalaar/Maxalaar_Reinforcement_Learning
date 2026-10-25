from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Callable, Dict, Set, Type, Union
import yaml

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.from_config import NotProvided, _NotProvided
from ray.rllib.utils.typing import LearningRateOrSchedule
from torch.optim import Optimizer

from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO


class ReinforcementLearningConfiguration:
    def __init__(self, algorithm_name: str = 'PPO'):
        # Generale
        self.training_name: str = 'base'
        self.algorithm_name: str = algorithm_name

        # Framework
        self.framework: str = 'torch'

        # Resources
        self.number_gpu: int | _NotProvided = NotProvided

        # Environment
        self.normalize_actions: bool | _NotProvided = NotProvided
        self.disable_environment_checking: bool | _NotProvided = NotProvided

        # Training
        self.gamma: float | _NotProvided = NotProvided
        self.flatten_observations: bool = False
        self.architecture: Optional[Type[RLModule]] = None
        self.architecture_configuration: Union[DefaultModelConfig, Dict, None] = None
        self.train_batch_size: int | _NotProvided = NotProvided
        self.learning_rate: float | _NotProvided = NotProvided
        self.exploration_configuration: dict | _NotProvided = NotProvided
        self.learner_connector: Callable | _NotProvided = NotProvided
        self.number_epochs: int | _NotProvided = NotProvided
        self.gradient_clip: float | _NotProvided = NotProvided
        self.gradient_clip_by: str | _NotProvided = NotProvided
        self.optimizer: Dict | _NotProvided = NotProvided

        # Multi Agent
        self.is_multi_agents: bool = False
        self.policies: Optional[Set[str]] = None
        self.policy_mapping_function: Optional[Callable] = None
        # What train_batch_size and rollout_fragment_length count: 'env_steps' (RLlib's default) or 'agent_steps'.
        # With a number of agents that varies (agents that die), only 'agent_steps' with batch_mode
        # 'truncate_episodes' gives batches of a fixed size (to a step of every agent per env runner).
        self.count_steps_by: str | _NotProvided = NotProvided

        # DQN only
        self.tau: float | _NotProvided = NotProvided
        self.target_network_update_frequency: int | _NotProvided = NotProvided
        self.training_intensity: int | _NotProvided = NotProvided
        self.replay_buffer_configuration: dict | _NotProvided = NotProvided
        self.epsilon: LearningRateOrSchedule | _NotProvided = NotProvided
        self.number_steps_sampled_before_learning_starts: int | _NotProvided = NotProvided

        self.number_step_return: int | _NotProvided = NotProvided
        self.use_noisy_exploration: bool | _NotProvided = NotProvided
        self.use_dueling_dqn: bool | _NotProvided = NotProvided
        self.use_double_q_function: bool | _NotProvided = NotProvided

        # PPO only
        self.use_generalized_advantage_estimator: bool | _NotProvided = NotProvided
        self.minibatch_size: int | _NotProvided = NotProvided
        self.lambda_gae: float | _NotProvided = NotProvided
        self.clip_policy_parameter: float | _NotProvided = NotProvided
        self.clip_value_function_parameter: float | _NotProvided = NotProvided
        self.entropy_coefficient: float | _NotProvided = NotProvided
        self.use_kullback_leibler_loss: bool | _NotProvided = NotProvided
        self.kullback_leibler_coefficient: float | _NotProvided = NotProvided
        self.kullback_leibler_target: float | _NotProvided = NotProvided
        self.value_function_loss_coefficient: float | _NotProvided = NotProvided

        # Environment runners
        self.batch_mode: str | _NotProvided = NotProvided  # 'complete_episodes' or 'truncate_episodes'
        self.rollout_fragment_length: int | _NotProvided = NotProvided
        self.number_environment_runners: int | _NotProvided = NotProvided
        self.number_environment_per_environment_runners: int | _NotProvided = NotProvided
        self.number_cpus_per_environment_runners: int | _NotProvided = NotProvided
        self.number_gpus_per_environment_runners: float | _NotProvided = NotProvided
        self.compress_observations: bool | _NotProvided = NotProvided
        self.sample_timeout_s: int | _NotProvided = NotProvided

        # Learners
        self.number_learners: int | _NotProvided = NotProvided
        self.number_cpus_per_learner: int | _NotProvided = NotProvided
        self.number_gpus_per_learner: int | _NotProvided = NotProvided

        # Evaluation
        self.evaluation_interval: int = 50
        self.evaluation_num_environment_runners: int | _NotProvided = NotProvided
        self.evaluation_duration_unit: str | _NotProvided = NotProvided
        self.evaluation_duration: int = 100
        self.evaluation_parallel_to_training: bool | _NotProvided = NotProvided
        self.evaluation_sample_timeout_s: int | _NotProvided = NotProvided

        # Callbacks
        self.callback: DefaultCallbacks | _NotProvided = NotProvided
        self.use_wandb: bool = False

        # Stopping Criterion
        self.stopping_criterion: Optional[dict] = None

        # Initialization from previous checkpoint (transfer learning)
        self.initialization_checkpoint_path: Optional[Path] = None

        # Checkpoint Configuration
        self.number_checkpoint_to_keep: Optional[int] = 10
        self.checkpoint_score_attribute: Optional[str] = 'evaluation/env_runners/episode_return_mean'
        self.checkpoint_score_order: str = 'max'
        self.checkpoint_frequency: int = 50
        self.checkpoint_at_end: Optional[bool] = None

    def to_yaml_file(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'reinforcement_learning_configuration.yaml'
        configuration_dictionary = {key: value for key, value in self.__dict__.items() if value is not NotProvided}

        with open(file_path, 'w') as file:
            yaml.dump(configuration_dictionary, file)

    def clone(self):
        deep_copy = deepcopy(self)

        for attribute_name in dir(deep_copy):
            if not attribute_name.startswith('_'):
                attribute_value = getattr(deep_copy, attribute_name)
                if type(attribute_value) == type(NotProvided):
                    setattr(deep_copy, attribute_name, NotProvided)

        return deep_copy






