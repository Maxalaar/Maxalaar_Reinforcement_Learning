from pathlib import Path
from typing import Tuple, Type

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.appo import APPOConfig, APPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig, Algorithm
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration


def build_algorithm_configuration(
        experimentation_configuration: ExperimentationConfiguration,
) -> Tuple[Type[Algorithm], AlgorithmConfig]:
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration

    algorithm: Algorithm
    algorithm_configuration: AlgorithmConfig

    if reinforcement_learning_configuration.algorithm_name == 'PPO':
        algorithm = PPO
        algorithm_configuration = PPOConfig()
    elif reinforcement_learning_configuration.algorithm_name == 'APPO':
        algorithm = APPO
        algorithm_configuration = APPOConfig()
    elif reinforcement_learning_configuration.algorithm_name == 'DQN':
        algorithm = DQN
        algorithm_configuration = DQNConfig()
    elif reinforcement_learning_configuration.algorithm_name == 'DreamerV3':
        from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
        algorithm = DreamerV3
        algorithm_configuration = DreamerV3Config()
    else:
        raise ValueError('Unsupported algorithm ' + str(reinforcement_learning_configuration.algorithm_name) + '.')

    algorithm_configuration.framework(reinforcement_learning_configuration.framework)

    # Environment
    algorithm_configuration.environment(
        env=experimentation_configuration.environment_name,
        env_config=experimentation_configuration.environment_configuration,
        normalize_actions=reinforcement_learning_configuration.normalize_actions,
        disable_env_checking=reinforcement_learning_configuration.disable_environment_checking,
    )

    # Reinforcement Learning Module
    policy_spec_kwargs = {}
    if reinforcement_learning_configuration.architecture is not None:
        policy_spec_kwargs['module_class'] = reinforcement_learning_configuration.architecture
    if reinforcement_learning_configuration.architecture_configuration is not None:
        policy_spec_kwargs['model_config'] = reinforcement_learning_configuration.architecture_configuration

    if reinforcement_learning_configuration.initialization_checkpoint_path is not None:
        rl_module_path = Path(reinforcement_learning_configuration.initialization_checkpoint_path) / 'learner_group' / 'learner' / 'rl_module'
        if not rl_module_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {rl_module_path}")
        if reinforcement_learning_configuration.is_multi_agents and reinforcement_learning_configuration.policies:
            # Multi-agent: each policy loads its weights from its subfolder
            per_policy_specs = {}
            for policy_name in reinforcement_learning_configuration.policies:
                policy_checkpoint_path = rl_module_path / policy_name
                if not policy_checkpoint_path.exists():
                    raise FileNotFoundError(f"Policy '{policy_name}' checkpoint not found: {policy_checkpoint_path}")
                per_policy_specs[policy_name] = RLModuleSpec(
                    **policy_spec_kwargs,
                    load_state_path=str(policy_checkpoint_path),
                )
            algorithm_configuration.rl_module(
                rl_module_spec=MultiRLModuleSpec(rl_module_specs=per_policy_specs),
            )
        else:
            # Single-agent: load directly from rl_module
            algorithm_configuration.rl_module(
                rl_module_spec=RLModuleSpec(**policy_spec_kwargs, load_state_path=str(rl_module_path)),
            )
    elif policy_spec_kwargs:
        algorithm_configuration.rl_module(
            rl_module_spec=RLModuleSpec(**policy_spec_kwargs),
        )

    # Multi Agent
    if reinforcement_learning_configuration.is_multi_agents:
        algorithm_configuration.multi_agent(
            policies=reinforcement_learning_configuration.policies,
            policy_mapping_fn=reinforcement_learning_configuration.policy_mapping_function,
        )

    # Training
    algorithm_configuration.training(
        gamma=reinforcement_learning_configuration.gamma,
        train_batch_size_per_learner=reinforcement_learning_configuration.train_batch_size,
        lr=reinforcement_learning_configuration.learning_rate,
        learner_connector=reinforcement_learning_configuration.learner_connector,
        num_epochs=reinforcement_learning_configuration.number_epochs,
        grad_clip=reinforcement_learning_configuration.gradient_clip,
        grad_clip_by=reinforcement_learning_configuration.gradient_clip_by,
        optimizer=reinforcement_learning_configuration.optimizer,
    )
    if reinforcement_learning_configuration.exploration_configuration is not NotProvided:
        algorithm_configuration.exploration_config = reinforcement_learning_configuration.exploration_configuration

    if isinstance(algorithm_configuration, PPOConfig):
        algorithm_configuration.training(
            use_gae=reinforcement_learning_configuration.use_generalized_advantage_estimator,
            minibatch_size=reinforcement_learning_configuration.minibatch_size,
            lambda_=reinforcement_learning_configuration.lambda_gae,
            grad_clip=reinforcement_learning_configuration.gradient_clip,
            clip_param=reinforcement_learning_configuration.clip_policy_parameter,
            vf_clip_param=reinforcement_learning_configuration.clip_value_function_parameter,
            learner_connector=reinforcement_learning_configuration.learner_connector,
            entropy_coeff=reinforcement_learning_configuration.entropy_coefficient,
            use_kl_loss=reinforcement_learning_configuration.use_kullback_leibler_loss,
            kl_coeff=reinforcement_learning_configuration.kullback_leibler_coefficient,
            kl_target=reinforcement_learning_configuration.kullback_leibler_target,
            vf_loss_coeff=reinforcement_learning_configuration.value_function_loss_coefficient,
        )

    if isinstance(algorithm_configuration, DQNConfig):
        algorithm_configuration.training(
            replay_buffer_config=reinforcement_learning_configuration.replay_buffer_configuration,
            epsilon=reinforcement_learning_configuration.epsilon,
            target_network_update_freq=reinforcement_learning_configuration.target_network_update_frequency,
            training_intensity=reinforcement_learning_configuration.training_intensity,
            num_steps_sampled_before_learning_starts=reinforcement_learning_configuration.number_steps_sampled_before_learning_starts,
            n_step=reinforcement_learning_configuration.number_step_return,
            noisy=reinforcement_learning_configuration.use_noisy_exploration,
            dueling=reinforcement_learning_configuration.use_dueling_dqn,
            double_q=reinforcement_learning_configuration.use_double_q_function,
        )

    # Environment runners
    algorithm_configuration.env_runners(
        batch_mode=reinforcement_learning_configuration.batch_mode,
        rollout_fragment_length=reinforcement_learning_configuration.rollout_fragment_length,
        num_env_runners=reinforcement_learning_configuration.number_environment_runners,
        num_envs_per_env_runner=reinforcement_learning_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=reinforcement_learning_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=reinforcement_learning_configuration.number_gpus_per_environment_runners,
        compress_observations=reinforcement_learning_configuration.compress_observations,
        sample_timeout_s=reinforcement_learning_configuration.sample_timeout_s,
    )

    if reinforcement_learning_configuration.flatten_observations:
        algorithm_configuration.env_runners(
            env_to_module_connector=lambda env, spaces, device: FlattenObservations(
                multi_agent=reinforcement_learning_configuration.is_multi_agents,
            ),
        )

    # Learners
    algorithm_configuration.learners(
        num_learners=reinforcement_learning_configuration.number_learners,
        num_cpus_per_learner=reinforcement_learning_configuration.number_cpus_per_learner,
        num_gpus_per_learner=reinforcement_learning_configuration.number_gpus_per_learner,
    )

    # Evaluation
    algorithm_configuration.evaluation(
        evaluation_interval=reinforcement_learning_configuration.evaluation_interval,
        evaluation_num_env_runners=reinforcement_learning_configuration.evaluation_num_environment_runners,
        evaluation_duration=reinforcement_learning_configuration.evaluation_duration,
        evaluation_parallel_to_training=reinforcement_learning_configuration.evaluation_parallel_to_training,
        evaluation_sample_timeout_s=reinforcement_learning_configuration.evaluation_sample_timeout_s,
    )

    # Callbacks
    if reinforcement_learning_configuration.callback is not NotProvided:
        algorithm_configuration.callbacks(reinforcement_learning_configuration.callback)

    # New API Stack
    algorithm_configuration.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    return algorithm, algorithm_configuration
