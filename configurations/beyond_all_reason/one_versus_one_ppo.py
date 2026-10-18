"""PPO training configuration — 1v1 minimal combat (curriculum step 1).

Scenario (beyond_all_reason_environment.scenarios.one_versus_one): flat 1024×1024-elmo arena, 1 armpw (Pawn) on
each side, deterministic spawns at opposite edges.
Chosen here, for this experiment: the agent against a Passive Pawn (does not move, does not shoot), one weapon slot
per unit, every command allowed, and a terminal reward (+1 victory, -1 defeat, no intermediate shaping). Entity
counts follow the scenario (no ally, one enemy).

Expected episode length: ~112 decisions (2 game-minutes at 32 frames/decision).
Measured throughput: ~160 decisions/s per env runner at engine speed 1000.

Usage (from this project root, in a conda environment where beyond_all_reason_environment is installed):
    python scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/beyond_all_reason/one_versus_one_ppo.py
"""

from pathlib import Path

import torch.nn as nn

from beyond_all_reason_environment import (
    ActionConfiguration,
    Agent,
    GameConfiguration,
    ObservationConfiguration,
    Passive,
    ReinforcementLearningEnvironmentConfiguration,
    RewardWeights,
)
from beyond_all_reason_environment.scenarios import one_versus_one
from maxalaar_reinforcement_learning.architectures.beyond_all_reason.beyond_all_reason_combat_rl_module import (
    BeyondAllReasonCombatRLModule,
)
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

one_versus_one_ppo = ExperimentationConfiguration(
    experimentation_name="beyond_all_reason/one_versus_one_ppo",
    environment_name="BeyondAllReasonCombat",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=one_versus_one, controllers=[Agent(), Passive()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# ── Architecture ──────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.architecture = BeyondAllReasonCombatRLModule
one_versus_one_ppo.reinforcement_learning_configuration.architecture_configuration = {
    # Actor
    "d_level1": 64,
    "d_level2": 64,
    "d_level3": 64,
    "entity_projector_hidden_sizes": [64],
    "weapon_projector_hidden_sizes": [32],
    "number_entity_attention_heads": 1,
    "number_category_attention_heads": 1,
    "number_global_attention_heads": 1,
    "number_global_attention_layers": 1,
    "global_feedforward_size": 128,
    "context_projector_hidden_sizes": [],
    "terrain_cnn_channels": [16, 32, 32],
    "action_hidden_sizes": [64, 32],
    "activation_function_class": nn.ReLU,
    # Centralized critic
    "critic_d": 64,
    "critic_entity_projector_hidden_sizes": [64],
    "critic_weapon_projector_hidden_sizes": [32],
    "critic_number_entity_attention_heads": 1,
    "critic_number_category_attention_heads": 1,
    "critic_number_global_attention_heads": 1,
    "critic_number_global_attention_layers": 1,
    "critic_global_feedforward_size": 128,
    "critic_context_projector_hidden_sizes": [],
    "critic_terrain_cnn_channels": [16, 32, 32],
}

# ── Multi-agent: 1 agent per episode, shared policy ───────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.is_multi_agents = True
one_versus_one_ppo.reinforcement_learning_configuration.policies = {"shared_policy"}
one_versus_one_ppo.reinforcement_learning_configuration.policy_mapping_function = (
    lambda agent_id, episode, **kwargs: "shared_policy"
)

# ── Environment ───────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.disable_environment_checking = True
one_versus_one_ppo.reinforcement_learning_configuration.normalize_actions = False
one_versus_one_ppo.reinforcement_learning_configuration.flatten_observations = False
one_versus_one_ppo.reinforcement_learning_configuration.compress_observations = False

# ── PPO hyperparameters ───────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.training_name = "base"
one_versus_one_ppo.reinforcement_learning_configuration.gamma = 0.99
one_versus_one_ppo.reinforcement_learning_configuration.lambda_gae = 0.95
one_versus_one_ppo.reinforcement_learning_configuration.learning_rate = 3e-4
one_versus_one_ppo.reinforcement_learning_configuration.clip_policy_parameter = 0.2
one_versus_one_ppo.reinforcement_learning_configuration.entropy_coefficient = 0.005
one_versus_one_ppo.reinforcement_learning_configuration.value_function_loss_coefficient = 1.0
one_versus_one_ppo.reinforcement_learning_configuration.number_epochs = 10
one_versus_one_ppo.reinforcement_learning_configuration.train_batch_size = 4000
one_versus_one_ppo.reinforcement_learning_configuration.minibatch_size = 512
one_versus_one_ppo.reinforcement_learning_configuration.batch_mode = "complete_episodes"
one_versus_one_ppo.reinforcement_learning_configuration.sample_timeout_s = 180

# ── Resources ─────────────────────────────────────────────────────────────────
# Each engine process uses ~2.3 CPU cores and ~4.7 GB RAM.
# 4 runners: ~19 GB RAM, ~9-10 cores for engines, comfortably within 31 GB / 20 cores.
one_versus_one_ppo.reinforcement_learning_configuration.number_environment_runners = 4
one_versus_one_ppo.reinforcement_learning_configuration.number_cpus_per_environment_runners = 3
one_versus_one_ppo.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
one_versus_one_ppo.reinforcement_learning_configuration.number_learners = 1
one_versus_one_ppo.reinforcement_learning_configuration.number_cpus_per_learner = 1
one_versus_one_ppo.reinforcement_learning_configuration.number_gpus_per_learner = 0

# ── Evaluation ────────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.evaluation_interval = 20
one_versus_one_ppo.reinforcement_learning_configuration.evaluation_num_environment_runners = 1
one_versus_one_ppo.reinforcement_learning_configuration.evaluation_duration = 5

# ── Stopping criterion ───────────────────────────────────────────────────────
# Set to None for a real training run; keep at a small value for smoke tests.
one_versus_one_ppo.reinforcement_learning_configuration.stopping_criterion = {"training_iteration": 3}

# ── Checkpoints ──────────────────────────────────────────────────────────────
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_frequency = 20
one_versus_one_ppo.reinforcement_learning_configuration.number_checkpoint_to_keep = 10
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_score_attribute = (
    "evaluation/env_runners/episode_return_mean"
)
one_versus_one_ppo.reinforcement_learning_configuration.checkpoint_score_order = "max"
