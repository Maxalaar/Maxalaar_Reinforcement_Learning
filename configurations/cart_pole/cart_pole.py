from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration


cart_pole = ExperimentationConfiguration(
    experimentation_name='cart_pole',
    environment_name='CartPoleRllib',
)

cart_pole.reinforcement_learning_configuration.number_gpus_per_learner = 0

cart_pole.reinforcement_learning_configuration.number_environment_runners = 1
cart_pole.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1
