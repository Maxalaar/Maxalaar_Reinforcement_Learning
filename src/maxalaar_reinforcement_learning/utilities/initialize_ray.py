import os
from typing import Optional

import ray

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration


def initialize_ray(experimentation_configuration: ExperimentationConfiguration, local_mode: Optional[bool] = None):
    os.environ.update(experimentation_configuration.ray_environment_variables)

    system_config = None
    if experimentation_configuration.ray_bind_to_localhost:
        system_config = {
            'health_check_period_ms': experimentation_configuration.ray_health_check_period_ms,
            'health_check_failure_threshold': experimentation_configuration.ray_health_check_failure_threshold,
        }

    ray.init(
        local_mode=experimentation_configuration.ray_local_mode if local_mode is None else local_mode,
        _temp_dir=experimentation_configuration.ray_temporary_directory,
        _node_ip_address='127.0.0.1' if experimentation_configuration.ray_bind_to_localhost else None,
        _system_config=system_config,
        runtime_env={'env_vars': experimentation_configuration.ray_environment_variables},
    )
