from pathlib import Path
from typing import Dict, List, Optional


class DisplayEpisodesConfiguration:
    force_ray_local_mode: bool = False
    working_directory: Optional[Path] = None
    render_mode: str = 'rgb_array'
    explore: bool = False
    use_neural_network_visualizer: bool = False
    max_connections_per_layer = 30
    policies_name: List[str] = []
    policies_name_to_batch_indices_to_visualize: Optional[Dict[str, int]] = None
    environment_configuration: Dict = {}
    checkpoint_score_attribute: Optional[str] = None
    checkpoint_score_order: Optional[str] = None
    num_gpus_per_env_runner: int = 1
    log_observations: bool = False
    log_actions: bool = False
    log_max_steps: Optional[int] = None