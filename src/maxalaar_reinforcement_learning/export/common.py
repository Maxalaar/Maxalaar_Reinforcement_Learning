from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule


def actor_initial_state(rl_module: RLModule) -> Optional[Dict[str, np.ndarray]]:
    initial_state = rl_module.get_initial_state()
    if not initial_state:
        return None
    return initial_state.get('actor', initial_state)


def dummy_inputs(rl_module: RLModule, batch_size: int) -> Tuple[List[str], Tuple[torch.Tensor, ...]]:
    model_device = next(rl_module.parameters()).device
    dummy_obs = torch.stack([
        torch.as_tensor(rl_module.observation_space.sample(), dtype=torch.float32)
        for _ in range(batch_size)
    ], dim=0).to(model_device)

    initial_state = actor_initial_state(rl_module)
    if initial_state is None:
        return ['obs'], (dummy_obs,)

    cell_size = initial_state['h'].shape[0]
    dummy_h = torch.zeros(batch_size, cell_size, device=model_device)
    dummy_c = torch.zeros(batch_size, cell_size, device=model_device)
    return ['obs', 'state_in_h', 'state_in_c'], (dummy_obs, dummy_h, dummy_c)
