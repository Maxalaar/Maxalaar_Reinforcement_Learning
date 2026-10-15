from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import onnxruntime
import torch
from ray.rllib.core.rl_module import RLModule

from maxalaar_reinforcement_learning.export.common import dummy_inputs


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model, is_stateful: bool):
        super().__init__()
        self.model = model
        self.is_stateful = is_stateful

    def forward(self, obs, state_in_h=None, state_in_c=None):
        obs_dict = {'obs': obs}
        if self.is_stateful:
            obs_dict['state_in_h'] = state_in_h
            obs_dict['state_in_c'] = state_in_c
        return self.model.forward_onnx(obs_dict)


def _output_names(rl_module: RLModule, is_stateful: bool) -> list[str]:
    if isinstance(rl_module.action_space, gym.spaces.Dict):
        names = sorted(rl_module.action_space.spaces.keys())
    else:
        names = ['output']
    if is_stateful:
        names += ['state_out_h', 'state_out_c']
    return names


def verify_onnx_export(onnx_file_path: Path, rl_module: RLModule, batch_size: int = 1024):
    input_names, inputs = dummy_inputs(rl_module, batch_size)
    is_stateful = len(inputs) > 1

    onnx_inputs = {name: tensor.cpu().numpy() for name, tensor in zip(input_names, inputs)}
    onnxruntime_session = onnxruntime.InferenceSession(str(onnx_file_path))
    onnxruntime_outputs = onnxruntime_session.run(None, onnx_inputs)

    with torch.no_grad():
        pytorch_outputs = rl_module.forward_onnx(dict(zip(input_names, inputs)))

    output_names = _output_names(rl_module, is_stateful)
    total_error = 0.0
    for i, name in enumerate(output_names):
        pt = pytorch_outputs[name].cpu().numpy()
        ort = onnxruntime_outputs[i]
        error = np.mean(np.abs(pt - ort))
        total_error += error
        print(f'[{name}] PyTorch: {pt[:2]}  ONNX: {ort[:2]}  error: {error:.6f}')
    print(f'Mean error across all outputs: {total_error / len(output_names):.6f}')


def rl_module_to_onnx(rl_module: RLModule, policy_name: str, onnx_path: Path, file_stem: Optional[str] = None) -> List[Path]:
    """The deterministic action is decoded inside the graph, batch axis dynamic."""
    print('Export policy to ONNX: ' + policy_name)
    rl_module.device = next(rl_module.parameters()).device

    input_names, args = dummy_inputs(rl_module, batch_size=32)
    is_stateful = len(args) > 1
    output_names = _output_names(rl_module, is_stateful)
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names + output_names}

    onnx_file_path: Path = onnx_path / str((file_stem or policy_name) + '.onnx')

    torch.onnx.export(
        OnnxWrapper(rl_module, is_stateful),
        args=args,
        f=onnx_file_path,
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print('Exported ONNX: ' + str(onnx_file_path))
    verify_onnx_export(onnx_file_path=onnx_file_path, rl_module=rl_module)
    print()

    return [onnx_file_path]
