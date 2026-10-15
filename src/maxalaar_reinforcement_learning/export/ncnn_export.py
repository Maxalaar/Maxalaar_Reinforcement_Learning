import copy
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule

from maxalaar_reinforcement_learning.export.common import dummy_inputs


class NcnnWrapper(torch.nn.Module):
    """Blob `in<i>` is the i-th input of `dummy_inputs`, blob `out<i>` the i-th of `_output_names`."""

    def __init__(self, model, is_stateful: bool):
        super().__init__()
        self.model = model
        self.is_stateful = is_stateful

    def forward(self, obs, state_in_h=None, state_in_c=None):
        obs_dict = {'obs': obs}
        if self.is_stateful:
            obs_dict['state_in_h'] = state_in_h
            obs_dict['state_in_c'] = state_in_c
        outputs = self.model.forward_inference_logits(obs_dict)
        return tuple(outputs[name] for name in _output_names(self.is_stateful))


def _output_names(is_stateful: bool) -> list[str]:
    names = ['logits']
    if is_stateful:
        names += ['state_out_h', 'state_out_c']
    return names


def action_layout(action_space: gym.Space) -> List[dict]:
    """Position of each action in the flat logits.
    discrete: `size` logits, action = start + argmax. box: [loc, log_std], action = loc."""
    if isinstance(action_space, gym.spaces.Dict):
        named_spaces = sorted(action_space.spaces.items())
    else:
        named_spaces = [('output', action_space)]

    layout = []
    offset = 0
    for name, space in named_spaces:
        if isinstance(space, gym.spaces.Discrete):
            size = int(space.n)
            length = size
            entry = {'name': name, 'type': 'discrete', 'size': size, 'start': int(space.start)}
        elif isinstance(space, gym.spaces.Box):
            size = int(np.prod(space.shape))
            length = 2 * size
            entry = {
                'name': name,
                'type': 'box',
                'size': size,
                'low': np.asarray(space.low, dtype=np.float64).reshape(-1).tolist(),
                'high': np.asarray(space.high, dtype=np.float64).reshape(-1).tolist(),
            }
        else:
            raise NotImplementedError(
                'ncnn export: unsupported action space "' + str(name) + '": ' + str(space)
            )
        entry['offset'] = offset
        entry['length'] = length
        layout.append(entry)
        offset += length

    return layout


def decode_actions(logits: np.ndarray, layout: List[dict]) -> Dict[str, np.ndarray]:
    actions = {}
    for entry in layout:
        action_logits = logits[entry['offset']: entry['offset'] + entry['length']]
        if entry['type'] == 'discrete':
            actions[entry['name']] = np.asarray(entry['start'] + int(np.argmax(action_logits)))
        else:
            actions[entry['name']] = action_logits[:entry['size']]
    return actions


# Layer types with no weights in the .bin. Any other unknown type stops the weight description.
_WEIGHTLESS_LAYER_TYPES = {
    'Input', 'Split', 'Concat', 'Slice', 'Reshape', 'Flatten', 'Permute', 'Squeeze', 'ExpandDims',
    'Crop', 'Noop', 'Dropout', 'BinaryOp', 'UnaryOp', 'Eltwise', 'ReLU', 'Sigmoid', 'TanH', 'Softmax',
    'Swish', 'GELU', 'Clip', 'Mish', 'HardSwish', 'HardSigmoid', 'ELU', 'SELU',
}

# `9=` of the layers with a fused activation (InnerProduct, Convolution...).
_FUSED_ACTIVATIONS = {0: None, 1: 'relu', 2: 'leaky_relu', 3: 'clip', 4: 'sigmoid', 5: 'mish', 6: 'hard_swish'}


def _parse_ncnn_param(ncnn_param_path: Path) -> List[dict]:
    """One dict per layer: type, name, input/output blobs and the raw `key=value` parameters."""
    lines = ncnn_param_path.read_text().splitlines()
    layers = []
    for line in lines[2:]:
        tokens = line.split()
        if not tokens:
            continue
        number_inputs, number_outputs = int(tokens[2]), int(tokens[3])
        blobs = tokens[4:4 + number_inputs + number_outputs]
        layers.append({
            'type': tokens[0],
            'name': tokens[1],
            'inputs': blobs[:number_inputs],
            'outputs': blobs[number_inputs:],
            'params': dict(token.split('=', 1) for token in tokens[4 + number_inputs + number_outputs:]),
        })
    return layers


def _fused_activation(params: Dict[str, str]) -> Optional[dict]:
    activation = _FUSED_ACTIVATIONS.get(int(params.get('9', 0)), 'unknown')
    if activation is None:
        return None
    # `-23310=count,value,...`: the activation parameters (leaky_relu: slope, clip: min,max).
    values = [float(value) for value in params.get('-23310', '0').split(',')[1:]]
    return {'type': activation, 'params': values}


def _apply_fused_activation(x: np.ndarray, activation: Optional[dict]) -> np.ndarray:
    if activation is None:
        return x
    if activation['type'] == 'relu':
        return np.maximum(x, 0.0)
    if activation['type'] == 'leaky_relu':
        return np.where(x > 0.0, x, x * activation['params'][0])
    if activation['type'] == 'clip':
        return np.clip(x, activation['params'][0], activation['params'][1])
    if activation['type'] == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-x))
    raise NotImplementedError('activation ' + activation['type'])


# Strongest connections kept per InnerProduct, for the in-game display.
NUMBER_DISPLAYED_CONNECTIONS = 512


def _strongest_connections(weight: np.ndarray, count: int = NUMBER_DISPLAYED_CONNECTIONS) -> dict:
    """The `count` strongest weights, strongest first, spread over the output neurons."""
    num_output = weight.shape[0]
    per_target = min(weight.shape[1], max(1, -(-count // num_output)))
    sources = np.argpartition(-np.abs(weight), per_target - 1, axis=1)[:, :per_target]
    targets = np.repeat(np.arange(num_output), per_target)
    sources = sources.reshape(-1)
    values = weight[targets, sources]
    order = np.argsort(-np.abs(values))[:count]
    return {
        'source': sources[order].tolist(),
        'target': targets[order].tolist(),
        'weight': [round(float(value), 6) for value in values[order]],
    }


def describe_ncnn_graph(ncnn_param_path: Path, ncnn_bin_path: Path, input_shapes: Dict[str, list]) -> Tuple[List[dict], List[dict]]:
    """Layers with their output shapes, and InnerProduct weight offsets in the .ncnn.bin with their
    strongest connections. Each located weight is checked by recomputing its layer."""
    import ncnn

    layers = _parse_ncnn_param(ncnn_param_path)

    net = ncnn.Net()
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.lightmode = False
    net.load_param(str(ncnn_param_path))
    net.load_model(str(ncnn_bin_path))
    extractor = net.create_extractor()
    for blob, shape in input_shapes.items():
        extractor.input(blob, ncnn.Mat(np.random.randn(*shape).astype(np.float32)).clone())
    blob_values = {}
    for layer in layers:
        for blob in layer['outputs']:
            _, mat = extractor.extract(blob)
            blob_values[blob] = np.array(mat)

    raw = ncnn_bin_path.read_bytes()
    offset = 0
    weights = []
    weights_located = True
    for layer in layers:
        if not weights_located or layer['type'] in _WEIGHTLESS_LAYER_TYPES:
            continue
        if layer['type'] == 'LSTM':
            # Skipped, not described, so the layers after it are still located.
            # Each blob: 4 bytes tag + float32 data (weight_xc, bias_c, weight_hc, optional weight_hr).
            num_output = int(layer['params']['0'])
            hidden_size = int(layer['params'].get('3', num_output))
            num_directions = 2 if int(layer['params'].get('2', 0)) == 2 else 1
            sizes = [int(layer['params']['1']), 4 * hidden_size * num_directions, 4 * hidden_size * num_output * num_directions]
            if num_output != hidden_size:
                sizes.append(hidden_size * num_output * num_directions)
            for size in sizes:
                if int.from_bytes(raw[offset:offset + 4], 'little') != 0:
                    print('ncnn export: non float32 weights in layer "' + layer['name'] + '", no weight description from it on')
                    weights_located = False
                    break
                offset += 4 + size * 4
            continue
        if layer['type'] != 'InnerProduct' or int(layer['params'].get('8', 0)) != 0:
            print('ncnn export: no weight description from layer "' + layer['name'] + '" (' + layer['type'] + ') on')
            weights_located = False
            continue

        num_output = int(layer['params']['0'])
        weight_size = int(layer['params']['2'])
        # Weight: 4 bytes tag (0 = float32) then data. Bias: raw float32.
        tag = int.from_bytes(raw[offset:offset + 4], 'little')
        if tag != 0:
            print('ncnn export: non float32 weights (tag ' + hex(tag) + ') in layer "' + layer['name'] + '", no weight description from it on')
            weights_located = False
            continue
        weight_offset = offset + 4
        offset = weight_offset + weight_size * 4
        bias_offset = None
        if int(layer['params'].get('1', 0)):
            bias_offset = offset
            offset += num_output * 4

        weight = np.frombuffer(raw, np.float32, weight_size, weight_offset).reshape(num_output, -1)
        bias = np.zeros(num_output, np.float32) if bias_offset is None else np.frombuffer(raw, np.float32, num_output, bias_offset)
        activation = _fused_activation(layer['params'])
        expected = blob_values[layer['outputs'][0]].reshape(-1)
        recomputed = _apply_fused_activation(weight @ blob_values[layer['inputs'][0]].reshape(-1) + bias, activation)
        relative_error = float(np.max(np.abs(recomputed - expected)) / max(1.0, float(np.max(np.abs(expected)))))
        if relative_error > 1e-3:
            raise RuntimeError('ncnn export: weights located for "' + layer['name'] + '" do not reproduce its output '
                               '(relative error ' + str(relative_error) + ')')

        weights.append({
            'layer': layer['name'],
            'num_input': weight.shape[1],
            'num_output': num_output,
            'weight_offset': weight_offset,
            'bias_offset': bias_offset,
            'connections': _strongest_connections(weight),
        })
    if weights_located and offset != len(raw):
        raise RuntimeError('ncnn export: ' + str(len(raw) - offset) + ' bytes of ' + str(ncnn_bin_path) + ' not accounted for')

    layer_descriptions = []
    for layer in layers:
        description = {
            'name': layer['name'],
            'type': layer['type'],
            'inputs': layer['inputs'],
            'outputs': [{'blob': blob, 'shape': list(blob_values[blob].shape)} for blob in layer['outputs']],
        }
        activation = _fused_activation(layer['params'])
        if activation is not None:
            description['activation'] = activation
        layer_descriptions.append(description)

    return layer_descriptions, weights


def verify_ncnn_export(ncnn_param_path: Path, ncnn_bin_path: Path, rl_module: RLModule, number_steps: int = 256):
    """Consecutive single steps, each side feeding back its own LSTM state."""
    import ncnn

    net = ncnn.Net()
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.load_param(str(ncnn_param_path))
    net.load_model(str(ncnn_bin_path))

    input_names, inputs = dummy_inputs(rl_module, batch_size=1)
    is_stateful = len(inputs) > 1
    output_names = _output_names(is_stateful)
    layout = action_layout(rl_module.action_space)
    action_distribution_class = rl_module.get_inference_action_dist_cls()

    pytorch_state = list(inputs[1:])
    ncnn_state = [tensor.cpu().numpy()[0] for tensor in inputs[1:]]
    max_errors = {name: 0.0 for name in output_names}
    action_errors = {entry['name']: 0.0 for entry in layout}

    for _ in range(number_steps):
        obs = dummy_inputs(rl_module, batch_size=1)[1][0]

        with torch.no_grad():
            pytorch_outputs = rl_module.forward_inference_logits(dict(zip(input_names, [obs, *pytorch_state])))
            pytorch_actions = (
                action_distribution_class.from_logits(pytorch_outputs['logits']).to_deterministic().sample()
            )
        if not isinstance(pytorch_actions, dict):
            pytorch_actions = {'output': pytorch_actions}

        extractor = net.create_extractor()
        for i, array in enumerate([obs.cpu().numpy()[0], *ncnn_state]):
            extractor.input('in' + str(i), ncnn.Mat(np.ascontiguousarray(array, dtype=np.float32)).clone())
        ncnn_outputs = {}
        for i, name in enumerate(output_names):
            _, mat = extractor.extract('out' + str(i))
            ncnn_outputs[name] = np.array(mat).reshape(-1)

        for name in output_names:
            error = np.max(np.abs(pytorch_outputs[name].cpu().numpy()[0].reshape(-1) - ncnn_outputs[name]))
            max_errors[name] = max(max_errors[name], float(error))

        ncnn_actions = decode_actions(ncnn_outputs['logits'], layout)
        for name, ncnn_action in ncnn_actions.items():
            pytorch_action = pytorch_actions[name].cpu().numpy()[0].reshape(-1)
            error = np.max(np.abs(pytorch_action - ncnn_action.reshape(-1)))
            action_errors[name] = max(action_errors[name], float(error))

        if is_stateful:
            pytorch_state = [pytorch_outputs['state_out_h'], pytorch_outputs['state_out_c']]
            ncnn_state = [ncnn_outputs['state_out_h'], ncnn_outputs['state_out_c']]

    for name, error in max_errors.items():
        print(f'[{name}] max error PyTorch vs ncnn over {number_steps} steps: {error:.6f}')
    for name, error in action_errors.items():
        print(f'[action {name}] max error PyTorch vs decoded ncnn: {error:.6f}')


def rl_module_to_ncnn(rl_module: RLModule, policy_name: str, ncnn_path: Path, file_stem: Optional[str] = None) -> List[Path]:
    """PyTorch -> pnnx -> ncnn, batch size 1. Outputs the raw logits, decoded with the json."""
    import pnnx

    print('Export policy to ncnn: ' + policy_name)
    # pip pnnx is CPU only.
    rl_module = copy.deepcopy(rl_module).cpu()
    rl_module.device = torch.device('cpu')

    input_names, args = dummy_inputs(rl_module, batch_size=1)
    is_stateful = len(args) > 1
    output_names = _output_names(is_stateful)

    stem = file_stem or policy_name
    ncnn_param_path = ncnn_path / (stem + '.ncnn.param')
    ncnn_bin_path = ncnn_path / (stem + '.ncnn.bin')
    metadata_path = ncnn_path / (stem + '.ncnn.json')

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        pnnx.export(
            NcnnWrapper(rl_module, is_stateful).eval(),
            str(temporary_path / (stem + '.pt')),
            args,
            device='cpu',
            pnnxparam=str(temporary_path / (stem + '.pnnx.param')),
            pnnxbin=str(temporary_path / (stem + '.pnnx.bin')),
            pnnxpy=str(temporary_path / (stem + '_pnnx.py')),
            pnnxonnx=str(temporary_path / (stem + '.pnnx.onnx')),
            ncnnparam=str(ncnn_param_path),
            ncnnbin=str(ncnn_bin_path),
            ncnnpy=str(temporary_path / (stem + '_ncnn.py')),
            fp16=False,
        )

    blob_names = set(ncnn_param_path.read_text().split())
    for blob_name in ['in' + str(i) for i in range(len(input_names))] + ['out' + str(i) for i in range(len(output_names))]:
        if blob_name not in blob_names:
            raise RuntimeError('ncnn export: blob "' + blob_name + '" missing from ' + str(ncnn_param_path))

    inputs = [
        {'blob': 'in' + str(i), 'name': name, 'shape': list(tensor.shape[1:])}
        for i, (name, tensor) in enumerate(zip(input_names, args))
    ]
    layers, weights = describe_ncnn_graph(
        ncnn_param_path, ncnn_bin_path, {entry['blob']: entry['shape'] for entry in inputs}
    )

    metadata = {
        'policy_name': policy_name,
        'inputs': inputs,
        'outputs': [
            {'blob': 'out' + str(i), 'name': name}
            for i, name in enumerate(output_names)
        ],
        'is_stateful': is_stateful,
        'actions': action_layout(rl_module.action_space),
        'layers': layers,
        'weights': weights,
    }
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print('Exported ncnn: ' + str(ncnn_param_path) + ', ' + str(ncnn_bin_path) + ', ' + str(metadata_path))
    verify_ncnn_export(ncnn_param_path=ncnn_param_path, ncnn_bin_path=ncnn_bin_path, rl_module=rl_module)
    print()

    return [ncnn_param_path, ncnn_bin_path, metadata_path]
