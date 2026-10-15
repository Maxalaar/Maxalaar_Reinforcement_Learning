from typing import Optional

from torch import nn
import torch.nn.functional as F

class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def create_dense_architecture(
        input_dimension,
        configuration_hidden_layers,
        output_dimension,
        activation_function_class,
        configuration_layer_normalizations=None,
        dropout: Optional[float] = None,
):
    if len(configuration_hidden_layers) != 0:
        layers = []
        current_input = input_dimension

        for i, hidden_size in enumerate(configuration_hidden_layers):
            layers.append(nn.Linear(current_input, hidden_size))
            layers.append(activation_function_class())

            if configuration_layer_normalizations is not None:
                if configuration_layer_normalizations[i]:
                    layers.append(NormalizationLayer())

            if dropout is not None and dropout != False and i < len(configuration_hidden_layers):
                layers.append(nn.Dropout(p=dropout))

            current_input = hidden_size

        layers.append(nn.Linear(current_input, output_dimension, bias=False))

        return nn.Sequential(*layers)
    else:
        return nn.Linear(input_dimension, output_dimension, bias=False)