from torch import nn, TensorType

activation_functions = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.SiLU,
)