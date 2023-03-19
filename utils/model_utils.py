from collections import OrderedDict

import torch.nn as nn


def create_mlp(num_layers: int, in_features: int, middle_features: int, out_features: int) -> nn.Sequential:
    """
    create multi-layer perceptron
    :param num_layers: amount of layers
    :param in_features: number of input features
    :param middle_features: number of intermediate features
    :param out_features: number of output features
    :return: multi-layer perceptron
    """
    layers = [("linear_1", nn.Linear(in_features, out_features if num_layers == 1 else middle_features))]
    for i in range(num_layers - 1):
        layers.append((f"batchnorm_{i+1}", nn.BatchNorm1d(num_features=middle_features)))
        layers.append((f"relu_{i+1}", nn.ReLU()))
        layers.append((f"linear_{i+2}", nn.Linear(middle_features,
                                                  out_features if i == num_layers - 2 else middle_features,
                                                  False if i == num_layers - 2 else True)
                       )
                      )
    return nn.Sequential(OrderedDict(layers))
