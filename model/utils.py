import torch.nn as nn
import numpy as np


def num_features(shape: dict) -> int:
    """
    Calculate the total dimension if every feature was concatenated together.
    """
    return sum(v * (k * 2 + 1) for k, v in shape.items())


def num_parameters(model: nn.Module) -> int:
    """
    Returns the total number of trainable parameters in the given model.
    """

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params
