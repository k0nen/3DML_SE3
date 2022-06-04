import torch
import torch.nn as nn
import sys
import time

from model.graph import GraphNormBias, GraphMaxPool
from model.attention import GraphSelfAttn

from model.convolution import ConvSE3
from model.convolution import *

from pdb import set_trace

"""
The author's code describes each feature with "Fiber" structure.
In our implementation, we simply use dict[int: int].
For example, feature shape {0: 1, 1: 3, 2:1} means that there are
- 1 type-0 field features (type-0 features are 1-D vectors)
- 3 type-1 field features (type-1 features are 3-D vectors)
- 1 type-2 field features (type-2 features are 5-D vectors)
in the corresponding feature.
"""


class SE3Transformer(nn.Module):
    """
    SE(3)-transformer for n-body simulations task.

    num_layers: Number of self_attention layers in our transformer.
                There will be (num_layers) self-attention layers with (si_m) interaction,
                and 1 final layer with (si_e) interaction.
    num_channels: Number of channels in hidden layers.
    num_degrees: Maximum type of hidden layer features.
                 Every hidden layer will contain
                 - (num_channel) type-0 features,
                 - (num_channel) type-1 features,
                 - ... ,
                 - (num_channel) type-(num_degree) features.
    div: QKV of each self-attention will have (num_channel/div) channels.
    n_heads: Number of heads for multi-head attention.
    si_m: Type of self-interaction for hidden self attention layers. ['1x1', 'att']
    si_e: Type of self-interaction for final self attention layer. ['1x1', 'att']
          The difference between the two values are:
          - 1x1 is Eq. (12) of the paper.
          - att is Eq. (13) of the paper.
    """

    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        num_degrees: int = 4,
        edge_dim: int = 1,
        div: int = 4,
        n_heads: int = 1,
        si_m: str = "1x1",
        si_e: str = "att",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.n_heads = n_heads
        self.si_m = si_m
        self.si_e = si_e

        # Define shape of features
        # shape_in: QM9 dataset has 6 atomic features and 4 edge features.
        shape_in = {0: 6}
        shape_mid = {i: num_channels for i in range(num_degrees)}
        shape_out = {0: num_channels * num_degrees}
        feature_out = sum(v * (k * 2 + 1) for k, v in shape_out.items())

        # Build architecture
        self.GraphBlock = nn.ModuleList()
        for i in range(self.num_layers):
            _shape_tmp = shape_in if i == 0 else shape_mid
            self.GraphBlock.append(
                GraphSelfAttn(
                    _shape_tmp, shape_mid, div, n_heads, edge_dim, self_int=si_m
                )
            )
            self.GraphBlock.append(GraphNormBias(shape_mid, batchnorm=True))
        self.GraphBlock.append(
            ConvSE3(shape_mid, shape_out, edge_dim, self_interaction=True)
        )
        self.GraphBlock.append(GraphMaxPool())

        self.MLP = nn.Sequential(
            nn.Linear(feature_out, feature_out), nn.ReLU(), nn.Linear(feature_out, 1)
        )

    def forward(self, G, basis):
        f = {"0": G.ndata["f"]}
        for layer in self.GraphBlock:
            f = layer(f, G, basis)
        f = self.MLP(f.squeeze(2))
        return f
