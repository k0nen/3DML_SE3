import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch.glob import MaxPooling

from model.utils import num_parameters

from pdb import set_trace


class GraphLinear(nn.Module):
    """
    Graph linear layer, equivalent to 1x1 convolution: See Eq.(12) of the paper.
    """

    def __init__(self, shape_in: dict, shape_out: dict):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out

        self.W = nn.ParameterDict()
        for t_out, ch_out in shape_out.items():
            ch_in = shape_in[t_out]
            _w = torch.randn(ch_out, ch_in) / np.sqrt(ch_in)
            self.W[f"{t_out}"] = nn.Parameter(_w)

    def forward(self, f):
        out = {}
        for k, v in f.items():
            key_str = f"{k}"
            if key_str in self.W:
                out[key_str] = torch.matmul(self.W[key_str], v)
        return out


class GraphAttentiveLinear(nn.Module):
    """
    Self interaction with attention: See Eq.(13) of the paper.
    """

    def __init__(self, shape_in: dict, shape_out: dict):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out

        self.W = nn.ModuleDict()
        for t_in, ch_in in shape_in.items():
            ch_out = shape_out[t_in]
            n_input = ch_in * ch_in
            n_hidden = ch_in * ch_out
            self.W[f"{t_in}"] = nn.Sequential(
                nn.LayerNorm(n_input),
                nn.LeakyReLU(),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.LayerNorm(n_hidden),
                nn.LeakyReLU(),
                nn.Linear(n_hidden, n_hidden, bias=True),
            )
            for i in [2, 5]:
                nn.init.kaiming_uniform_(self.W[f"{t_in}"][i].weight)

    def forward(self, f):
        out = {}
        for k, v in f.items():
            # Shape of v: [*v_shape_head, shape_in[k], 2*k+1]
            v_shape_head = v.shape[:-2]
            ch_in = self.shape_in[int(k)]
            ch_out = self.shape_out[int(k)]

            dot = torch.einsum("...ac,...bc->...ab", [v, v]).view(*v_shape_head, -1)
            dot_sign = dot.sign()
            dot = abs(dot).clamp_min(1e-12) * dot_sign
            attn = self.W[k](dot)
            attn = attn.view(*v_shape_head, ch_out, ch_in)
            attn = F.softmax(attn, dim=-1)

            out[k] = torch.einsum("xab,xbc->xac", [attn, v])
        return out


class GraphNormBias(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.
    This module will re-scale each feature vector by a function of the feature's norm.
    """

    def __init__(self, shape_in: dict, batchnorm: bool = False):
        """
        If batchnorm = False, the scale is determined by adding learnable bias value followed by relu.
        If batchnorm = True, the scale is determined by batch normalization followed by relu.
        """

        super().__init__()
        self.shape_in = shape_in
        self.batchnorm = batchnorm

        if batchnorm:
            self.net = nn.ModuleDict()
        else:
            self.net = nn.ParameterDict()
        for k, v in self.shape_in.items():
            if batchnorm:
                self.net[f"{k}"] = nn.Sequential(nn.LayerNorm(v), nn.ReLU())
            else:
                self.net[f"{k}"] = nn.Parameter(torch.randn(1, v) / np.sqrt(v))

    def forward(self, f, G, basis):
        out = {}
        for k, v in f.items():
            norm = v.norm(2, -1, keepdim=True).clamp_min(1e-12).expand_as(v)
            phase = v / norm  # Unit vector in the direction of each feature

            if self.batchnorm:
                scale = self.net[f"{k}"](norm[..., 0])
            else:
                scale = F.relu(
                    norm[..., 0] + self.net[f"{k}"]
                )  # Scaling factor of each feature
            out[k] = scale.unsqueeze(-1) * phase

        return out


class GraphCat(nn.Module):
    """
    Concat operation (no trainable parameters).
    The order of input does matter;
    note that only types which exist in f1 will be concatenated.
    """

    def __init__(self, shape_1: dict, shape_2: dict):
        super().__init__()
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.shape_out = {
            k: v + self.shape_2.get(k, 0) for k, v in self.shape_1.items()
        }

    def forward(self, f1, f2):
        out = {}
        for k, v in f1.items():
            key_str = f"{k}"
            if key_str in f2:
                out[key_str] = torch.cat([f1[key_str], f2[key_str]], 1)
            else:
                out[key_str] = f1[key_str]
        return out


class GraphMaxPool(nn.Module):
    """
    Max pooling operation.
    """

    def __init__(self):
        super().__init__()
        self.maxpool = MaxPooling()

    def forward(self, f, G, basis):
        h = f["0"]
        return self.maxpool(G, h)
