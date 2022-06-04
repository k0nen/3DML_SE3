import torch
import torch.nn as nn
import numpy as np
import dgl.function as dgl_fn
from dgl.nn.pytorch.softmax import edge_softmax

from model.graph import GraphLinear, GraphAttentiveLinear, GraphCat
from model.convolution import ConvSE3
from model.utils import num_features, num_parameters


class GraphAttention(nn.Module):
    """
    SE(3)-equivariant multi-head attention block.
    This module only calculates the attention values given QKV as input; thus there are no trainable parameters.
    For full self-attention with QKV & residual connection, see GraphSelfAttn below.
    """

    def __init__(self, shape_value: dict, shape_key: dict, n_heads: int):
        super().__init__()
        self.shape_value = shape_value
        self.shape_key = shape_key
        self.n_heads = n_heads

    def message_passing(self, t_value):
        """
        Create a function that aggregates features to generate type-(t_value) output features.
        This function will be applied to the whole graph using G.update_all() in the forward pass.
        In this case, it multiplies the attention weights and value, which will be summed up in G.update_all().
        """

        def fn(edges):
            attn = edges.data["a"]
            value = edges.data[f"v{t_value}"]
            message = attn.view(*attn.shape, 1, 1) * value
            return {"m": message}

        return fn

    def forward(self, q, k, v, G):
        with G.local_scope():
            # Add node features to local graph scope
            for t_value, ch_value in self.shape_value.items():
                G.edata[f"v{t_value}"] = v[f"{t_value}"].view(
                    -1, self.n_heads, ch_value // self.n_heads, t_value * 2 + 1
                )
            _tmp = [
                k[f"{i}"].view(*k[f"{i}"].shape[:-2], self.n_heads, -1)
                for i in self.shape_key
            ]
            G.edata["k"] = torch.cat(_tmp, -1)
            _tmp = [
                q[f"{i}"].view(*q[f"{i}"].shape[:-2], self.n_heads, -1)
                for i in self.shape_key
            ]
            G.ndata["q"] = torch.cat(_tmp, -1)

            # Compute attention weights and softmax
            # 'k'*'q' results are stored to 'e'
            G.apply_edges(dgl_fn.e_dot_v("k", "q", "e"))
            n_edges = G.edata["k"].shape[0]
            e = G.edata.pop("e")
            e = e.view([n_edges, self.n_heads])
            e /= np.sqrt(num_features(self.shape_key))
            G.edata["a"] = edge_softmax(G, e)

            # Perform message passing for each output type of V
            # Sum up the product of attention and value
            for t_value in self.shape_value:
                G.update_all(
                    message_func=self.message_passing(t_value),
                    reduce_func=dgl_fn.sum("m", f"out_{t_value}"),
                )
            return {
                f"{t_value}": G.ndata[f"out_{t_value}"].view(
                    -1, ch_value, t_value * 2 + 1
                )
                for t_value, ch_value in self.shape_value.items()
            }


class GraphSelfAttn(nn.Module):
    """
    SE(3)-equivariant multi-head self-attention and residual connection.
    Effectively, the forward pass of this module is Eq.(10) of the paper.
    """

    def __init__(
        self,
        shape_in: dict,
        shape_out: dict,
        div: float = 4,
        n_heads: int = 1,
        edge_dim: int = 0,
        self_int: str = "1x1",  # ['1x1', 'att']
    ):
        """
        shape_in: dict of (type, channel) describing input
        shape_out: dict of (type, channel) describing output
        div: QKV of each self-attention will have (channel / div) channels.
        n_heads: Number of heads for multi-head attention.
        edge_dim: # of dimensions in edge embedding
        self_int: Type of self-interaction.
                  - 1x1 is Eq. (12) of the paper.
                  - att is Eq. (13) of the paper.
        """

        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.div = div
        self.n_heads = n_heads
        self.edge_dim = edge_dim

        # Intermediate feature shape
        self.shape_Q = {k: int(v // div) for k, v in shape_out.items() if k in shape_in}
        self.shape_V = {k: int(v // div) for k, v in shape_out.items()}
        self.shape_K = {k: int(v // div) for k, v in shape_out.items() if k in shape_in}
        self.shape_VK = {
            k: int(v // div) * (2 if k in shape_in else 1) for k, v in shape_out.items()
        }

        self.attn = nn.ModuleDict()

        # Projections
        # We fuse Value/Key projections then slice them later for improved performance.
        self.attn["vk"] = ConvSE3(shape_in, self.shape_VK, edge_dim=edge_dim)
        self.attn["q"] = GraphLinear(shape_in, self.shape_Q)

        # Attention
        self.attn["attn"] = GraphAttention(self.shape_V, self.shape_K, n_heads)

        # Skip connection
        self.cat = GraphCat(self.shape_V, shape_in)
        if self_int == "att":
            self.project = GraphAttentiveLinear(self.cat.shape_out, shape_out)
        elif self_int == "1x1":
            self.project = GraphLinear(self.cat.shape_out, shape_out)

    def split_vk(self, vk):
        keys, values = {}, {}
        for k, v in vk.items():
            if int(k) in self.shape_in.keys():
                values[k], keys[k] = torch.chunk(v, chunks=2, dim=-2)
            else:
                values[k] = v
        return values, keys

    def forward(self, f, G, basis):
        vk = self.attn["vk"](f, G, basis)
        q = self.attn["q"](f)
        v, k = self.split_vk(vk)
        z = self.attn["attn"](q, k, v, G)

        z = self.cat(z, f)
        z = self.project(z)

        return z
