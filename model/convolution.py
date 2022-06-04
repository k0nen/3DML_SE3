import torch
import torch.nn as nn
import numpy as np
import dgl.function as dgl_fn

from model.utils import num_parameters


class ConvSE3(nn.Module):
    """
    ConvSE3 is the equivalent of a graph conv layer in a GCN.
    The exact behavior depends on the value of self_interaction.

    - self_interaction = True
    In this case, this module corresponds to Eq. (10) of the paper.
    This module will return a dictionary of node features with output type as key.

    - self_interaction = False
    In this case, this module corresponds to (2). value message of Eq. (10) of the paper.
    This module will return a dictionary of edge features with output type as key.
    """

    def __init__(
        self,
        shape_in: dict,
        shape_out: dict,
        edge_dim: int = 0,
        self_interaction: bool = False,
    ):
        """
        shape_in: dict of (type, channel) describing input
        shape_out: dict of (type, channel) describing output
        edge_dim: # of dimensions in edge embedding
        """

        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        # Neighbor-interaction weights
        # Used to aggregate features from neighbor of a node.
        self.kernel_neighbor = nn.ModuleDict()
        for (t_in, ch_in) in shape_in.items():
            for (t_out, ch_out) in shape_out.items():
                module_key = f"{t_in},{t_out}"
                self.kernel_neighbor[module_key] = SingleConv(
                    t_in, t_out, ch_in, ch_out, edge_dim
                )

        if self_interaction:
            self.kernel_self = nn.ParameterDict()
            for (t_in, ch_in) in shape_in.items():
                if t_in in shape_out.keys():
                    ch_out = shape_out[t_in]
                    W = nn.Parameter(torch.randn(1, ch_out, ch_in) / np.sqrt(ch_in))
                    self.kernel_self[f"{t_in}"] = W

    def message_passing_node(self, t_out):
        """
        This function will be applied to the whole graph using G.update_all() in the forawrd pass.
        """

        def fn(edges):
            message = 0

            # Self-interaction
            if t_out in self.shape_in.keys():
                dst = edges.dst[f"{t_out}"]
                W = self.kernel_self[f"{t_out}"]
                message += torch.matmul(W, dst)

            # Neighbor-interaction
            for t_in, ch_in in self.shape_in.items():
                d_in = 2 * t_in + 1
                src = edges.src[f"{t_in}"].view(-1, ch_in * d_in, 1)
                edge = edges.data[f"{t_in},{t_out}"]
                message += torch.matmul(edge, src)

            return {"message": message.view(message.shape[0], -1, t_out * 2 + 1)}

        return fn

    def message_passing_edge(self, t_out):
        """
        This function will be applied to the whole graph using G.apply_edges() in the forawrd pass.
        """

        def fn(edges):
            message = 0

            # Neighbor-interaction
            for (t_in, ch_in) in self.shape_in.items():
                d_in = 2 * t_in + 1
                src = edges.src[f"{t_in}"].view(-1, ch_in * d_in, 1)

                if t_in == 1 and ch_in > 1:
                    # Add relative position to existing feature vector
                    rel = (edges.dst["x"] - edges.src["x"]).view(-1, 3, 1)
                    src[..., :d_in, :1] = src[..., :d_in, :1] + rel

                edge = edges.data[f"{t_in},{t_out}"]
                message += torch.matmul(edge, src)

            return {f"out_{t_out}": message.view(message.shape[0], -1, t_out * 2 + 1)}

        return fn

    def forward(self, h, G, basis):
        """
        h: dict of features
        G: minibatch of graphs
        basis: basis
        """

        with G.local_scope():
            # Add node features to local graph scope
            for (k, v) in h.items():
                G.ndata[k] = v

            # Add edge features to local graph scope
            f = G.edata["r"]
            if "w" in G.edata.keys():
                w = G.edata["w"]
                f = torch.cat([w, f], -1)

            for (t_in, ch_in) in self.shape_in.items():
                for (t_out, ch_out) in self.shape_out.items():
                    module_key = f"{t_in},{t_out}"
                    G.edata[module_key] = self.kernel_neighbor[module_key](f, basis)

            # For each type-(t_out), apply message passing function
            if self.self_interaction:
                for t_out in self.shape_out.keys():
                    G.update_all(
                        self.message_passing_node(t_out),
                        dgl_fn.mean("message", f"out_{t_out}"),
                    )
                return {f"{t_out}": G.ndata[f"out_{t_out}"] for t_out in self.shape_out}

            else:
                for t_out in self.shape_out.keys():
                    G.apply_edges(self.message_passing_edge(t_out))
                return {f"{t_out}": G.edata[f"out_{t_out}"] for t_out in self.shape_out}


class SingleConv(nn.Module):
    """
    Basic building block for a single pairwise convolution.
    One instance of SingleConv can perform a convolution from
    (ch_in) type-(t_in) features to (ch_out) type-(t_out) features.

    Effectively, the forward pass of this module is Eq.(8) of the paper
    where k=t_in and l=t_out.
    """

    def __init__(self, t_in: int, t_out: int, ch_in: int, ch_out: int, edge_dim: int):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.edge_dim = edge_dim

        # A type-x feature has 2x+1 dimensions.
        self.d_out = t_out * 2 + 1
        self.fwd_out = self.d_out * self.ch_out

        # We need this amount of radial functions for this layer.
        # See subscript J of Eq.(8) in the paper.
        self.num_freq = min(t_in, t_out) * 2 + 1

        # Declare radial functions
        self.rf = RadialFunc(self.num_freq, ch_in, ch_out, edge_dim)

    def forward(self, f, basis):
        R = self.rf(f).view(-1, self.ch_out, 1, self.ch_in, 1, self.num_freq)
        basis_key = f"{self.t_in},{self.t_out}"

        kernel = torch.sum(R * basis[basis_key], -1)

        d_out = self.t_out * 2 + 1
        return kernel.view(kernel.shape[0], self.fwd_out, -1)


class RadialFunc(nn.Module):
    """
    The radial function which takes radius as input.

    Effectively, the forward pass of this module is phi^{lk} of Eq.(8) of the paper.
    Note the parallel computation over subscript J.

    num_freq: Dimension ( = 2 * type + 1 )
    """

    def __init__(self, num_freq: int, ch_in: int, ch_out: int, edge_dim: int):
        super().__init__()
        self.num_freq = num_freq
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.net = nn.Sequential(
            nn.Linear(edge_dim + 1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_freq * ch_in * ch_out),
        )

        for i in [0, 3, 6]:
            nn.init.kaiming_uniform_(self.net[i].weight)

    def forward(self, x):
        out = self.net(x)
        return out
