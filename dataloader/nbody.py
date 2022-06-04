import dgl
import torch

import pickle
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from dataloader.basis import precompute_basis


_DTYPE = np.float32


class NBodyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        train: bool,
        basis_dir: str = None,
        target_t: int = 5,
    ):
        """
        target_t is the number of timesteps we want to predict into the future.
        Since the dataset is sampled every 100 timesteps and default value is 5,
        we are predicting 500 timesteps into the future in this setting.

        basis_dir is the filename of precomputed basis.
        This must be specified in order to train the model; None value should only be used when
        this class is used as a base dataloader to precompute the basis itself.
        """

        super().__init__()
        self.data_dir = data_dir
        self.basis_dir = basis_dir
        self.train = train
        self.target_t = target_t

        with open(data_dir, "rb") as f:
            self.data = pickle.load(f)

            self.data["points"] = np.swapaxes(self.data["points"], 2, 3)
            self.data["vel"] = np.swapaxes(self.data["vel"], 2, 3)

            self.len = self.data["points"].shape[0]
            self.n_frames = self.data["points"].shape[1]
            self.n_points = self.data["points"].shape[2]

        # Construct a fully connected graph with n_points nodes
        # Will be repeatedly used in __getitem__()
        src, dst = [], []
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self.src = np.array(src)
        self.dst = np.array(dst)

        if basis_dir is not None:
            with open(basis_dir, "rb") as f:
                self.basis = pickle.load(f)
        else:
            self.basis = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        # Determine which time window (frame_0 to frame_T) to get item.
        last_possible = self.n_frames - self.target_t
        if self.train:
            frame_0 = np.random.choice(range(last_possible))
        else:
            frame_0 = int(last_possible / self.len * idx)
        frame_T = frame_0 + self.target_t

        # Construct tensor with features
        # Code from authors
        x_0 = torch.tensor(self.data["points"][idx, frame_0].astype(_DTYPE))
        x_T = torch.tensor(self.data["points"][idx, frame_T].astype(_DTYPE)) - x_0
        v_0 = torch.tensor(self.data["vel"][idx, frame_0].astype(_DTYPE))
        v_T = torch.tensor(self.data["vel"][idx, frame_T].astype(_DTYPE)) - v_0
        charges = torch.tensor(self.data["charges"][idx].astype(_DTYPE))

        # Create a graph and add feature info
        # Node data: position, velocity, charge
        # Edge data: relative position, charge(same/different)
        G = dgl.graph((self.src, self.dst))
        G.ndata["x"] = x_0.unsqueeze(1)  # [n_points, 1, 3]
        G.ndata["v"] = v_0.unsqueeze(1)  # [n_points, 1, 3]
        G.ndata["c"] = charges.unsqueeze(1)  # [n_points, 1, 1]
        G.edata["d"] = x_0[self.dst] - x_0[self.src]  # Relative position
        G.edata["r"] = torch.sqrt(
            (G.edata["d"] ** 2).sum(-1, keepdim=True)
        )  # Relative distance
        G.edata["w"] = charges[self.src] * charges[self.dst]

        # Inputs to the network will be extracted from G.
        # Goal of network is to predict values in x_T, v_T.
        # Basis and r are preprocessed info of graphs.
        idx_q, idx_r = idx // 128, idx % 128
        if self.basis is not None:
            basis = {k: v[idx_r : idx_r + 20] for k, v in self.basis[idx_q].items()}
            return G, basis, x_T, v_T
        else:
            return G, x_T, v_T


def nbody_collate_fn(batch):
    """
    Collate function for nbody.
    NBodyDataset.__getitem__() returns 3 objects + optional extra (basis).
    - G is a DGL graph, so we collate by calling dgl.batch().
    - x_T and v_T is tensor, so we simply use torch.stack().
    - basis of a graph is given as a dict with keys f'{type_in},{type_out}'.
      Therefore, we construct a single dictionary for this batch
      where the keys are the same, and the value tensors of each graph in this batch
      are stacked into a single tensor.
    """
    batch_list = list(map(list, zip(*batch)))

    batch_G = dgl.batch(batch_list[0])
    batch_x_T = torch.stack(batch_list[2])
    batch_v_T = torch.stack(batch_list[3])

    basis_list = batch_list[1]
    batch_basis = {
        k: torch.cat([b[k] for b in basis_list], dim=0) for k in basis_list[0].keys()
    }

    return batch_G, batch_basis, batch_x_T, batch_v_T
