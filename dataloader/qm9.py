import dgl
import torch
import pickle

from dgl.data import QM9EdgeDataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from pdb import set_trace


class QM9Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str,  ## ['train', 'val', 'test']
        task_idx: int = 0,
        basis_dir: str = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.basis_dir = basis_dir
        self.mode = mode
        self.task_idx = task_idx

        if mode == "train":
            self.mode_offset = 0
            self.ne_offset = 0
            self.len = 100000
        elif mode == "val":
            self.mode_offset = 100000
            self.ne_offset = 3705622
            self.len = 18000
        elif mode == "test":
            self.mode_offset = 118000
            self.ne_offset = 4451598
            self.len = 12831
        else:
            raise Exception(f"Wrong QM9 dataset mode {mode}")

        self.qm9_base = QM9EdgeDataset(verbose=True, raw_dir=self.data_dir)

        if basis_dir is not None:
            with open(basis_dir, "rb") as f:
                self.basis = pickle.load(f)
        else:
            self.basis = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        true_idx = idx + self.mode_offset
        G_base, y = self.qm9_base[true_idx]
        y_task = y[0, self.task_idx : self.task_idx + 1]

        G_base_src, G_base_dst = G_base.edges()
        G = dgl.graph((G_base_src, G_base_dst))
        G.ndata["x"] = G_base.ndata["pos"].unsqueeze(1)  # [n_points, 1, 3]
        G.ndata["f"] = G_base.ndata["attr"][:, :6].unsqueeze(2)  # [n_points, 6, 1]
        G.edata["d"] = (
            G.ndata["x"][G_base_dst] - G.ndata["x"][G_base_dst]
        )  # Relative position
        G.edata["r"] = torch.sqrt(
            (G.edata["d"] ** 2).sum(-1, keepdim=True)
        )  # Relative distance
        G.edata["w"] = G_base.edata["edge_attr"].unsqueeze(1)

        if self.basis is not None:
            true_idx_q = true_idx // 512
            idx_q = idx // 512
            basis_start = (
                self.qm9_base.ne_cumsum[true_idx]
                - self.qm9_base.ne_cumsum[self.mode_offset + idx_q * 512]
            )
            basis_final = (
                self.qm9_base.ne_cumsum[true_idx + 1]
                - self.qm9_base.ne_cumsum[self.mode_offset + idx_q * 512]
            )
            basis = {
                k: v[basis_start:basis_final] for k, v in self.basis[idx_q].items()
            }
            return G, basis, y_task
        else:
            return G, y_task


def qm9_collate_fn(batch):
    """
    Collate function for QM9.
    - G is a DGL graph, so we collate by calling dgl.batch().
    - y_task is a tensor, so we simply use torch.stack().
    - basis of a graph is given as a dict with keys f'{type_in},{type_out}'.
      Therefore, we construct a single dictionary for this batch
      where the keys are the same, and the value tensors of each graph in this batch
      are stacked into a single tensor.
    """
    batch_list = list(map(list, zip(*batch)))

    batch_G = dgl.batch(batch_list[0])
    batch_y_task = torch.stack(batch_list[2])

    basis_list = batch_list[1]
    batch_basis = {
        k: torch.cat([b[k] for b in basis_list], dim=0) for k in basis_list[0].keys()
    }

    return batch_G, batch_basis, batch_y_task


if __name__ == "__main__":
    z = QM9Dataset("../data/", "train", task_idx=0)
    set_trace()
