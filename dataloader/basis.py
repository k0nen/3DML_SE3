import pickle
import numpy as np
import torch
import torch.nn as nn
from functools import lru_cache
from typing import Dict, List
import e3nn.o3 as o3
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm


def _get_relative_pos(graph):
    x = graph.ndata["x"]
    src, dst = graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos.squeeze(1)


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int) -> Tensor:
    """Get the (cached) Q^{d_out,d_in}_J matrices from Eq. (8) of the paper"""
    return o3.wigner_3j(
        J, d_in, d_out, dtype=torch.float64, device=torch.device("cuda")
    ).permute(2, 1, 0)


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    """Spherical harmonics are provided in the e3nn library"""
    all_degrees = list(range(2 * max_degree + 1))
    sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
    return torch.split(sh, [d * 2 + 1 for d in all_degrees], dim=1)


@torch.jit.script
def get_basis_script(
    max_degree: int,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
) -> Dict[str, Tensor]:
    basis = {}
    idx = 0
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f"{d_in},{d_out}"
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(
                    torch.einsum(
                        "n f, k l f -> n k l",
                        spherical_harmonics[J].float(),
                        Q_J.float(),
                    )
                )

            basis[key] = (
                torch.stack(K_Js, 3).unsqueeze(2).unsqueeze(1)
            )  # Stack so order is n 1 k 1 l f
            idx += 1

    return basis


def get_basis(relative_pos: Tensor, max_degree: int = 4):
    """
    Returns a dict of basis with keys f'{a},{b}', where a,b = 0,1,...,max_degree.
    """
    spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    clebsch_gordon = get_all_clebsch_gordon(max_degree)
    basis = get_basis_script(max_degree, spherical_harmonics, clebsch_gordon)
    return basis


def precompute_basis(loader: DataLoader, save_dir: str):
    """
    Precompute the basis of dataset and save into a file.
    The basis in stored in a list of dicts, length len(loader),
    where each dict has keys f'{a},{b}', and the values are basis of the particular batch of graphs.

    Note: We can extend this function to hold the basis on memory at the beginning of training script.
    """

    bases = []
    for i, x_T in tqdm(enumerate(loader), total=len(loader)):
        relative_pos = x_T
        basis = get_basis(_get_relative_pos(relative_pos).cuda())
        bases.append({k: v.cpu() for k, v in basis.items()})
    with open(save_dir, "wb") as f:
        pickle.dump(bases, f)
