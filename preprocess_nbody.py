import dgl
from torch.utils.data import DataLoader

from dataloader.basis import precompute_basis
from dataloader.nbody import NBodyDataset

if __name__ == "__main__":
    """
    Precomputing basis for n-body dataset.
    Make sure you have train/test data in the data directory.
    """

    train_dataset = NBodyDataset(
        data_dir="data/nbody_train.pkl", train=True, target_t=5
    )
    test_dataset = NBodyDataset(data_dir="data/nbody_test.pkl", train=False, target_t=5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]),
    )
    precompute_basis(train_loader, "basis/nbody_basis_train.pkl")
    precompute_basis(test_loader, "basis/nbody_basis_test.pkl")
