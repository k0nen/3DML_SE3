import dgl
from torch.utils.data import DataLoader

from dataloader.basis import precompute_basis
from dataloader.qm9 import QM9Dataset

if __name__ == "__main__":
    """
    Precomputing basis for QM9 dataset.
    Make sure you have train/test data in the data directory.
    """

    train_dataset = QM9Dataset(data_dir="data", mode="train", task_idx=0)
    val_dataset = QM9Dataset(data_dir="data", mode="val", task_idx=0)
    test_dataset = QM9Dataset(data_dir="data", mode="test", task_idx=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=lambda samples: dgl.batch([sample[0] for sample in samples]),
    )
    precompute_basis(train_loader, "./basis/qm9_basis_train.pkl")
    precompute_basis(val_loader, "./basis/qm9_basis_val.pkl")
    precompute_basis(test_loader, "./basis/qm9_basis_test.pkl")
