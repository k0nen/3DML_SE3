import dgl
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import time
import os
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict

from dataloader.qm9 import QM9Dataset, qm9_collate_fn
from model.se3_qm9 import SE3Transformer
from model.utils import num_parameters

from pdb import set_trace

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


def train_epoch(args, epoch, model, train_loader, loss_fn, optimizer, writer):
    model.train()
    loss_epoch = 0
    n_iters = len(train_loader)

    for i, (G, basis, y_task) in enumerate(train_loader):
        time_model_begin = time.time()
        G = G.to(torch.device("cuda"))
        basis = {k: v.cuda().detach() for k, v in basis.items()}
        y_task = y_task.cuda()

        for param in model.parameters():
            param.grad = None

        pred = model(G, basis)
        loss = loss_fn(pred, y_task)
        loss_epoch += loss.item()

        if i % 100 == 0:
            logging.debug(f"Iter {i} Loss: {loss.item(): .8f}")
            writer.add_scalar('train_loss_dense', loss.item(), n_iters * epoch + i)

        loss.backward()
        optimizer.step()

    loss_epoch /= n_iters
    logging.info(f"Epoch {epoch} train: {loss_epoch: .8f}")
    return loss_epoch


def test_epoch(args, epoch, model, test_loader, loss_fn):
    model.eval()
    loss_epoch = 0
    n_iters = len(test_loader)

    with torch.no_grad():
        for i, (G, basis, y_task) in enumerate(test_loader):
            time_model_begin = time.time()
            G = G.to(torch.device("cuda"))
            basis = {k: v.cuda() for k, v in basis.items()}
            y_task = y_task.cuda()

            pred = model(G, basis)
            loss = loss_fn(pred, y_task)
            loss_epoch += loss.item()

            if i % 100 == 0:
                logging.debug(f"Iter {i} Loss: {loss.item(): .8f}")

    loss_epoch /= n_iters
    logging.info(f"Epoch {epoch} test: {loss_epoch: .8f}")
    return loss_epoch


def train_full(args):
    if len(args.run_name) > 0:
        writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    else:
        writer = SummaryWriter()
    
    train_dataset = QM9Dataset(
        data_dir=args.train_data_dir,
        mode='train',
        task_idx=args.task_idx,
        basis_dir=args.train_basis_dir,
    )
    # val_dataset = QM9Dataset(
    #     data_dir=args.val_data_dir,
    #     mode="val",
    #     task_idx=args.task_idx,
    #     basis_dir=args.val_basis_dir,
    # )
    test_dataset = QM9Dataset(
        data_dir=args.test_data_dir,
        mode="test",
        task_idx=args.task_idx,
        basis_dir=args.test_basis_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=qm9_collate_fn,
        num_workers=0,
    )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=qm9_collate_fn,
    #     num_workers=16,
    # )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=qm9_collate_fn,
        num_workers=0,
    )

    # shape_in: QM9 dataset has 6 atomic features and 4 edge features.
    model = SE3Transformer(
        num_layers=args.num_layers,
        num_channels=args.num_channels,
        num_degrees=args.num_degrees,
        edge_dim=4,
        div=args.div,
        n_heads=args.n_heads,
    ).cuda()
    logging.info(f"Parameter count: {num_parameters(model)}")

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1e5

    for i in range(args.epoch):
        train_loss = train_epoch(args, i, model, train_loader, loss_fn, optimizer, writer)
        writer.add_scalar('train_loss', train_loss, i)
        
        if i % 2 == 0:
            test_loss = test_epoch(args, i, model, test_loader, loss_fn)
            writer.add_scalar('test_loss', test_loss, i)
            if test_loss < best_loss:
                torch.save(model.state_dict(), args.best_save_name)
                best_loss = test_loss

    logging.info(f"Best test loss: {best_loss: .8f}")
    writer.flush()


if __name__ == "__main__":
    args = edict()
    
    args.run_name = "qm9_homo_run2"

    args.train_data_dir = "./data"
    args.val_data_dir = "./data"
    args.test_data_dir = "./data"
    args.train_basis_dir = "./basis/qm9_basis_train.pkl"
    args.val_basis_dir = "./basis/qm9_basis_val.pkl"
    args.test_basis_dir = "./basis/qm9_basis_test.pkl"
    args.best_save_name = "./data/qm9_best_homo_run2.pt"

    args.task_idx = 2

    args.num_layers = 7  # 7
    args.num_channels = 32  # 32
    args.num_degrees = 4  # 4
    args.div = 2.0  # 2.0
    args.n_heads = 8  # 8

    args.lr = 0.001  # 0.001

    args.batch_size = 32  # 32
    args.epoch = 101  # 51

    args.log_level = "DEBUG"
    logging.getLogger().setLevel(args.log_level)
    
    torch.set_num_threads(32)

    train_full(args)
