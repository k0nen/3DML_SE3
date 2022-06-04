import dgl
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import time
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from dataloader.nbody import NBodyDataset, nbody_collate_fn
from model.se3_nbody import SE3Transformer

from pdb import set_trace


def train_epoch(args, epoch, model, train_loader, loss_fn, optimizer):
    model.train()
    loss_epoch = 0
    n_iters = len(train_loader)

    for i, (G, basis, x_T, v_T) in enumerate(train_loader):
        time_model_begin = time.time()
        G = G.to(torch.device("cuda"))
        basis = {k: v.cuda() for k, v in basis.items()}
        x_T = x_T.view(-1, 3).cuda()
        v_T = v_T.view(-1, 3).cuda()

        target = torch.stack([x_T, v_T], dim=1)

        optimizer.zero_grad()
        pred = model(G, basis)
        loss = loss_fn(pred, target)
        loss_epoch += loss.item()

        logging.debug(f"Iter {i} Loss: {loss.item(): .8f}")

        loss.backward()
        optimizer.step()

    loss_epoch /= n_iters
    logging.info(f"Epoch {epoch} train: {loss_epoch: .8f}")
    return loss_epoch


def test_epoch(args, epoch, model, test_loader, loss_fn):
    model.eval()
    loss_x_epoch = 0
    loss_v_epoch = 0
    n_iters = len(test_loader)

    for i, (G, basis, x_T, v_T) in enumerate(test_loader):
        G = G.to(torch.device("cuda"))
        basis = {k: v.cuda() for k, v in basis.items()}
        x_T = x_T.view(-1, 3).cuda()
        v_T = v_T.view(-1, 3).cuda()

        target = torch.stack([x_T, v_T], dim=1)

        pred = model(G, basis)
        x_pred, v_pred = pred.split(1, dim=1)
        loss_x = loss_fn(x_pred.squeeze(1), x_T)
        loss_v = loss_fn(v_pred.squeeze(1), v_T)

        loss_x_epoch += loss_x.item()
        loss_v_epoch += loss_v.item()

    loss_x_epoch /= n_iters
    loss_v_epoch /= n_iters
    loss_epoch = (loss_x_epoch + loss_v_epoch) / 2
    logging.info(
        f"Epoch {epoch} test:  {loss_epoch: .8f} (x: {loss_x_epoch: .8f}, v: {loss_v_epoch: .8f})"
    )
    return loss_epoch


def train_full(args):
    train_dataset = NBodyDataset(
        data_dir=args.train_data_dir,
        train=True,
        basis_dir=args.train_basis_dir,
        target_t=args.target_t,
    )
    test_dataset = NBodyDataset(
        data_dir=args.test_data_dir,
        train=False,
        basis_dir=args.test_basis_dir,
        target_t=args.target_t,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=nbody_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=nbody_collate_fn,
    )

    model = SE3Transformer(
        num_layers=args.num_layers,
        num_channels=args.num_channels,
        num_degrees=args.num_degrees,
        div=args.div,
        n_heads=args.n_heads,
    ).cuda()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1e5

    for i in range(args.epoch):
        train_loss = train_epoch(args, i, model, train_loader, loss_fn, optimizer)
        test_loss = test_epoch(args, i, model, test_loader, loss_fn)

        if test_loss < best_loss:
            torch.save(model.state_dict(), args.best_save_name)
            best_loss = test_loss

    logging.info(f"Best test loss: {test_loss: .8f}")


if __name__ == "__main__":
    args = edict()

    args.train_data_dir = "data/nbody_train.pkl"
    args.train_basis_dir = "basis/nbody_basis_train.pkl"
    args.test_data_dir = "data/nbody_test.pkl"
    args.test_basis_dir = "basis/nbody_basis_test.pkl"

    args.best_save_name = "data/nbody_best.pt"

    # To change this argument, you must re-create the simulation dataset as well.
    args.target_t = 5  # 5

    # Model construction
    args.num_layers = 4  # 4
    args.num_channels = 8  # 8
    args.num_degrees = 4  # 4
    args.div = 4.0  # 4.0
    args.n_heads = 2  # 2

    args.lr = 0.001  # 0.001

    args.batch_size = 128  # 128
    args.epoch = 100  # 100

    args.log_level = "INFO"
    logging.getLogger().setLevel(args.log_level)

    train_full(args)
