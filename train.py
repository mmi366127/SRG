import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
import torch
from torch import nn 
from torch.utils.data import DataLoader

from utils import AverageCalculator, accuracy, plot_train_stats, loss_fn, max_and_average_L
from datasets import Mushrooms, Phishing, W8A, IJCNN1
from srg import Naive_sampler
from svrg import SVRG, SVRG_Snapshot
from sgd import SGD_Vanilla
from model import Model

# Some macros
OUTPUT_DIR = "outputs"

BATCH_SIZE_LARGE = 256

def train_SGD_one_iter(model, optimizer, train_loader, loss_fn, device):
    model.train()
    loss = AverageCalculator()
    acc = AverageCalculator()
    
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        yhat = model(X)
        loss_iter = loss_fn(yhat, y, model)

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        # logging 
        acc_iter = accuracy(yhat, y)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return model.w.weight.data, loss.avg, acc.avg

def train_SRG_one_iter(model, optimizer, train_loader, loss_fn, device, sampler):
    model.train()
    loss = AverageCalculator()
    acc = AverageCalculator()
    
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        yhat = model(X)

        loss_iter, L2_norm = loss_fn(yhat, y, model, return_norm = True)
        # sampler.update(L2_norm)

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        acc_iter = accuracy(yhat, y)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)

    return model.w.weight.data ,loss.avg, acc.avg

def train_SVRG_one_iter(model_k, model_snapshot, optimizer_inner, optimizer_snapshot, train_loader, snapshot_loader, loss_fn, device):
    model_k.train()
    model_snapshot.train()
    loss = AverageCalculator()
    acc = AverageCalculator()

    optimizer_snapshot.zero_grad()
    for X, y in snapshot_loader:
        X = X.to(device)
        y = y.to(device)
        yhat = model_snapshot(X)
        snapshot_loss = loss_fn(yhat, y, model) / len(snapshot_loader)
        snapshot_loss.backward()
    
    # pass the current paramesters of optimizer_0 to optimizer_k 
    mu = optimizer_snapshot.get_param_groups()
    optimizer_inner.set_mu(mu)
    l2_reg = 1 / len(train_loader)

    for X, y in train_loader:
        # flatten the imags
        X = X.to(device)
        y = y.to(device)

        # compute loss
        yhat = model_k(X)
        loss_iter = loss_fn(yhat, y, model_k, L2_reg = l2_reg)

        optimizer_inner.zero_grad()
        loss_iter.backward()

        optimizer_snapshot.zero_grad()
        yhat_snap = model_snapshot(X)
        loss_snap = loss_fn(yhat_snap, y, model)
        loss_snap.backward()

        optimizer_inner.step(optimizer_snapshot.get_param_groups())

        # logging
        acc_iter = accuracy(yhat, y)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)

    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_inner.get_param_groups())
    
    return model_k.w.weight.data, loss.avg, acc.avg

def validate(model, val_loader, loss_fn, device):
    """
        Validation
    """
    model.eval()
    loss = AverageCalculator()
    acc = AverageCalculator()

    for X, y in val_loader:
        X = X.to(device)
        yhat = model(X)
        y = y.to(device)

        # calculating loss and accuracy
        loss_iter = loss_fn(yhat, y)
        acc_iter = accuracy(yhat, y)
        
        # logging 
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train classifiers via SGD and SVRG on MNIST dataset.")

    parser.add_argument('--optimizer', type=str, default="SGD",
                        help="optimizer")
    parser.add_argument('--dataset', type=str, default="Mushrooms",
                        help="dataset")
    parser.add_argument('--n_iter', type=int, default=30,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=None,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--store_stats_interval', type=int, default=10,
                        help="how often the training statistics are stored.")
    parser.add_argument('--save', type=bool, default=True,
                        help="Do you want save the result?")

    # Configuring the device: CPU or GPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print("Using device: {}".format(device))

    args = parser.parse_args()
    args_dict = vars(args)

    if args.dataset == 'Mushrooms':
        train_set = Mushrooms()
    
    elif args.dataset == 'Phishing':
        train_set = Phishing()
    
    elif args.dataset == 'W8A':
        train_set = W8A()

    elif args.dataset == 'IJCNN1':
        train_set = IJCNN1()

    else:
        raise ValueError("Unknown dataset!")

    print(f'Data size: {len(train_set)}, # of Features: {train_set.numFeatures}')

    if args.optimizer == 'SRG':
        mySampler = Naive_sampler(train_set.numInstance)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=mySampler)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    snapshot_loader = DataLoader(train_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)

    model = Model(train_set.numFeatures).to(device)

    if args.optimizer == 'SVRG':
        model_snapshot = Model(train_set.numFeatures).to(device)

    if args.lr is not None:
        lr = args.lr  # learning rate
    else:
        max_L, ave_L = max_and_average_L(train_set)
        lr = 1.0 / (2.0 * ((len(train_set) - args.batch_size) / (args.batch_size * (len(train_set) - 1)) * max_L + (len(train_set) * (args.batch_size - 1)) / (args.batch_size * (len(train_set) - 1)) * ave_L))
    print(f'learning rate: {lr}')
    n_iter = args.n_iter  # the number of training iterations
    stats_interval = args.store_stats_interval # the period of storing training statistics

    # the optimizer 
    if args.optimizer == "SGD" or args.optimizer == "SRG":
        optimizer = SGD_Vanilla(model.parameters(), lr=lr)
    elif args.optimizer == "SVRG":
        optimizer = SVRG(model.parameters(), lr=lr)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())
    else:
        raise ValueError("Unknown optimizer!")

    # Create a folder for storing output results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = timestamp + "_" + args.optimizer
        log_dir = os.path.join(OUTPUT_DIR, model_name)
        if not os.path.isdir(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args_dict, f)

    # Store training stats
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    for iteration in range(n_iter):
        t0 = time.time()

        # Training 
        if args.optimizer == "SGD":
            para_weights, train_loss, train_acc = train_SGD_one_iter(model, optimizer, train_loader, loss_fn, device)
        elif args.optimizer == "SVRG":
            para_weights, train_loss, train_acc = train_SVRG_one_iter(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, snapshot_loader, loss_fn, device)
        elif args.optimizer == "SRG":
            para_weights, train_loss, train_acc = train_SRG_one_iter(model, optimizer, train_loader, loss_fn, device, mySampler)
        else:
            raise ValueError("Unknown optimizer")
        
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        
        print_format = "iteration: {}, train loss: {:.4f}, train acc: {:.4f}, time: {:.2f} sec"
        print(print_format.format(iteration, train_loss, train_acc, time.time() - t0))

        # save data and plot 
        if (iteration + 1) % stats_interval == 0:
            np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all),
                val_loss=np.array(val_loss_all), val_acc=np.array(val_acc_all), para_weights = para_weights.cpu().numpy())
            plot_train_stats(train_loss_all, val_loss_all, train_acc_all, val_acc_all, log_dir, acc_low=0.9)
    
    # Training finished
    open(os.path.join(log_dir, 'done'), 'a').close()