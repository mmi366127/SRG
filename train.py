import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
import torch
from torch import nn 
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import AverageCalculator, accuracy, plot_train_stats, loss_fn, max_and_average_L, LRScheduler
from datasets import Mushrooms, Phishing, W8A, IJCNN1, SYNTHETIC
from model import Model, LinearRegression
from srg import Naive_sampler, SRG, SRG_cal
from svrg import SVRG, SVRG_Snapshot
from sgd import SGD_Vanilla

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
        loss_iter = loss_fn(yhat, y, model, L2_reg=0.5/len(train_loader))

        optimizer.zero_grad()
        loss_iter.backward()
        optimizer.step()

        # logging 
        acc_iter = accuracy(yhat, y)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return model.w.weight.data, loss.avg, acc.avg

def train_SRG_one_iter(model, optimizer, optimizer_cal, train_loader, loss_fn, device, sampler):
    model.train()
    loss = AverageCalculator()
    acc = AverageCalculator()
    
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        weight = torch.tensor(sampler.get_weight()).to(device)
        upd = []
        optimizer.zero_grad()
        for i, (X_, y_) in enumerate(zip(X, y)):
            X_ = X_.unsqueeze(0)
            y_ = y_.unsqueeze(0)
            yhat = model(X_)
            
            optimizer_cal.zero_grad()
            loss_iter = loss_fn(yhat, y_, model, L2_reg=0.5/len(train_loader))
            loss_iter.backward()
            L2_norm = 0
            with torch.no_grad():
                for group in optimizer_cal.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            L2_norm += p.grad.data.pow(2.0).sum()
                        
            upd.append(L2_norm)
            
            acc_iter = accuracy(yhat, y)

            loss.update(loss_iter.data.item())
            acc.update(acc_iter)

            optimizer.add_grad(optimizer_cal.get_param_groups(), weight[i])

        sampler.update(upd)
        optimizer.step()

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
    l2_reg = 0.5 / len(train_loader)

    for X, y in train_loader:
        # flatten the imags
        X = X.to(device)
        y = y.to(device)

        # compute loss
        yhat = model_k(X)
        loss_iter = loss_fn(yhat, y, model_k, L2_reg=l2_reg)

        optimizer_inner.zero_grad()
        loss_iter.backward()

        optimizer_snapshot.zero_grad()
        yhat_snap = model_snapshot(X)
        loss_snap = loss_fn(yhat_snap, y, model, L2_reg=l2_reg)
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
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            yhat = model(X)
            y = y.to(device)

            # calculating loss and accuracy
            loss_iter = loss_fn(yhat, y, model, L2_reg=0.5/len(train_loader))
            acc_iter = accuracy(yhat, y)
            # print(loss_iter, acc_iter, yhat)
            
            # logging 
            loss.update(loss_iter.data.item())
            acc.update(acc_iter)
    
    return loss.avg, acc.avg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train classifiers via SGD and SVRG on MNIST dataset.")

    parser.add_argument('--optimizer', type=str, default="GD",
                        help="optimizer")
    parser.add_argument('--dataset', type=str, default="Mushrooms",
                        help="dataset")
    parser.add_argument('--n_iter', type=int, default=20,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=None,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size")
    parser.add_argument('--store_stats_interval', type=int, default=10,
                        help="how often the training statistics are stored.")
    parser.add_argument('--save', type=bool, default=True,
                        help="Do you want save the result?")
    parser.add_argument('--save_iter_weight', type=bool, default=True,
                        help="Do you want to save the weight when training?")
    parser.add_argument('--lr_decay', help="Do you want to use lr scheduler?", action="store_true")


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
    
    elif args.dataset == 'SYN':
        loss_fn = lambda x, y, _, L2_reg=0 : F.mse_loss(x, y)
        train_set = SYNTHETIC()

    else:
        raise ValueError("Unknown dataset!")

    print(f'Data size: {len(train_set)}, # of Features: {train_set.numFeatures}')

    if args.optimizer == 'SRG':
        mySampler = Naive_sampler(train_set.numInstance)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=mySampler)
    elif args.optimizer == 'GD':
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    snapshot_loader = DataLoader(train_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)
    valid_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'SYN':
        model = LinearRegression(train_set.numFeatures).to(device)
    else:
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
    if args.optimizer == "SGD" or args.optimizer == "GD":
        optimizer = SGD_Vanilla(model.parameters(), lr=lr)
    elif args.optimizer == "SVRG":
        optimizer = SVRG(model.parameters(), lr=lr)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())
    elif args.optimizer == "SRG":
        optimizer = SRG(model.parameters(), lr=lr)
        optimizer_cal = SRG_cal(model.parameters())
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
    train_para_all = []

    if args.lr_decay:
        scheduler = LRScheduler(optimizer, lr)

    for iteration in range(n_iter):
        t0 = time.time()

        # Training 
        if args.optimizer == "SGD" or args.optimizer == "GD":
            para_weights, train_loss, train_acc = train_SGD_one_iter(model, optimizer, train_loader, loss_fn, device)
        elif args.optimizer == "SVRG":
            para_weights, train_loss, train_acc = train_SVRG_one_iter(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, snapshot_loader, loss_fn, device)
        elif args.optimizer == "SRG":
            para_weights, train_loss, train_acc = train_SRG_one_iter(model, optimizer, optimizer_cal, train_loader, loss_fn, device, mySampler)
        else:
            raise ValueError("Unknown optimizer")
        
        _loss, _acc = validate(model, valid_loader, loss_fn, device)
        print(_loss, _acc)

        train_loss_all.append(_loss)
        train_acc_all.append(_acc)
        train_para_all.append(para_weights.cpu().numpy())
        
        print_format = "iteration: {}, train loss: {:.4f}, train acc: {:.4f}, valid loss: {:.4f}, valid acc: {:.4f}, time: {:.2f} sec"
        print(print_format.format(iteration, train_loss, train_acc, _loss, _acc, time.time() - t0))

        if args.lr_decay:
            scheduler.step()

        # save data and plot 
        if (iteration + 1) % stats_interval == 0:
            if args.save_iter_weight:
                np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                    train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all), para_weights=np.array(train_para_all),
                    val_loss=np.array(val_loss_all), val_acc=np.array(val_acc_all))
            else:
                np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                    train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all), para_weight=para_weights.cpu().numpy(),
                    val_loss=np.array(val_loss_all), val_acc=np.array(val_acc_all))
            plot_train_stats(train_loss_all, val_loss_all, train_acc_all, val_acc_all, log_dir, acc_low=0.9)
    
    # Training finished
    open(os.path.join(log_dir, 'done'), 'a').close()