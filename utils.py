import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, random
import numpy as np
import os

class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)

def loss_fn(y, target, model, L2_reg = 0.0001, return_norm = False):
    # binary cross entropy + L2 regularization
    L2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    if return_norm:
        # return torch.log(1.0 + torch.exp(-target * y)) + L2_reg * L2_norm, L2_norm
        return F.binary_cross_entropy_with_logits(y, target) + L2_reg * L2_norm, L2_norm
    # return torch.sum(torch.log(1.0 + torch.exp(-target * y))) + L2_reg * L2_norm
    # return F.binary_cross_entropy(y, target) + L2_reg * L2_norm
    return F.binary_cross_entropy_with_logits(y, target) + L2_reg * L2_norm

def accuracy(yhat, labels):
    # yhat = torch.sigmoid(yhat)
    return (torch.where(yhat > 0.5, 1, 0) == labels).sum().data.item() / float(len(labels))

def plot_train_stats(train_loss, val_loss, train_acc, val_acc, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey='row')
    axes[0][0].plot(np.array(train_loss))
    axes[0][0].set_title("Training Loss")
    axes[0][1].plot(np.array(val_loss))
    axes[0][1].set_title("Validation Loss")
    axes[1][0].plot(np.array(train_acc))
    axes[1][0].set_title("Training Accuracy")
    axes[1][0].set_ylim(acc_low, 1)
    axes[1][1].plot(np.array(val_acc))
    axes[1][1].set_title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_stats.png'))
    plt.close()

def max_and_average_L(dataset):
    res_max, res_ave = 0.0, 0.0
    for x_i, y_i in dataset:
        _norm = torch.norm(x_i).data.item()
        _norm = _norm * _norm
        res_ave += _norm
        res_max = max(res_max, _norm)
    res_max += 1.0 / len(dataset)
    res_ave = (res_ave + 1) / len(dataset)
    return res_max, res_ave
