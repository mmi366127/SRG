from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, random
import numpy as np
import os

class SRG_sampler:
    def __init__(self, numInstance: int):
        self.numInstance = numInstance
        self.lam = np.ones(numInstance) / numInstance
        self.pi = [i for i in range(numInstance)]
        self.p = np.ones(numInstance)
        self.a = np.ones(numInstance)
        self.eps = 1.0 / numInstance


    def update(self, idx: int, L2_norm):
        self.a[idx] = L2_norm 
        self.pi.sort(key = lambda x: -self.a[x])

    def sample(self):
        # return random.randint(0, self.numInstance - 1)

        # calculate lambda(i)
        part_sum = 0
        for i in range(1, self.numInstance + 1):
            idx = self.pi[i - 1] + 1
            part_sum += self.a[idx - 1]
            self.lam[idx - 1] = part_sum / (1 - (self.numInstance - i) * self.eps)

        # calculate rho
        rho = 0
        for i in range(self.numInstance):
            if self.a[i] >= self.eps * self.lam[i]:
                rho = i + 1

        # calculate p
        for i in range(self.numInstance):
            idx = self.pi[i] + 1
            if idx <= rho:
                self.p[i] = self.a[i] / self.lam[rho - 1]
            else:
                self.p[i] = self.eps
    
        self.p /= np.sum(self.p)

        return np.random.choice(np.arange(self.numInstance), p = self.p)

# class SRG_sampler(WeightedRandomSampler):
#     def __init__(self, numInstances: int):
#         samples_weight = np.ones(numInstances) / numInstances
#         samples_weight = torch.from_numpy(samples_weight)
#         super(SRG_sampler, self).__init__(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

#     def update(self):
#         pass

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

