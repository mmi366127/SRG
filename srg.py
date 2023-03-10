from torch.utils.data import Sampler
from torch.optim import Optimizer
import random, ctypes
from ctypes import *
import numpy as np

lib = ctypes.CDLL('./cpp/liblibrary-python.so')

lib.getObject.argtypes = [c_int, c_double]
lib.insertObject.argtypes = [c_double, c_int]
lib.updateObject.argtypes = [c_int, c_double]
lib.getWeight.restype = c_double
lib.sample.restype = c_int

from torch.optim import Optimizer

class SRG(Optimizer):
    """
        This class is for calculating the gradient of one iteration.
        
        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params, lr):
        print("Using SRG optimizer ...")
        defaults = dict(lr=lr)
        super(SRG, self).__init__(params, defaults)
    
    def add_grad(self, params, weight):
        """
            Set the mean gradient for the current iteration. 
        """
        for my_group, new_group in zip(self.param_groups, params):  
            for mu, new_mu in zip(my_group['params'], new_group['params']):
                if mu.grad is None : continue
                mu.grad += weight * new_mu.grad

    def step(self):
        """
            This function implements a single optimization step via SGD        
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-group['lr'])

class SRG_cal(Optimizer):
    """
        This class for calculating the average gradient (i.e., snapshot) of all samples.

        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SRG_cal, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups

class Naive_sampler(Sampler):
    def __init__(self, numInstance: int):
        self.numInstance = numInstance
        self.lam = np.ones(numInstance)
        self.pi = [i for i in range(numInstance)]
        self.a = np.zeros(numInstance)
        self.p = np.ones(numInstance)
        self.eps = 0.5 / numInstance
        self.need_update = True
        self.update_list = []
        self.weight = []
    
    def update(self, L2_norm: float):
        # print(self.update_list)
        for (i, idx) in enumerate(self.update_list):
            self.a[idx] = L2_norm[i]
        # print(L2_norm)

        # self.a /= np.sum(self.a)
        self.pi.sort(key = lambda x: -self.a[x])
        self.need_update = True
        self.update_list = []
        self.weight = []

    def get_weight(self):
        return self.weight

    def __len__(self):
        return self.numInstance

    def __iter__(self):
        for i in range(self.numInstance):
            if self.need_update and np.sum(self.a) > 0.000000001:
                # Naive_sampler.first = True
                # calculate lambda(i)
                part_sum = 0
                for i in range(1, self.numInstance + 1):
                    idx = self.pi[i - 1] + 1
                    part_sum += self.a[idx - 1]
                    self.lam[i - 1] = part_sum / (1 - (self.numInstance - i) * self.eps)

                # calculate rho
                rho = 0
                for i in range(self.numInstance):
                    idx = self.pi[i]
                    if self.a[idx] >= self.eps * self.lam[i]:
                        rho = i + 1

                # calculate p
                for i in range(self.numInstance):
                    idx = self.pi[i]
                    if idx <= rho - 1:
                        self.p[i] = self.a[i] / self.lam[rho - 1]
                    else:
                        self.p[i] = self.eps
        
                # print(np.sum(self.p))
                # print(self.a)
                self.need_update = False
            
            self.p /= np.sum(self.p)
            index = np.random.choice(np.arange(self.numInstance), p=self.p)
            self.weight.append(1.0 / (self.p[index] * self.numInstance))
            self.update_list.append(index)
            yield index

class RBTree_sampler(object):
    def __init__(self, numInstance):
        lib.getObject(c_int(numInstance), c_double(0.99999 / numInstance))
        self.numInstance = numInstance
        for i in range(numInstance):
            lib.insertObject(c_double(1.0), c_int(i))
        self.update_list = []
        self.weight = []

    def __del__(self):
        lib.delObject()

    def update(self, L: list):
        for (i, norm) in enumerate(L):
            self.update_(norm, self.update_list[i])
        self.update_list = []
        self.weight = []

    def get_weight(self):
        return self.weight

    def update_(self, val: float, idx: int):
        lib.updateObject(c_int(idx), c_double(val))

    def sample(self) -> int:
        index = int(lib.sample())
        w = float(lib.getWeight())
        self.update_list.append(index)
        self.weight.append(w)
        return index

    def __len__(self):
        return self.numInstance

    def __iter__(self):
        for i in range(self.numInstance):
            index = self.sample()
            yield index

if __name__ == '__main__':

    loli = RBTree_sampler(10)

    for i in range(1000):
        idx = loli.sample()
        loli.update(random.random(), idx)

    
    