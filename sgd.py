from torch.optim import Optimizer

class SGD_Vanilla(Optimizer):
    """
        Implement stochastic gradient descent
        
        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params, lr=0.001):
        print("Using SGD optimizer ...")
        defaults = dict(lr=lr)
        super(SGD_Vanilla, self).__init__(params, defaults)

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