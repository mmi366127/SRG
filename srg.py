from torch.utils.data import Sampler
import random, math
import numpy as np

RED = 1
BLACK = 0

class node(object):
    def __init__(self, val: float, idx: int):
        self.size = 1
        self.key = val
        self.sum = val
        self.idx = idx
        self.color = RED
        self.left = None
        self.right = None
        self.parent = None
    
    def __eq__(self, x):
        return x is not None and self.idx == x.idx

def size(x: node) -> int:
        return 0 if x is None else x.size

def sum(x: node) -> float:
    return 0 if x is None else x.sum

def pull(x: node):
    if x is not None:
        x.size = 1 + size(x.left) + size(x.right)
        x.sum = x.key + sum(x.left) + sum(x.right)

class RB_sampler(object):
    def __init__(self, n: int, eps: float) -> None:
        self.n = n
        self.eps = eps
        self.root = None
        self.find_node = [None for i in range(n)]
        for i in range(n):
            self.insert(1.0, i)

    # This seems good
    def left_rotate(self, x: node) -> None:
        temp_x = x
        if x.right is None:
            return
        y = x.right
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        
        y.parent = x.parent
        if x.parent is not None:
            if x == x.parent.left:
                x.parent.left = y
            else:
                x.parent.right = y
        else:
            self.root = y
        
        y.left = x
        x.parent = y
        pull(x)
        pull(y)
        x = temp_x

    # This seems good
    def right_rotate(self, x: node) -> None:
        temp_x = x
        if x.left is None:
            return
        y = x.left

        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        
        y.parent = x.parent

        if x.parent is not None:
            if x == x.parent.left:
                x.parent.left = y
            else:
                x.parent.right = y

        else:
            self.root = y
        
        y.right = x
        x.parent = y
        pull(x)
        pull(y)
        x = temp_x

    # This seems goods
    def insert_fix(self, x: node) -> None:
        temp_x = x
        while x.parent is not None and x.parent.color == RED:
            parent = x.parent
            grantp = parent.parent
            if parent == grantp.left:
                temp = grantp.right
                if temp is not None and temp.color == RED:
                    temp.color = BLACK
                    parent.color = BLACK
                    grantp.color = RED
                    x = grantp
                    continue
                if parent.right == x:
                    self.left_rotate(parent)
                    temp = parent 
                    parent = x
                    x = temp
                parent.color = BLACK
                grantp.color = RED
                self.right_rotate(grantp)
            else:
                temp = grantp.left
                if temp is not None and temp.color == RED:
                    temp.color = BLACK
                    parent.color = BLACK
                    grantp.color = RED
                    x = grantp
                    continue
                if parent.left == x:
                    self.right_rotate(parent)
                    temp = parent
                    parent = x
                    x = temp
                parent.color = BLACK
                grantp.color = RED
                self.left_rotate(grantp)

        self.root.color = BLACK
        x = temp_x

    # This seems good
    def insert(self, val: float, idx: int) -> node:
        x = node(val, idx)
        temp_x = x
        self.find_node[idx] = x
        p, q = self.root, None
        if self.root is None:
            x.color = BLACK
            self.root = x
        else:
            while p is not None:
                q = p
                if p.key < x.key:
                    p = p.right
                else:
                    p = p.left
            x.parent = q
            if q.key < x.key:
                q.right = x
            else:
                q.left = x
        
            temp = x.parent
            while temp is not None:
                pull(temp)
                temp = temp.parent
            self.insert_fix(x)
        return temp_x

    def erase_fix(self, x: node, p: node) -> None:
        temp_x = x
        temp_p = p
        return 
        while (x is None or x.color == BLACK) and x != self.root:
            if p.left is None : break
            if p.right is None : break
            if p.left == x:
                temp = p.right
                if temp.color == RED:
                    temp.color = BLACK
                    p.color = RED
                    self.left_rotate(p)
                    temp = p.right
                
                if temp is None: continue

                if (temp.left is None or temp.left.color == BLACK) and (temp.right is None or temp.right.color == BLACK):
                    temp.color = RED
                    x = p
                    p = x.parent

                else:
                    if temp.right is None or temp.right.color == BLACK:
                        temp.left.color = BLACK
                        temp.color = RED
                        self.right_rotate(temp)
                        temp = p.right
                    
                    temp.color = p.color
                    p.color = BLACK
                    if temp.right is not None:
                        temp.right.color = BLACK
                    self.left_rotate(p)
                    x = self.root
                    break
            else:
                temp = p.left
                if temp.color == RED:
                    temp.color = BLACK
                    p.color = RED
                    self.right_rotate(p)

                if temp is None: continue

                if (temp.left is None or temp.left.color == BLACK) and (temp.right is None or temp.right.color == BLACK):
                    temp.color = RED
                    x = p
                    p = x.parent
                
                else:
                    if temp.left is None or temp.left.color == BLACK:
                        temp.right.color = BLACK
                        temp.color = RED
                        self.left_rotate(temp)
                        temp = p.left
                    
                    temp.color = p.color
                    p.color = BLACK
                    if temp.left is not None:
                        temp.left.color = BLACK
                    self.right_rotate(p)
                    x = self.root
                    break
     
        if x is not None:
            x.color = BLACK
        x = temp_x
        p = temp_p

    def erase(self, p: node) -> None:
        temp_p = p
        if self.root is None or p is None:
            return
        if p.left is not None and p.right is not None:
            r = p.right
            while r.left is not None:
                r = r.left
            
            if p.parent is not None:
                if p.parent.left == p:
                    p.parent.left = r
                else:
                    p.parent.right = r
            else:
                self.root = r

            child = r.right
            parent = r.parent
            color = r.color

            if parent == p:
                parent = r
            else:
                if child is not None:
                    child.parent = parent
                parent.left = child
                r.right = p.right
                p.right.parent = r

            r.parent = p.parent
            r.color = p.color
            r.left = p.left
            p.left.parent = r

            # pull nodes
            temp = parent
            while temp is not None:
                pull(temp)
                temp = temp.parent

            if color == BLACK:
                self.erase_fix(child, parent)

            p = temp_p
            return

        child = p.left if p.left is not None else p.right
        parent = p.parent
        color = p.color

        if child is not None:
            child.parent = parent

        if parent is not None:
            if parent.left == p:
                parent.left = child
            else:
                parent.right = child

        else:
            self.root = child

        temp = parent
        while temp is not None:
            pull(temp)
            temp = temp.parent

        if color == BLACK:
            self.erase_fix(child, parent)
        p = temp_p

    def rank(self, val: float) -> int:
        res = 0
        p = self.root
        while p is not None:
            if p.key > val:
                res += size(p.right) + 1
                p = p.left
            else:
                p = p.right

        return res + 1

    def rank(self, x: node) -> int:
        temp_x = x
        y = x
        x = x.parent
        ret = size(y.right)
        while x is not None:
            if x.right != y:
                ret += size(x.right) + 1
            y = x
            x = x.parent
        x = temp_x
        return ret + 1

    def solve(self):
        v = self.root 
        w = None
        r = size(v.right) + 1
        s = sum(v.right) + v.key
        c = 1.0 - self.eps * (size(self.root) - r)
        while v is not None:
            if v.key >= self.eps * (size(self.root) - r):
                w = v
                v = v.left
                if v is not None:
                    r += size(v.right) + 1
                    s += sum(v.right) + v.key
                    c = 1.0 - self.eps * (self.n - r)
            else:
                v = v.right
                if v is not None:
                    r -= size(v.left) + 1
                    s -= sum(v.left) + v.parent.key
                    c = 1.0 - self.eps * (self.n - r)
        return w, r, s 

    def select_sum(self, x: float) -> node:
        p = self.root
        while x > 0.0:
            if sum(p.right) >= x:
                p = p.right
            else:
                x -= sum(p.right) + p.key
                if x <= 0.0: return p
                p = p.left

    def select_rank(self, rk: int) -> node:
        p = self.root
        while p is not None:
            if size(p.right) >= rk:
                p = p.right
            else:
                rk -= size(p.right) + 1
                if rk == 0: return p
                p = p.left

    def sample(self) -> int:
        w, rho, ps = self.solve()
        u = random.random()
        lambda_ = ps / (1.0 - self.eps * (self.n - rho))
        if u < 1.0 - self.eps * (self.n - rho):
            s = lambda_ * u
            v = self.select_sum(s)
            return v.idx
        else:
            u -= 1.0 - self.eps * (self.n - rho)
            r = self.n - math.floor(u / self.eps)
            v = self.select_rank(r)
            return v.idx

    def inorder(self, x: node):
        if x is None: return
        self.inorder(x.left)
        print(f' {x.key}', end = '')
        self.inorder(x.right)

    def display(self):
        print('Tree: ', end = '')
        self.inorder(self.root)
        print('')

    def update(self, val: float, idx: int):
        self.erase(self.find_node[idx])
        self.find_node[idx] = self.insert(val, idx)


class Naive_sampler(Sampler):
    def __init__(self, numInstance: int):
        self.numInstance = numInstance
        self.lam = np.ones(numInstance) / numInstance
        self.pi = [i for i in range(numInstance)]
        self.p = np.ones(numInstance)
        self.a = np.ones(numInstance)
        self.eps = 1.0 / numInstance
        self.need_update = True
        self.update_set = set()
    
    def update(self, L2_norm: float):
        for idx in self.update_set:
            self.a[idx] = L2_norm
        self.pi.sort(key = lambda x: -self.a[x])
        self.need_update = True
        self.update_set = set()

    def __iter__(self):
        for i in range(self.numInstance):
            if self.need_update:
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
                self.need_update = False
            
            index = np.random.choice(np.arange(self.numInstance), p = self.p)
            self.update_set.add(index)
            yield index


if __name__ == '__main__':

    loli = RB_sampler(n = 10, eps = 0.1)

    for i in range(10):
        loli.insert(random.random(), i)
        loli.display()

    for i in range(10):
        print(f'erase {loli.find_node[i].key}')
        loli.erase(loli.find_node[i])
        loli.display()

    for i in range(10):
        loli.insert(random.random(), i)
        loli.display()

    for i in range(10):
        idx = loli.sample()
        print(idx)
        loli.update(random.random(), i)

    
    