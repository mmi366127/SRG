import numpy as np

if __name__ == '__main__':

    n, m = 1000, 10

    X = np.random.normal(0, 1, size=(n, m))
    w = np.random.normal(0, 1, size=(1, m))
    e = np.random.standard_cauchy(size=(n))
    y = np.matmul(X, w.T).squeeze() + e
    print(X.shape)
    print(y.shape)
    with open('./dataset/synthetic.txt', 'w') as f:
        for _X, _y in zip(X, y):
            line = f'{_y}'
            for i, val in enumerate(_X):
                line += f' {i}:{val}'
            f.write(line + '\n')

