import numpy as np

epsilon = 1e-5

class ICA:
    def __init__(self, x):
        self.x = np.matrix(x)

    def ica(self): #独立成分分析
        self.fit()
        z = self.whiten()
        y = self.analyze(z)
        return y

    def fit(self): #平均を0にする
        self.x -= self.x.mean(axis=1)

    def whiten(self): #白色化
        sigma = np.cov(self.x, rowvar=True, bias=True)
        D, E = np.linalg.eigh(sigma)
        E = np.asmatrix(E)
        Dh = np.diag(np.array(D) ** (-1/2))
        V = E * Dh * E.T
        z = V * self.x
        return z

    def normalize(self, x): #正規化
        if x.sum() < 0:
            x *= -1
        return x / np.linalg.norm(x)

    def analyze(self, z):
        c, r = self.x.shape
        W = np.empty((0, c))
        for _ in range(c): #観測数分だけアルゴリズムを実行する
            vec_w = np.random.rand(c, 1)
            vec_w = self.normalize(vec_w)
            while True:
                vec_w_prev = vec_w
                vec_w = np.asmatrix((np.asarray(z) * np.asarray(vec_w.T * z) ** 3).mean(axis=1)).T - 3 * vec_w
                vec_w = self.normalize(np.linalg.qr(np.asmatrix(np.concatenate((W, vec_w.T), axis=0)).T)[0].T[-1].T) #直交化法と正規化
                if np.linalg.norm(vec_w - vec_w_prev) < epsilon: #収束判定
                    W = np.concatenate((W, vec_w.T), axis=0)
                    break
        y = W * z
        return y
