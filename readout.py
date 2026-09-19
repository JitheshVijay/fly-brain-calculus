"""L2-regularised multinomial logistic readout + evaluation helpers."""
import numpy as np
from scipy.optimize import minimize


def _prep(F, mu=None, sd=None):
    if mu is None:
        mu = F.mean(0)
        sd = F.std(0)
        sd[sd < 1e-6] = 1.0   # constant columns -> contribute nothing, never inf
    Z = (F - mu) / sd
    return np.hstack([Z, np.ones((len(Z), 1), np.float32)]).astype(np.float64), mu, sd


def fit(Ftr, ytr, Fte, yte, C=1.0, K=4):
    Xtr, mu, sd = _prep(Ftr)
    Xte, _, _ = _prep(Fte, mu, sd)
    d = Xtr.shape[1]
    Y = np.eye(K)[ytr]

    def obj(w):
        Wm = w.reshape(d, K)
        z = Xtr @ Wm
        z -= z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        loss = (lse - (z * Y).sum(1)).mean() + 0.5 * C * (Wm[:-1] ** 2).sum() / len(Xtr)
        p = np.exp(z - lse[:, None])
        g = Xtr.T @ (p - Y) / len(Xtr)
        g[:-1] += C * Wm[:-1] / len(Xtr)
        return loss, g.ravel()

    r = minimize(obj, np.zeros(d * K), jac=True, method='L-BFGS-B',
                 options={'maxiter': 400})
    Wm = r.x.reshape(d, K)
    acc = lambda X, y: float((np.argmax(X @ Wm, 1) == y).mean())
    return acc(Xtr, ytr), acc(Xte, yte)


def learning_curve(Ftr, ytr, Fte, yte, sizes, C=1.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    out = []
    for n in sizes:
        s = idx[:n]
        _, te = fit(Ftr[s], ytr[s], Fte, yte, C=C)
        out.append(te)
    return np.array(out)
