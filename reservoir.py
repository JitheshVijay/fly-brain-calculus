"""Leaky integrate-and-fire reservoir over the MaleCNS connectome (torch CSR).

The connectome is FROZEN. Nothing inside it is ever trained. Item current is
injected into photoreceptors; spike counts are read from descending neurons.
Encoder and readout port are engineered choices, not claims about what these
cells do in the animal.

Fly photoreceptors are histaminergic, i.e. INHIBITORY, so a network started
from rest cannot propagate anything. A tonic background current (`bg`) puts the
network in a spontaneously active regime that the item's inhibition sculpts.
That is both what makes it work and what the real visual system does.
"""
import numpy as np, torch
from scipy.sparse import csr_matrix, coo_matrix

torch.set_num_threads(6)


def load_graph(path='out/graph.npz', min_syn=3):
    d = np.load(path)
    W = csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
    if min_syn > 1:
        W.data = np.where(np.abs(W.data) >= min_syn, W.data, 0.0)
        W.eliminate_zeros()
    return W


def _to_torch(W):
    """W is i->j; the reservoir needs j<-i, so transpose once here."""
    Wt = W.T.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(Wt.indptr.astype(np.int64)),
        torch.from_numpy(Wt.indices.astype(np.int64)),
        torch.from_numpy(Wt.data.astype(np.float32)), size=Wt.shape)


class Reservoir:
    def __init__(self, W, in_idx, out_idx, gain=0.08, drive=14.0, bg=0.35,
                 decay=0.9, thresh=1.0, T=12):
        self.Wt = _to_torch(W)
        self.N = W.shape[0]
        self.in_idx = torch.from_numpy(np.asarray(in_idx, dtype=np.int64))
        self.out_idx = torch.from_numpy(np.asarray(out_idx, dtype=np.int64))
        self.gain, self.drive, self.bg = gain, drive, bg
        self.decay, self.thresh, self.T = decay, thresh, T

    def run(self, X, batch=256, stats=False):
        n = X.shape[0]
        out = np.zeros((n, len(self.out_idx)), dtype=np.float32)
        rates = []
        for b0 in range(0, n, batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[b0:b0 + batch].T))  # (in, B)
            B = xb.shape[1]
            V = torch.zeros((self.N, B), dtype=torch.float32)
            inj = torch.full((self.N, B), self.bg, dtype=torch.float32)
            inj[self.in_idx] += self.drive * xb
            acc = torch.zeros((len(self.out_idx), B), dtype=torch.float32)
            S = None
            for t in range(self.T):
                V.mul_(self.decay).add_(inj)
                if S is not None:
                    V.add_(torch.sparse.mm(self.Wt, S), alpha=self.gain)
                S = (V > self.thresh).float()
                V.mul_(1.0 - S)                      # spike -> reset to 0
                acc.add_(S.index_select(0, self.out_idx))
                if stats:
                    rates.append(float(S.mean()))
            out[b0:b0 + batch] = acc.T.numpy()
        return (out, np.array(rates)) if stats else out


def arm(W, kind, sign, rng):
    """Null models. Each destroys exactly one property of the real graph."""
    C = W.tocoo()
    r, c, d = C.row.copy(), C.col.copy(), C.data.copy()
    N = W.shape[0]
    if kind == 'real':
        pass
    elif kind == 'sign_shuffled':
        perm = rng.permutation(N)
        ns = np.sign(sign[perm]).astype(np.float32); ns[ns == 0] = 1.0
        d = np.abs(d) * ns[r]
    elif kind == 'weight_shuffled':
        d = np.abs(d)[rng.permutation(len(d))] * np.sign(d)
    elif kind == 'rewired_degree':
        c = c[rng.permutation(len(c))]
    elif kind == 'erdos_renyi':
        r = rng.integers(0, N, len(d)); c = rng.integers(0, N, len(d))
        d = d[rng.permutation(len(d))]
    else:
        raise ValueError(kind)
    M = coo_matrix((d.astype(np.float32), (r, c)), shape=(N, N)).tocsr()
    M.sum_duplicates()
    return M
