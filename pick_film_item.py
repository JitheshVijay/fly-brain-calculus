"""Pick the item the film shows: HELD OUT, and answered correctly.

The first cut filmed an item that was in the readout's own training set, where
train accuracy is 1.000 and getting it right means nothing. The item on screen
has to be one the fly had never seen.
"""
import numpy as np, json, scipy.optimize as so
from reservoir import Reservoir, load_graph
import task, readout

CFG = dict(gain=0.08, drive=20.0, bg=0.35, T=12)
NTR, NTE = 4000, 1500

W = load_graph(min_syn=3); m = np.load('out/meta.npz')
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
ro = np.sort(np.random.default_rng(0).choice(W.shape[0], 8000, replace=False))

qs, opts, y = task.make_items(NTR + NTE, np.random.default_rng(7))
X = task.encode(qs, opts, len(pr))
F = Reservoir(W, pr, ro, **CFG).run(X, batch=256)

Xa, mu, sd = readout._prep(F[:NTR]); Xb, _, _ = readout._prep(F[NTR:], mu, sd)
ytr, yte = y[:NTR], y[NTR:]
K, d = 4, Xa.shape[1]; Y = np.eye(K)[ytr]
def obj(w):
    M = w.reshape(d, K); z = Xa @ M; z -= z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    loss = (lse - (z*Y).sum(1)).mean() + 0.5*(M[:-1]**2).sum()/len(Xa)
    p = np.exp(z - lse[:,None]); g = Xa.T @ (p - Y)/len(Xa); g[:-1] += M[:-1]/len(Xa)
    return loss, g.ravel()
M = so.minimize(obj, np.zeros(d*K), jac=True, method='L-BFGS-B',
                options={'maxiter':400}).x.reshape(d, K)

Z = Xb @ M
P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
pred = P.argmax(1)
acc = float((pred == yte).mean())
print(f'held-out accuracy {acc:.3f} over {NTE} items')

right = np.flatnonzero(pred == yte)
conf = P[right, yte[right]]
# a confidently-correct item, but not a degenerate p=1.0 one
order = right[np.argsort(np.abs(conf - 0.93))]
k = int(order[0])
g = NTR + k
print(f'\nchose HELD-OUT item {k} (global {g})')
print('  ' + task.render(qs[g], opts[g], yte[k]))
print('  probabilities: ' + '  '.join(f'{"ABCD"[i]}={P[k,i]:.3f}' for i in range(4)))
print(f'  predicted {"ABCD"[pred[k]]}, correct {"ABCD"[yte[k]]}')

json.dump({'global_index': g, 'question': int(qs[g]),
           'options': opts[g].tolist(), 'label': int(yte[k]),
           'pred': int(pred[k]), 'probs': P[k].tolist(),
           'heldout_accuracy': acc, 'n_train': NTR, 'n_test': NTE},
          open('out/film_item.json', 'w'), indent=2)
print('\nwrote out/film_item.json')
