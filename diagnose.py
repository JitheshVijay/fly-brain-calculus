"""Is the negative result about the connectome, or about my readout port?

If the item is decodable from the encoding but NOT from descending-neuron
activity, the reservoir is destroying information rather than failing to
compute with it, and the comparison says nothing about the wiring.
"""
import numpy as np, time
from reservoir import Reservoir, load_graph
import task, readout

W = load_graph(min_syn=3); m = np.load('out/meta.npz')
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
dn, N = m['descending'], W.shape[0]
rng = np.random.default_rng(3)
NTR, NTE = 1200, 500
qs, opts, y = task.make_items(NTR + NTE, rng)
X = task.encode(qs, opts, len(pr))

# probe target: the question exponent n, 11 classes. Pure input readability.
nlab = qs - task.N_MIN
K = task.N_MAX - task.N_MIN + 1

def probe(F, name):
    tr, te = readout.fit(F[:NTR], nlab[:NTR], F[NTR:], nlab[NTR:], K=K)
    print(f'  decode question n from {name:28s}: train {tr:.3f}  held-out {te:.3f}   (chance {1/K:.3f})', flush=True)
    return te

print('=== can the item even be read back? ===', flush=True)
probe(X, 'the raw encoding')

wide = np.sort(np.random.default_rng(0).choice(N, 20000, replace=False))
r = Reservoir(W, pr, dn, gain=0.08, drive=20.0, bg=0.35, T=12)
r.out_idx = __import__('torch').from_numpy(wide)
t0 = time.time(); Fw = r.run(X, batch=256); print(f'  (wide readout run {time.time()-t0:.0f}s)', flush=True)
probe(Fw[:, np.searchsorted(wide, dn[np.isin(dn, wide)])], f'{np.isin(dn,wide).sum()} descending (subset)')
probe(Fw, '20000 random neurons')

print('\n=== does a wider readout rescue the TASK? ===', flush=True)
ytr, yte = y[:NTR], y[NTR:]
tr, te = readout.fit(Fw[:NTR], ytr, Fw[NTR:], yte)
print(f'  power rule from 20000 neurons : train {tr:.3f}  held-out {te:.3f}   (chance 0.250)', flush=True)

act = (Fw > 0).mean(0)
print(f'\n  readout neurons that ever spike: {(act>0).mean()*100:.1f}%   '
      f'that vary across items: {(Fw.std(0) > 1e-6).mean()*100:.1f}%')
