import numpy as np, time
from reservoir import Reservoir, load_graph
import task, readout
W = load_graph(min_syn=3); m = np.load('out/meta.npz')
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
dn = m['descending']
rng = np.random.default_rng(7); NTR, NTE = 1200, 500
qs, opts, y = task.make_items(NTR + NTE, rng)
X = task.encode(qs, opts, len(pr)); ytr, yte = y[:NTR], y[NTR:]
tr, te = readout.fit(X[:NTR], ytr, X[NTR:], yte)
print(f'no_reservoir : train {tr:.3f}  held-out {te:.3f}   (chance 0.250)', flush=True)
for drive, T in ((6.0,12), (20.0,12), (20.0,24)):
    r = Reservoir(W, pr, dn, gain=0.08, drive=drive, bg=0.35, T=T)
    t0=time.time(); F = r.run(X, batch=256); dt=time.time()-t0
    tr, te = readout.fit(F[:NTR], ytr, F[NTR:], yte)
    print(f'real drive={drive:5.1f} T={T:2d} : train {tr:.3f}  held-out {te:.3f}  ({dt:.0f}s)', flush=True)
