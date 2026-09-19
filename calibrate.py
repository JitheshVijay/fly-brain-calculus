import numpy as np, time
from scipy.sparse import csr_matrix
from reservoir import Reservoir
import task

d = np.load('out/graph.npz'); m = np.load('out/meta.npz')
W = csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
dn = m['descending']
rng = np.random.default_rng(0)
qs, opts, labels = task.make_items(64, rng)
X = task.encode(qs, opts, len(pr))

print(f'{"bg":>5} {"gain":>7} {"drive":>6} | {"rate t0":>8} {"rate tEnd":>9} | '
      f'{"DN>0":>6} {"between-item feat std":>21}  time')
for bg in (0.25, 0.35):
    for gain in (0.01, 0.03, 0.08):
        for drive in (4.0,):
            r = Reservoir(W, pr, dn, gain=gain, drive=drive, bg=bg, T=20)
            t0 = time.time(); F, rates = r.run(X, batch=64, stats=True); dt = time.time()-t0
            Fn = F / max(F.max(), 1)
            print(f'{bg:5.2f} {gain:7.3f} {drive:6.1f} | {rates[1]*100:7.2f}% {rates[-1]*100:8.2f}% | '
                  f'{(F>0).mean()*100:5.1f}% {Fn.std(axis=0).mean():21.4f}  {dt:.1f}s')
