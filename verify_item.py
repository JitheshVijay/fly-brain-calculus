"""Check what the fly ACTUALLY answered on the item shown in the film.

The overlay says "the fly answered C". That has to be a measurement, not a
caption. Train the readout exactly as experiment.py does, then predict the one
item the activity trace was recorded from.
"""
import numpy as np, json
from reservoir import Reservoir, load_graph
import task, readout

CFG = dict(gain=0.08, drive=20.0, bg=0.35, T=12)
NTR = 4000

W = load_graph(min_syn=3); m = np.load('out/meta.npz')
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
ro = np.sort(np.random.default_rng(0).choice(W.shape[0], 8000, replace=False))

qs, opts, y = task.make_items(NTR, np.random.default_rng(7))          # training items
A = np.load('out/activity.npz')
q1, o1, l1 = np.array([A['question']]), A['options'][None], int(A['label'])

Xtr = task.encode(qs, opts, len(pr))
X1 = task.encode(q1, o1, len(pr))
res = Reservoir(W, pr, ro, **CFG)
Ftr = res.run(Xtr, batch=256)
F1 = res.run(X1, batch=1)

Z, mu, sd = readout._prep(Ftr)
import scipy.optimize as so
Wm = None
tr, te = readout.fit(Ftr, y, F1, np.array([l1]))
print(f'readout trained on {NTR} items; train acc {tr:.3f}')

# recover the prediction explicitly
Xa, mu, sd = readout._prep(Ftr)
Xb, _, _ = readout._prep(F1, mu, sd)
K = 4; d = Xa.shape[1]; Y = np.eye(K)[y]
def obj(w):
    M = w.reshape(d, K); z = Xa @ M; z -= z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1)); loss = (lse - (z*Y).sum(1)).mean() + 0.5*(M[:-1]**2).sum()/len(Xa)
    p = np.exp(z - lse[:,None]); g = Xa.T @ (p - Y)/len(Xa); g[:-1] += M[:-1]/len(Xa)
    return loss, g.ravel()
r = so.minimize(obj, np.zeros(d*K), jac=True, method='L-BFGS-B', options={'maxiter':400})
M = r.x.reshape(d, K)
logits = (Xb @ M)[0]
p = np.exp(logits - logits.max()); p /= p.sum()
pred = int(np.argmax(logits))
print(f'\nitem: {task.render(q1[0], o1[0], l1)}')
print(f'  probabilities: ' + '  '.join(f'{"ABCD"[i]}={p[i]:.3f}' for i in range(4)))
print(f'  PREDICTED: {"ABCD"[pred]}    CORRECT: {"ABCD"[l1]}    -> {"RIGHT" if pred==l1 else "WRONG"}')
json.dump({'pred': pred, 'label': l1, 'probs': p.tolist()}, open('out/film_item_prediction.json','w'))
