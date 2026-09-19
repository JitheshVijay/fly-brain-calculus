"""Does the real MaleCNS wiring help a fly learn the power rule?

Six arms. Each null destroys exactly ONE property of the real graph, so a gap
between arms is attributable to that property. All arms share the same items,
the same readout neurons and the same readout fitting procedure.

  real            the connectome as reconstructed
  sign_shuffled   same topology + magnitudes, E/I identity permuted
  weight_shuffled same topology + signs, synapse counts permuted across edges
  rewired_degree  in/out degree sequences preserved, specific wiring destroyed
  erdos_renyi     random graph, matched edge count and weight distribution
  no_reservoir    linear readout straight on the encoding (is the fly decorative?)
"""
import numpy as np, json, time
from reservoir import Reservoir, load_graph, arm
import task, readout

NTR, NTE = 4000, 1500
SIZES = [50, 100, 200, 400, 800, 1600, 4000]
NULL_SEEDS = [0, 1, 2]
READOUT_WIDTH = 8000
CFG = dict(gain=0.08, drive=20.0, bg=0.35, T=12)
MIN_SYN = 3

T0 = time.time()
def log(m): print(f'[{(time.time()-T0)/60:5.1f}m] {m}', flush=True)

W = load_graph(min_syn=MIN_SYN)
m = np.load('out/meta.npz'); sign = m['sign']
pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'], m['photoreceptor_R8']]))
dn = m['descending']
ro = np.sort(np.random.default_rng(0).choice(W.shape[0], READOUT_WIDTH, replace=False))

rng = np.random.default_rng(7)
qs, opts, y = task.make_items(NTR + NTE, rng)
audit = task.audit_shortcuts(qs, opts, y)
assert max(audit.values()) < 0.32, f'task leaks: {audit}'
X = task.encode(qs, opts, len(pr))
ytr, yte = y[:NTR], y[NTR:]
log(f'items {NTR}+{NTE}  nnz {W.nnz:,}  readout {READOUT_WIDTH}  cfg {CFG}')
log('shortcut audit: ' + ', '.join(f'{k}={v:.3f}' for k, v in audit.items()))

results, extra = {}, {}

def record(name, F):
    tr, te = readout.fit(F[:NTR], ytr, F[NTR:], yte)
    curve = readout.learning_curve(F[:NTR], ytr, F[NTR:], yte, SIZES)
    results.setdefault(name, []).append(dict(train=tr, test=te, curve=curve.tolist()))
    log(f'{name:>16}  train {tr:.3f}  held-out {te:.3f}   curve {np.round(curve,3).tolist()}')

record('no_reservoir', X)

log('real connectome ...')
Freal = Reservoir(W, pr, ro, **CFG).run(X, batch=256)
record('real', Freal)

log('side result: descending-neuron readout only ...')
Fdn = Reservoir(W, pr, dn, **CFG).run(X, batch=256)
tr, te = readout.fit(Fdn[:NTR], ytr, Fdn[NTR:], yte)
extra['real_descending_only'] = dict(train=tr, test=te, n_readout=int(len(dn)))
log(f'  descending-only: train {tr:.3f}  held-out {te:.3f}')

log('readout-width scaling on the real arm ...')
widths = {}
for w in (250, 1000, 4000, 8000):
    sub = np.sort(np.random.default_rng(5).choice(READOUT_WIDTH, w, replace=False))
    _, te = readout.fit(Freal[:NTR][:, sub], ytr, Freal[NTR:][:, sub], yte)
    widths[w] = te
    log(f'  width {w:5d}: held-out {te:.3f}')
extra['width_scaling'] = widths
del Fdn

for kind in ('sign_shuffled', 'weight_shuffled', 'rewired_degree', 'erdos_renyi'):
    for s in NULL_SEEDS:
        log(f'{kind} seed {s} ...')
        Wn = arm(W, kind, sign, np.random.default_rng(1000 + s))
        record(kind, Reservoir(Wn, pr, ro, **CFG).run(X, batch=256))
        del Wn

json.dump(dict(config=CFG, min_syn=MIN_SYN, readout_width=READOUT_WIDTH,
               sizes=SIZES, n_train=NTR, n_test=NTE, shortcut_audit=audit,
               extra=extra, results=results),
          open('out/results.json', 'w'), indent=2)
log('wrote out/results.json')

print('\n=== SUMMARY (held-out accuracy, chance = 0.250) ===')
base = np.mean([r['test'] for r in results['real']])
for k, v in results.items():
    a = np.array([r['test'] for r in v])
    sd = f' +/- {a.std():.3f}' if len(a) > 1 else '        '
    print(f'  {k:>16}  {a.mean():.3f}{sd}   vs real {a.mean()-base:+.3f}   (n={len(a)})')
