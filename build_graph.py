"""Build a signed, indexed sparse connectome from MaleCNS v1.0 flat files.

Outputs out/graph.npz:
  indptr/indices/data  CSR of W, shape (N,N), W[i,j] = signed weight of edge i->j
  body_ids             int64 (N,) bodyId per index
  plus out/meta.npz with population index arrays.

Sign convention (standard for Drosophila LIF models):
  acetylcholine +1 | gaba -1 | glutamate -1 (GluCl) | histamine -1
  dopamine/serotonin/octopamine -> 0 for fast transmission; handled as
  neuromodulatory gates instead. unknown -> 0 (edge dropped), counted.
"""
import numpy as np, pyarrow.feather as pf, time, collections, json, os

T0 = time.time()
def log(m): print(f'[{time.time()-T0:6.1f}s] {m}', flush=True)

os.makedirs('out', exist_ok=True)

# ---- annotations -> the neuron universe -------------------------------------
ann = pf.read_table('data/body-annotations-male-cns-v1.0-minconf-0.5.feather')
body = np.asarray(ann.column('bodyId').to_numpy(), dtype=np.int64)
superclass = np.array(ann.column('superclass').to_pylist(), dtype=object)
klass = np.array(ann.column('class').to_pylist(), dtype=object)
ctype = np.array(ann.column('type').to_pylist(), dtype=object)

keep = np.array([s is not None for s in superclass])
body, superclass, klass, ctype = body[keep], superclass[keep], klass[keep], ctype[keep]
order = np.argsort(body)
body, superclass, klass, ctype = body[order], superclass[order], klass[order], ctype[order]
N = len(body)
log(f'neuron universe: {N:,} annotated bodies')

# ---- neurotransmitter -> sign ------------------------------------------------
nt = pf.read_table('data/body-neurotransmitters-male-cns-v1.0.feather')
nt_body = np.asarray(nt.column('body').to_numpy(), dtype=np.int64)
nt_cons = np.array(nt.column('consensus_nt').to_pylist(), dtype=object)
nt_gt = np.array(nt.column('ground_truth').to_pylist(), dtype=object)

SIGN = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0, 'histamine': -1.0,
        'dopamine': 0.0, 'serotonin': 0.0, 'octopamine': 0.0, 'unclear': 0.0}
log('consensus_nt distribution: ' + json.dumps(
    dict(collections.Counter(str(x) for x in nt_cons).most_common(12))))
n_gt = int(sum(1 for x in nt_gt if x))
log(f'ground-truth (experimentally confirmed) NT rows: {n_gt:,} / {len(nt_gt):,}')

pos = {b: i for i, b in enumerate(body)}
sign = np.zeros(N, dtype=np.float32)
known = np.zeros(N, dtype=bool)
gt_mask = np.zeros(N, dtype=bool)
for b, c, g in zip(nt_body, nt_cons, nt_gt):
    i = pos.get(int(b))
    if i is None:
        continue
    s = SIGN.get(str(c).lower())
    if s is not None and s != 0.0:
        sign[i] = s; known[i] = True
    if g:
        gt_mask[i] = True
log(f'signed neurons: {known.sum():,}/{N:,} ({100*known.sum()/N:.1f}%)  '
    f'exc {int((sign>0).sum()):,} inh {int((sign<0).sum()):,}  '
    f'ground-truth-backed {int(gt_mask.sum()):,}')

# ---- edges -------------------------------------------------------------------
w = pf.read_table('data/connectome-weights-male-cns-v1.0-minconf-0.5.feather')
pre = np.asarray(w.column('body_pre').to_numpy(), dtype=np.int64)
post = np.asarray(w.column('body_post').to_numpy(), dtype=np.int64)
wt = np.asarray(w.column('weight').to_numpy(), dtype=np.int64)
del w
log(f'raw edges: {len(pre):,}')

ip = np.searchsorted(body, pre); ip[ip >= N] = 0
iq = np.searchsorted(body, post); iq[iq >= N] = 0
ok = (body[ip] == pre) & (body[iq] == post)
ip, iq, wt = ip[ok], iq[ok], wt[ok]
log(f'edges between annotated neurons: {len(ip):,} ({100*len(ip)/len(ok):.1f}% of raw)')
log(f'  synapses represented: {int(wt.sum()):,}')

ok2 = known[ip]
ip, iq, wt = ip[ok2], iq[ok2], wt[ok2]
log(f'edges with a known presynaptic sign: {len(ip):,} ({100*ok2.mean():.1f}% kept)')

data = (wt.astype(np.float32) * sign[ip])

from scipy.sparse import coo_matrix
W = coo_matrix((data, (ip, iq)), shape=(N, N), dtype=np.float32).tocsr()
W.sum_duplicates()
log(f'CSR built: {W.shape}, nnz={W.nnz:,}')

np.savez('out/graph.npz', indptr=W.indptr, indices=W.indices, data=W.data,
         shape=np.array(W.shape), body_ids=body)

def idx(mask): return np.flatnonzero(mask).astype(np.int32)
pop = {
    'photoreceptor_R16': idx(ctype == 'R1-R6'),
    'photoreceptor_R7':  idx(np.array([bool(t) and t.startswith('R7') for t in ctype])),
    'photoreceptor_R8':  idx(np.array([bool(t) and t.startswith('R8') for t in ctype])),
    'ol_sensory':        idx(superclass == 'ol_sensory'),
    'descending':        idx(superclass == 'descending_neuron'),
    'kenyon_cell':       idx(klass == 'Kenyon_Cell'),
    'mbon':              idx(klass == 'MBON'),
    'dan':               idx(klass == 'DAN'),
    'dan_PAM':           idx(np.array([bool(t) and t.startswith('PAM') for t in ctype]) & (klass == 'DAN')),
    'dan_PPL':           idx(np.array([bool(t) and t.startswith('PPL') for t in ctype]) & (klass == 'DAN')),
    'motor':             idx(np.isin(superclass, ['vnc_motor', 'cb_motor'])),
}
np.savez('out/meta.npz', sign=sign, known=known, ground_truth=gt_mask,
         types=ctype.astype('U32'), superclass=superclass.astype('U32'),
         klass=np.array([k if k else '' for k in klass], dtype='U32'), **pop)
log('populations: ' + ', '.join(f'{k}={len(v)}' for k, v in pop.items()))
log('wrote out/graph.npz + out/meta.npz')
