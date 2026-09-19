"""Choose a stratified cast of neurons to render: enough to read as a brain,
few enough to draw 60 times a second."""
import numpy as np, json

m = np.load('out/meta.npz')
d = np.load('out/graph.npz')
body = d['body_ids']; sc = m['superclass']; kl = m['klass']; ty = m['types']
rng = np.random.default_rng(4)

QUOTA = {                      # superclass -> how many to draw
    'ol_intrinsic': 170, 'cb_intrinsic': 190, 'vnc_intrinsic': 90,
    'visual_projection': 70, 'ol_sensory': 60, 'cb_sensory': 35,
    'vnc_sensory': 35, 'ascending_neuron': 30, 'descending_neuron': 90,
    'vnc_motor': 40, 'visual_centrifugal': 20,
}
NAMED = {'kenyon_cell': 70, 'mbon': 60, 'dan': 40}

pick = []
for s, k in QUOTA.items():
    idx = np.flatnonzero(sc == s)
    if len(idx) == 0: continue
    pick.append(rng.choice(idx, min(k, len(idx)), replace=False))
for key, k in NAMED.items():
    idx = m[key]
    pick.append(rng.choice(idx, min(k, len(idx)), replace=False))

sel = np.unique(np.concatenate(pick))
meta = [{'idx': int(i), 'body': int(body[i]), 'superclass': str(sc[i]),
         'klass': str(kl[i]), 'type': str(ty[i])} for i in sel]
json.dump(meta, open('out/cast.json', 'w'))
print(f'selected {len(sel)} neurons')
import collections
for s, c in collections.Counter(x['superclass'] for x in meta).most_common():
    print(f'  {c:4d}  {s}')
