"""Load SWC skeletons into per-neuron line segments, in 8nm dataset units."""
import numpy as np, json, os


def load_swc(path, max_pts=900):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 7:
                continue
            rows.append((int(p[0]), float(p[2]), float(p[3]), float(p[4]), int(p[6])))
    if not rows:
        return None
    ids = np.array([r[0] for r in rows])
    xyz = np.array([(r[1], r[2], r[3]) for r in rows], dtype=np.float32)
    par = np.array([r[4] for r in rows])
    pos = {int(i): k for k, i in enumerate(ids)}
    seg = [(k, pos[int(p)]) for k, p in enumerate(par) if int(p) in pos]
    if not seg:
        return None
    seg = np.array(seg, dtype=np.int32)
    if len(seg) > max_pts:                      # thin long arbours, keep shape
        seg = seg[np.linspace(0, len(seg) - 1, max_pts).astype(int)]
    return xyz[seg[:, 0]], xyz[seg[:, 1]]


def load_cast(cast_path='out/cast.json', skel_dir='skel'):
    cast = json.load(open(cast_path))
    out = []
    for c in cast:
        p = os.path.join(skel_dir, f"{c['body']}.swc")
        if not os.path.exists(p):
            continue
        r = load_swc(p)
        if r is None:
            continue
        out.append(dict(**c, a=r[0], b=r[1]))
    return out


if __name__ == '__main__':
    cast = load_cast()
    A = np.concatenate([c['a'] for c in cast])
    print(f'{len(cast)} neurons, {len(A):,} segments')
    print('bounds (8nm units):')
    for i, ax in enumerate('xyz'):
        print(f'  {ax}: {A[:,i].min():10.0f} .. {A[:,i].max():10.0f}  '
              f'({(A[:,i].max()-A[:,i].min())*8/1000:.0f} um)')
    import collections
    for s in ('ol_sensory', 'descending_neuron', 'vnc_motor', 'cb_intrinsic'):
        sub = np.concatenate([c['a'] for c in cast if c['superclass'] == s])
        print(f'  {s:20s} y range {sub[:,1].min():8.0f}..{sub[:,1].max():8.0f}')
