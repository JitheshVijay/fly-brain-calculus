"""Record what one calculus question does to the brain, over time.

Runs the LIF network twice on the same background state: once with the item
injected, once with blank input. The per-neuron difference is the item-driven
signal, which is what is worth looking at. Raw firing saturates near 18% and
renders as an undifferentiated blob.
"""
import numpy as np, torch, json
from reservoir import load_graph
import task

T_STEPS = 48
CFG = dict(gain=0.08, bg=0.35, decay=0.9, thresh=1.0)
DRIVE = 20.0


def run_trace(W, pr, X1, watch, T=T_STEPS, drive=DRIVE):
    """Return (T, len(watch)) spike record."""
    from reservoir import _to_torch
    Wt = _to_torch(W)
    N = W.shape[0]
    watch_t = torch.from_numpy(np.asarray(watch, dtype=np.int64))
    V = torch.zeros((N, 1), dtype=torch.float32)
    inj = torch.full((N, 1), CFG['bg'], dtype=torch.float32)
    if X1 is not None:
        inj[torch.from_numpy(pr.astype(np.int64))] += drive * torch.from_numpy(X1[:, None])
    rec = np.zeros((T, len(watch)), dtype=np.float32)
    S = None
    for t in range(T):
        V.mul_(CFG['decay']).add_(inj)
        if S is not None:
            V.add_(torch.sparse.mm(Wt, S), alpha=CFG['gain'])
        S = (V > CFG['thresh']).float()
        V.mul_(1.0 - S)
        rec[t] = S.index_select(0, watch_t)[:, 0].numpy()
    return rec


if __name__ == '__main__':
    W = load_graph(min_syn=3)
    m = np.load('out/meta.npz')
    pr = np.sort(np.concatenate([m['photoreceptor_R16'], m['photoreceptor_R7'],
                                 m['photoreceptor_R8']]))
    cast = json.load(open('out/cast.json'))
    watch = np.array([c['idx'] for c in cast])

    # the film's item is chosen by pick_film_item.py: HELD OUT of the
    # readout's training set, and answered correctly. Never regenerate it here.
    fi = json.load(open('out/film_item.json'))
    qs = np.array([fi['question']])
    opts = np.array([fi['options']])
    y = np.array([fi['label']])
    X = task.encode(qs, opts, len(pr))
    print('item:', task.render(qs[0], opts[0], y[0]), flush=True)
    print(f"  held out; fly predicted {'ABCD'[fi['pred']]} at p={max(fi['probs']):.3f}", flush=True)

    on = run_trace(W, pr, X[0], watch)
    off = run_trace(W, pr, None, watch)
    diff = on - off
    np.savez('out/activity.npz', on=on, off=off, diff=diff, watch=watch,
             question=qs[0], options=opts[0], label=y[0],
             pred=fi['pred'], conf=max(fi['probs']))
    print(f'traced {T_STEPS} steps x {len(watch)} cast neurons')
    print(f'  population rate on/off: {on.mean()*100:.1f}% / {off.mean()*100:.1f}%')
    print(f'  cast neurons ever differing: {(np.abs(diff).sum(0) > 0).mean()*100:.1f}%')
    frac = np.abs(diff).mean(1)
    print('  item-driven divergence per step:',
          ' '.join(f'{v*100:.1f}' for v in frac[:16]), '...')
