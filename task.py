"""The curriculum: d/dx x^n, as a 4-option flo101-style MCQ.

Shortcut-free by construction. EVERY option has the form (m, m-1), so every
option is "shaped like" a correct power-rule answer and the only way to pick
the right one is to match the option's m against the question's n. That makes
the task pure coincidence detection between two codes.

An earlier version drew distractors from a pool of classic error modes. It
leaked: "pick the option where exponent == coefficient - 1" scored 0.71 without
ever reading the question, and both the reservoir and the no-reservoir control
found that rule and stopped there. audit_shortcuts() exists so that class of
bug cannot pass silently again.
"""
import numpy as np

N_MIN, N_MAX = 2, 12
VAL_MAX = 14


def make_items(n_items, rng):
    """Return (questions, options, labels); options[k] is (4,2) of (coeff, exp)."""
    qs = rng.integers(N_MIN, N_MAX + 1, size=n_items)
    opts = np.zeros((n_items, 4, 2), dtype=np.int64)
    labels = rng.integers(0, 4, size=n_items)
    valid = np.arange(2, VAL_MAX + 1)
    for k in range(n_items):
        n = int(qs[k])
        # Uniform over the whole valid range, NOT a symmetric window around n.
        # A symmetric window makes n the median coefficient far too often
        # ("pick the middle one" scored 0.43 without reading the question).
        pool = list(rng.choice(valid[valid != n], size=3, replace=False))
        j = 0
        for slot in range(4):
            m = n if slot == labels[k] else pool[j]
            if slot != labels[k]:
                j += 1
            opts[k, slot] = (m, m - 1)
    return qs, opts, labels


def _codebook(width, seed=11, density=0.18):
    """Random sparse binary code per value, SHARED across slots.

    A topographic bump code would let a linear readout do arithmetic on bump
    positions. Random codes remove that: deciding whether option i is correct
    requires detecting that two codes COINCIDE, which is multiplicative and
    cannot be done by a linear map on the encoding.
    """
    rng = np.random.default_rng(seed)
    return (rng.random((VAL_MAX + 2, width)) < density).astype(np.float32)


def encode(qs, opts, n_cells):
    """9 slots: [question n] + 4 x [coeff, exp] -> (n_items, n_cells) float32."""
    slots, n_items = 9, len(qs)
    w = n_cells // slots
    CB = _codebook(w)
    X = np.zeros((n_items, n_cells), dtype=np.float32)
    for k in range(n_items):
        X[k, 0:w] = CB[qs[k]]
        for s in range(4):
            b = (1 + 2 * s) * w
            X[k, b:b + w] = CB[opts[k, s, 0]]
            X[k, b + w:b + 2 * w] = CB[opts[k, s, 1]]
    return X


def audit_shortcuts(qs, opts, labels):
    """Accuracy of question-blind heuristics. All must sit at chance (0.25)."""
    n = len(qs)
    rules = {
        'exp == coeff - 1':   lambda k: [i for i in range(4) if opts[k, i, 1] == opts[k, i, 0] - 1],
        'largest coeff':      lambda k: [int(np.argmax(opts[k, :, 0]))],
        'smallest coeff':     lambda k: [int(np.argmin(opts[k, :, 0]))],
        'always slot A':      lambda k: [0],
        '2nd smallest coeff': lambda k: [int(np.argsort(opts[k, :, 0])[1])],
        '3rd smallest coeff': lambda k: [int(np.argsort(opts[k, :, 0])[2])],
    }
    out = {}
    for name, f in rules.items():
        hit = 0.0
        for k in range(n):
            c = f(k)
            if c:
                hit += (labels[k] in c) / len(c)
        out[name] = hit / n
    return out


def render(q, opt, label):
    def fmt(c, e):
        cs = '' if c == 1 else str(c)
        return f'{cs}x^{e}' if e != 1 else f'{cs}x'
    body = '   '.join(f'{"ABCD"[i]}) {fmt(*opt[i])}' for i in range(4))
    return f'd/dx x^{q} = ?   {body}    [correct: {"ABCD"[label]}]'


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    qs, opts, labels = make_items(5, rng)
    for k in range(5):
        print(render(qs[k], opts[k], labels[k]))
    qs, opts, labels = make_items(6000, np.random.default_rng(1))
    print('\nshortcut audit (all must be ~0.25):')
    for name, a in audit_shortcuts(qs, opts, labels).items():
        flag = '  <-- LEAK' if a > 0.30 else ''
        print(f'   {a:.3f}  {name}{flag}')
