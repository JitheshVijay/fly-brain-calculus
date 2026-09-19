"""Six mastery curves on one axis: the campaign's central artifact."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('out/results.json'))
sizes, res = R['sizes'], R['results']

STYLE = {
    'real':            ('#1b7f4b', '-',  2.6, 'Real MaleCNS connectome'),
    'sign_shuffled':   ('#c0392b', '--', 1.8, 'E/I signs shuffled'),
    'weight_shuffled': ('#e08b1a', '--', 1.8, 'Synapse counts shuffled'),
    'rewired_degree':  ('#2b6cb0', '--', 1.8, 'Degree-preserving rewire'),
    'erdos_renyi':     ('#7d5ba6', '--', 1.8, 'Erdos-Renyi null'),
    'no_reservoir':    ('#8a8a8a', ':',  1.8, 'No reservoir (encoding only)'),
}

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
for k, (col, ls, lw, lab) in STYLE.items():
    if k not in res: continue
    curves = np.array([r['curve'] for r in res[k]])
    mu = curves.mean(0)
    ax.plot(sizes, mu, ls, color=col, lw=lw, label=lab, marker='o', ms=4)
    if len(curves) > 1:
        ax.fill_between(sizes, curves.min(0), curves.max(0), color=col, alpha=0.13, lw=0)

ax.axhline(0.25, color='#444', lw=1, ls=(0, (2, 3)))
ax.text(sizes[0], 0.258, 'chance (4 options)', fontsize=8.5, color='#444')
ax.set_xscale('log'); ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel('practice items seen')
ax.set_ylabel('held-out accuracy, d/dx $x^n$')
ax.set_title('Can a fruit fly connectome learn the power rule?\n'
             'Same curriculum, same readout, one property destroyed per arm',
             fontsize=11.5, loc='left')
ax.legend(fontsize=8.6, loc='upper left', framealpha=0.95)
ax.grid(alpha=0.22, lw=0.6); ax.set_ylim(0.15, 1.0)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('out/mastery_curves.png')
print('wrote out/mastery_curves.png')

print(f'\n{"arm":>18}  {"held-out":>9}  {"vs real":>8}')
base = np.mean([r['test'] for r in res['real']]) if 'real' in res else float('nan')
for k in STYLE:
    if k not in res: continue
    a = np.array([r['test'] for r in res[k]])
    print(f'{k:>18}  {a.mean():9.3f}  {a.mean()-base:+8.3f}')
