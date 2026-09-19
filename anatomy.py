"""Labelled anatomy still: what you are looking at, and what the colours mean."""
import numpy as np, json, collections
from PIL import Image, ImageDraw
import skeletons, render
from movie import font, INK, DIM, HOT

W, H, PITCH = 2000, 1250, -0.45
cast = skeletons.load_cast()
pts, owner, cols = render.build_points(cast, max_per=700)
cam = render.Camera(pts, W, H, margin=0.62); cam.fit(pts, 0.0, PITCH)
u, v, d = cam.project_fitted(pts, 0.0, PITCH)
bright = np.full(len(cast), 0.34, dtype=np.float32)
img = render.splat(u, v, d, owner, cols, bright, W, H)
im = Image.fromarray((render.tonemap(render.bloom(img), gain=1.25) * 255).astype(np.uint8))
dr = ImageDraw.Draw(im, 'RGBA')

# centroid of each superclass in screen space, for leader lines
cen = {}
for k, c in enumerate(cast):
    cen.setdefault(c['superclass'], []).append(k)
own_u = {k: [] for k in range(len(cast))}
for i in range(0, len(u), 7):
    own_u[owner[i]].append((u[i], v[i]))

def centroid(sc, side=None):
    """side='left'/'right' for bilateral populations.

    Averaging both optic lobes puts the anchor in the middle of the central
    brain, which is exactly where the optic lobe is not."""
    P = [p for k in cen.get(sc, []) for p in own_u[k]]
    if not P: return None
    A = np.array(P, dtype=float)
    if side:
        mid = A[:, 0].mean()
        A = A[A[:, 0] < mid] if side == 'left' else A[A[:, 0] > mid]
    return A[:, 0].mean(), A[:, 1].mean()

CALLOUTS = [
    ('ol_sensory',        'left optic lobe\n6,091 photoreceptors, every one inhibitory', (-520, -190), 'left'),
    ('cb_intrinsic',      'central brain\nwhere the item gets mixed',          (330, -280), None),
    ('descending_neuron', 'neck connective\n1,314 descending neurons',         (430, 30),   None),
    ('vnc_motor',         'ventral nerve cord\nthe insect spinal cord',        (400, 180),  None),
]
for sc, label, off, side in CALLOUTS:
    c = centroid(sc, side)
    if c is None: continue
    x, y = c; tx, ty = x + off[0], y + off[1]
    dr.line([(x, y), (tx + (60 if off[0] < 0 else 0), ty + 26)],
            fill=(*DIM, 150), width=2)
    dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(*HOT, 210), width=2)
    for i, ln in enumerate(label.split('\n')):
        dr.text((tx, ty + i * 32), ln, font=font(27 if i == 0 else 22, i == 0),
                fill=(*(INK if i == 0 else DIM), 235))

dr.text((70, 62), 'Drosophila melanogaster, adult male central nervous system',
        font=font(40, True), fill=(*INK, 255))
dr.text((70, 118), f'{len(cast)} of 166,700 neurons drawn from real reconstructed morphology  ·  '
        'MaleCNS v1.0  ·  CC-BY', font=font(23), fill=(*DIM, 240))

# colour key
y0 = H - 250
dr.text((70, y0 - 44), 'colour = neuron class', font=font(22, True), fill=(*DIM, 240))
items = [(s, render.PALETTE[s]) for s in
         ('ol_sensory', 'ol_intrinsic', 'visual_projection', 'cb_intrinsic',
          'descending_neuron', 'vnc_intrinsic', 'vnc_motor')]
for i, (s, c) in enumerate(items):
    yy = y0 + (i % 4) * 34
    xx = 70 + (i // 4) * 330
    rgb = tuple(int(255 * v) for v in c)
    dr.rectangle([xx, yy + 8, xx + 26, yy + 20], fill=(*rgb, 255))
    dr.text((xx + 40, yy), s.replace('_', ' '), font=font(21), fill=(*DIM, 235))

im.save('out/anatomy.png')
print('wrote out/anatomy.png', im.size)
