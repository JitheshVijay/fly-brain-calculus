"""Closing card: the finding, over a clean dimmed render of the brain.

Must render its own background. Reusing the last film frame drags that frame's
baked-in question overlay along with it and the two collide.
"""
import numpy as np, json
from PIL import Image, ImageDraw
import skeletons, render
from movie import font, INK, DIM, HOT, W, H, PITCH

N_OUT = 150                                   # 5s at 30fps

R = json.load(open('out/results.json'))
def mean(k): return float(np.mean([r['test'] for r in R['results'][k]]))

cast = skeletons.load_cast()
pts, owner, cols = render.build_points(cast, max_per=700)
cam = render.Camera(pts, W, H, margin=0.80); cam.fit(pts, 0.0, PITCH)
A = np.load('out/activity.npz')
env = np.abs(A['diff']).sum(0); env = env / (np.percentile(env, 99.5) + 1e-6)
bright = 0.16 + 0.9 * np.clip(env, 0, 1.6).astype(np.float32)
u, v, d = cam.project_fitted(pts, 0.0, PITCH)
img = render.splat(u, v, d, owner, cols, bright, W, H)
base = Image.fromarray(
    (render.tonemap(render.bloom(img, passes=3, k=0.6), gain=0.42) * 255).astype(np.uint8))

VALUE_X = 1010                 # numbers right-align here; Arial is proportional
LINES = [
    (74, True,  INK, 'The fly learns the power rule.', None),
    (32, False, DIM, f"{mean('real'):.3f} held out, against {mean('no_reservoir'):.3f} with the brain taken out.", None),
    (0,  False, DIM, '', None),
    (50, True,  HOT, 'Its own wiring is not why.', None),
    (0,  False, DIM, '', None),
    (29, False, DIM, 'Erdos-Renyi random graph, matched edge count', f"{mean('erdos_renyi'):.3f}"),
    (29, False, DIM, 'Degree-preserving rewire', f"{mean('rewired_degree'):.3f}"),
    (29, False, DIM, 'Excitatory / inhibitory signs shuffled', f"{mean('sign_shuffled'):.3f}"),
    (29, True,  INK, 'The real MaleCNS connectome', f"{mean('real'):.3f}"),
    (0,  False, DIM, '', None),
    (26, False, INK, 'Same curriculum every arm. We ran the controls and published them.', None),
]

for i in range(N_OUT):
    im = base.copy()
    d2 = ImageDraw.Draw(im, 'RGBA')
    # gradient scrim: a hard-edged rectangle leaves a visible seam
    scrim = Image.new('L', (W, 1))
    px = scrim.load()
    for x in range(W):
        t = min(1.0, max(0.0, (x - 900) / 420.0))
        px[x, 0] = int(190 * (1.0 - t) ** 1.5)
    scrim = scrim.resize((W, H))
    im.paste(Image.new('RGB', (W, H), (5, 7, 13)), (0, 0), scrim)
    d2 = ImageDraw.Draw(im, 'RGBA')
    a = min(1.0, i / 20.0) * min(1.0, (N_OUT - i) / 18.0)
    y = 250
    for n, (sz, bold, col, txt, val) in enumerate(LINES):
        if not txt:
            y += 30; continue
        fa = a * min(1.0, max(0.0, (i - 4 - n * 3) / 16.0))
        if fa > 0.004:
            fill = (col[0], col[1], col[2], int(255 * fa))
            d2.text((130, y), txt, font=font(sz, bold), fill=fill)
            if val:
                fo = font(sz, bold)
                d2.text((VALUE_X - d2.textlength(val, font=fo), y), val,
                        font=fo, fill=fill)
        y += int(sz * 1.60)
    d2.text((130, H - 118), 'flo101.com', font=font(24),
            fill=(*DIM, int(255 * a * 0.85)))
    im.save(f'frames/f{576 + i:05d}.png')
print(f'wrote {N_OUT} outro frames')
