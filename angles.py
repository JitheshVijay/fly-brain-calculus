import numpy as np, skeletons, render
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

cast = skeletons.load_cast()
pts, owner, cols = render.build_points(cast, max_per=400)
bright = np.full(len(cast), 0.30, dtype=np.float32)
combos = [(0.0,0.0),(0.0,0.35),(1.571,0.0),(1.571,0.25),(0.7,0.15),(2.6,0.15),(0.0,-0.5),(3.14,0.2)]
fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor='#05070d')
for ax, (yaw, pitch) in zip(axes.ravel(), combos):
    cam = render.Camera(pts, 640, 420); cam.fit(pts, yaw, pitch)
    u, v, d = cam.project_fitted(pts, yaw, pitch)
    img = render.splat(u, v, d, owner, cols, bright, cam.W, cam.H)
    ax.imshow(render.tonemap(render.bloom(img), gain=1.1))
    ax.set_title(f'yaw={yaw:.2f} pitch={pitch:.2f}', color='w', fontsize=9)
    ax.axis('off')
fig.tight_layout(); fig.savefig('out/angles.png', dpi=90, facecolor='#05070d')
print('wrote out/angles.png')
