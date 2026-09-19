import numpy as np, time, json
import skeletons, render

t0=time.time()
cast = skeletons.load_cast()
pts, owner, cols = render.build_points(cast)
cam = render.Camera(pts, 1600, 900)
print(f'{len(cast)} neurons, {len(pts):,} points, load {time.time()-t0:.1f}s', flush=True)
bright = np.full(len(cast), 0.30, dtype=np.float32)
u, v, d = cam.project(pts, yaw=0.55)
img = render.splat(u, v, d, owner, cols, bright, cam.W, cam.H)
render.to_png(render.tonemap(render.bloom(img), gain=2.2), 'out/still_brain.png')
print(f'wrote out/still_brain.png in {time.time()-t0:.1f}s')
