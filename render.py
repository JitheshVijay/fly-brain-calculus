"""Additive point-splat renderer for connectome skeletons.

matplotlib line collections cannot draw 750k segments 600 times. Projecting to
a float buffer with bincount and blooming it is both far faster and closer to
the look we want: neurons as light, not as wireframe.
"""
import numpy as np

BG = np.array([0.020, 0.028, 0.055], dtype=np.float32)

# population -> base colour (RGB 0-1)
PALETTE = {
    'ol_sensory':        (0.20, 0.95, 0.95),
    'ol_intrinsic':      (0.16, 0.62, 0.86),
    'visual_projection': (0.35, 0.80, 1.00),
    'visual_centrifugal':(0.30, 0.70, 0.95),
    'cb_sensory':        (1.00, 0.78, 0.30),
    'cb_intrinsic':      (0.95, 0.62, 0.35),
    'cb_motor':          (1.00, 0.55, 0.40),
    'ascending_neuron':  (0.70, 0.85, 0.55),
    'descending_neuron': (1.00, 0.42, 0.55),
    'vnc_intrinsic':     (0.62, 0.48, 0.92),
    'vnc_sensory':       (0.55, 0.65, 0.95),
    'vnc_motor':         (0.98, 0.85, 0.55),
}
DEFAULT = (0.60, 0.65, 0.80)


def build_points(cast, max_per=700):
    """Flatten the cast into one point cloud + a per-point neuron index."""
    pts, owner, cols = [], [], []
    for k, c in enumerate(cast):
        P = np.concatenate([c['a'], c['b']])
        if len(P) > max_per:
            P = P[np.linspace(0, len(P) - 1, max_per).astype(int)]
        pts.append(P)
        owner.append(np.full(len(P), k, dtype=np.int32))
        cols.append(PALETTE.get(c['superclass'], DEFAULT))
    return (np.concatenate(pts).astype(np.float32),
            np.concatenate(owner),
            np.array(cols, dtype=np.float32))


class Camera:
    def __init__(self, pts, W=1280, H=720, margin=0.90):
        self.W, self.H = W, H
        self.centre = pts.mean(0)
        self.radius = np.abs(pts - self.centre).max()
        self.margin = margin

    def fit(self, pts, yaw, pitch, zoom=1.0):
        """Set scale/offset from the PROJECTED extent, not the 3D radius."""
        x, y, _ = self._rot(pts, yaw, pitch)
        sx = self.margin * self.W / (np.ptp(x) + 1e-6)
        sy = self.margin * self.H / (np.ptp(y) + 1e-6)
        self.s = min(sx, sy) * zoom
        self.cx = 0.5 * (x.min() + x.max())
        self.cy = 0.5 * (y.min() + y.max())
        return self

    def _rot(self, pts, yaw, pitch):
        P = pts - self.centre
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        x = P[:, 0] * cy + P[:, 2] * sy
        z = -P[:, 0] * sy + P[:, 2] * cy
        y = P[:, 1] * cp - z * sp
        d = P[:, 1] * sp + z * cp
        return x, y, d

    def project_fitted(self, pts, yaw, pitch):
        x, y, d = self._rot(pts, yaw, pitch)
        u = ((x - self.cx) * self.s + self.W / 2).astype(np.int32)
        v = ((y - self.cy) * self.s + self.H / 2).astype(np.int32)
        return u, v, d

    def project(self, pts, yaw, pitch=0.16, zoom=1.0):
        P = pts - self.centre
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        x = P[:, 0] * cy + P[:, 2] * sy
        z = -P[:, 0] * sy + P[:, 2] * cy
        y = P[:, 1] * cp - z * sp
        d = P[:, 1] * sp + z * cp
        s = self.margin * min(self.W, self.H) / (2 * self.radius) * zoom
        u = (x * s + self.W / 2).astype(np.int32)
        v = (y * s + self.H / 2).astype(np.int32)      # +y is down = brain on top
        return u, v, d


def splat(u, v, d, owner, colours, bright, W, H, depth_fade=0.55):
    """Accumulate RGB with a mild depth cue. bright is per-neuron 0..N."""
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d, o = u[ok], v[ok], d[ok], owner[ok]
    dn = (d - d.min()) / (np.ptp(d) + 1e-6)
    w = (1.0 - depth_fade * dn).astype(np.float32) * bright[o]
    flat = v * W + u
    img = np.zeros((H * W, 3), dtype=np.float32)
    for ch in range(3):
        img[:, ch] = np.bincount(flat, weights=w * colours[o, ch], minlength=H * W)
    return img.reshape(H, W, 3)


def bloom(img, passes=3, k=0.55):
    """Cheap separable blur, added back for glow."""
    g = img.copy()
    for _ in range(passes):
        g = (g + np.roll(g, 1, 0) + np.roll(g, -1, 0)) / 3.0
        g = (g + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 3.0
    return img + k * g


def tonemap(img, gain=1.0, gamma=0.72):
    x = np.clip(img * gain, 0, None)
    x = 1.0 - np.exp(-x)                     # soft shoulder, no clipped whites
    x = x ** gamma                           # lift midtones, keep arbours legible
    return np.clip(BG + x * (1.0 - BG), 0, 1)


def to_png(img, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.image as mi
    mi.imsave(path, np.clip(img, 0, 1))
