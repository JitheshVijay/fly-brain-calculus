"""Render the film: one fly brain, one calculus question, real activity.

Frames are the point-splat render with per-neuron brightness driven by the
item-driven activity trace, under a slow camera sweep. Text is composited on
top with PIL.
"""
import numpy as np, json, os, shutil, sys
from PIL import Image, ImageDraw, ImageFont
import skeletons, render

W, H, FPS = 1920, 1080, 30
SUB = 12                       # frames per simulation step
PITCH = -0.45
YAW_SWEEP = 0.38
OUT = 'frames'

FONT_DIR = '/System/Library/Fonts/Supplemental'
def font(sz, bold=False):
    p = f"{FONT_DIR}/Arial{' Bold' if bold else ''}.ttf"
    return ImageFont.truetype(p, sz)

def draw_power(d, xy, base, expo, size, fill, bold=False):
    """Draw `<base>x` with a real raised exponent.

    Arial has no glyphs for most Unicode superscripts, so translating digits
    into U+2070.. renders tofu boxes. Draw the exponent as smaller raised text
    instead; it always works and reads better.
    """
    x, y = xy
    d.text((x, y), base, font=font(size, bold), fill=fill)
    x += d.textlength(base, font=font(size, bold))
    if expo is not None:
        d.text((x + size * 0.03, y - size * 0.30), str(expo),
               font=font(int(size * 0.62), bold), fill=fill)
        x += d.textlength(str(expo), font=font(int(size * 0.62), bold)) + size * 0.06
    return x


def opt_parts(c, e):
    cs = '' if c == 1 else str(c)
    return f'{cs}x', (e if e != 1 else None)

INK, DIM, HOT = (236, 240, 248), (128, 140, 165), (120, 240, 210)
REVEAL = 430                   # frame the answer starts lighting up (of 576)


def main():
    A = np.load('out/activity.npz')
    diff, q, opts, lab = A['diff'], int(A['question']), A['options'], int(A['label'])
    pred, conf = int(A['pred']), float(A['conf'])
    T = diff.shape[0]

    cast = skeletons.load_cast()
    pts, owner, cols = render.build_points(cast, max_per=700)
    cam = render.Camera(pts, W, H, margin=0.80)
    cam.fit(pts, 0.0, PITCH)
    print(f'{len(cast)} neurons, {len(pts):,} points', flush=True)

    # per-neuron activity envelope: rectified item-driven difference, decayed
    act = np.abs(diff)
    env = np.zeros_like(act)
    running = np.zeros(act.shape[1], dtype=np.float32)
    for t in range(T):
        running = 0.82 * running + act[t]
        env[t] = running
    env /= (np.percentile(env, 99.5) + 1e-6)
    env = np.clip(env, 0, 1.6)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    total = T * SUB
    for f in range(total):
        t = f / SUB
        i0 = min(int(t), T - 1); i1 = min(i0 + 1, T - 1); frac = t - i0
        e = (1 - frac) * env[i0] + frac * env[i1]

        bright = 0.16 + 1.5 * e.astype(np.float32)
        yaw = YAW_SWEEP * np.sin(2 * np.pi * f / total)
        u, v, d = cam.project_fitted(pts, yaw, PITCH)
        img = render.splat(u, v, d, owner, cols, bright, W, H)
        rgb = render.tonemap(render.bloom(img, passes=3, k=0.6), gain=1.15)

        im = Image.fromarray((rgb * 255).astype(np.uint8))
        overlay(im, f, total, i0, q, opts, lab, float(np.abs(diff[i0]).mean()),
                pred, conf)
        im.save(f'{OUT}/f{f:05d}.png')
        if f % 60 == 0:
            print(f'  frame {f}/{total}', flush=True)
    print(f'rendered {total} frames')


def overlay(im, f, total, step, q, opts, lab, div, pred, conf):
    d = ImageDraw.Draw(im, 'RGBA')
    fade = lambda a, b: max(0.0, min(1.0, (f - a) / max(1, b - a)))
    al = lambda c, a: (c[0], c[1], c[2], int(255 * max(0.0, min(1.0, a))))

    # title, first 3s
    a = min(fade(6, 30), 1 - fade(80, 110))
    if a > 0.01:
        d.text((90, 84), 'One fruit fly.', font=font(62, True), fill=al(INK, a))
        d.text((90, 158), '166,700 neurons. 25,582,938 connections.',
               font=font(30), fill=al(DIM, a))

    # the question, from 3s on
    a = fade(95, 125)
    if a > 0.01:
        x = draw_power(d, (90, 84), 'd/dx   x', q, 58, al(INK, a), True)
        d.text((x + 14, 84), '= ?', font=font(58, True), fill=al(INK, a))
        for i in range(4):
            reveal = fade(REVEAL + i * 10, REVEAL + 34 + i * 10)
            correct = (i == lab)
            lit = correct and reveal > 0.01
            col = HOT if lit else INK
            aa = a * (0.70 + 0.30 * reveal) if lit else a * (0.72 - 0.34 * reveal)
            b, e = opt_parts(*opts[i])
            xx = 96
            d.text((xx, 178 + i * 48), f'{"ABCD"[i]})  ', font=font(32, lit), fill=al(col, aa))
            xx += d.textlength(f'{"ABCD"[i]})  ', font=font(32, lit))
            draw_power(d, (xx, 178 + i * 48), b, e, 32, al(col, aa), lit)
        r2 = fade(REVEAL + 60, REVEAL + 100)
        if r2 > 0.01:
            d.text((96, 386), f'the fly answered {"ABCD"[pred]}   ·   p = {conf:.2f}',
                   font=font(27, True), fill=al(HOT, r2 * a))
            d.text((96, 428), 'held-out item. the fly had never seen it.',
                   font=font(22), fill=al(DIM, r2 * a))

    # persistent instrument readout
    a = fade(95, 130)
    if a > 0.01:
        y0 = im.height - 132
        d.text((90, y0), f'step {step + 1:02d} / 48', font=font(24), fill=al(DIM, a))
        d.text((90, y0 + 34), f'item-driven divergence from baseline   {div*100:4.1f}%',
               font=font(24), fill=al(DIM, a))
        bw = 420
        d.rectangle([90, y0 + 74, 90 + bw, y0 + 80], fill=al((40, 48, 70), a))
        d.rectangle([90, y0 + 74, 90 + int(bw * min(1, div / 0.09)), y0 + 80],
                    fill=al(HOT, a))

    # credit, right-aligned so it never clips
    cr = 'MaleCNS v1.0  ·  HHMI Janelia FlyEM + Cambridge + Google Research  ·  CC-BY'
    d.text((im.width - 90 - d.textlength(cr, font=font(20)), im.height - 62),
           cr, font=font(20), fill=al(DIM, 0.75 * fade(20, 60)))


if __name__ == '__main__':
    main()
