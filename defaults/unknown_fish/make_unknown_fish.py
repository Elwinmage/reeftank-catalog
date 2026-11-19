"""Build the atlas of the placeholder fish: a black fish with a white "?".

The atlas follows the layout of the other fish of the catalog: 8 columns of
208x112 frames, a swim clip (0-42), an idle clip (43-87) and a turn clip
(88-113). The fish is drawn with vector shapes, then animated by shifting its
columns up and down (a travelling wave, larger at the tail).

    python make_unknown_fish.py <out folder>   (named after the folder)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

S = 4  # supersampling
W, H = 208, 112  # frame size
COLS = 8
K = 0.93  # fish size in the frame
CX0, CY0 = 104.0, 56.0
CLIPS = {"swim": (43, 24, True), "idle": (45, 12, True), "turn": (26, 24, False)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HEAD_X, TAIL_X = (200 - CX0) * K + CX0, (4 - CX0) * K + CX0


def tx(pts):
    """Drawing coordinates -> pixels of the supersampled canvas."""
    return [(((x - CX0) * K + CX0) * S, ((y - CY0) * K + CY0) * S) for x, y in pts]


def bezier(p0, p1, p2, p3, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        out.append(
            tuple(a * p0[j] + b * p1[j] + c * p2[j] + d * p3[j] for j in range(2))
        )
    return out


def rotate(pts, deg, pivot):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [
        (
            pivot[0] + (x - pivot[0]) * c - (y - pivot[1]) * s,
            pivot[1] + (x - pivot[0]) * s + (y - pivot[1]) * c,
        )
        for x, y in pts
    ]


TAIL = (
    bezier((52, 52), (36, 44), (20, 30), (4, 14))
    + bezier((4, 14), (12, 40), (12, 72), (4, 98))
    + bezier((4, 98), (20, 82), (36, 68), (52, 60))
)
DORSAL = bezier((74, 28), (84, 12), (104, 4), (128, 8)) + bezier(
    (128, 8), (138, 14), (146, 24), (150, 32)
)
ANAL = bezier((70, 82), (64, 94), (58, 102), (50, 106)) + bezier(
    (50, 106), (66, 106), (80, 98), (92, 88)
)
VENTRAL = bezier((112, 84), (116, 96), (124, 106), (136, 108)) + bezier(
    (136, 108), (140, 100), (142, 92), (142, 84)
)
BODY = (
    bezier((48, 56), (58, 28), (92, 20), (126, 22))
    + bezier((126, 22), (158, 24), (184, 38), (200, 54))
    + bezier((200, 54), (186, 74), (158, 90), (124, 92))
    + bezier((124, 92), (92, 94), (58, 86), (48, 56))
)
PECT = bezier((134, 60), (146, 54), (156, 62), (158, 74)) + bezier(
    (158, 74), (148, 80), (136, 76), (134, 60)
)
PECT_PIVOT = (136, 66)

SIZE = (W * S, H * S)


def gradient(top, bottom):
    t = np.linspace(0, 1, SIZE[1])[:, None, None]
    col = np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t
    arr = np.repeat(col, SIZE[0], axis=1).astype(np.uint8)
    alpha = np.full((SIZE[1], SIZE[0], 1), 255, np.uint8)
    return Image.fromarray(np.concatenate([arr, alpha], axis=2), "RGBA")


def paint(img, pts, top, bottom, edge):
    mask = Image.new("L", SIZE, 0)
    ImageDraw.Draw(mask).polygon(tx(pts), fill=255)
    ring = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask)
    img.paste(Image.new("RGBA", SIZE, edge + (255,)), (0, 0), ring)
    img.paste(gradient(top, bottom), (0, 0), mask)
    return mask


def draw_base(pect_deg):
    """The fish without its question mark (supersampled)."""
    img = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    fin = ((46, 46, 54), (16, 16, 20), (6, 6, 8))
    for shape in (TAIL, DORSAL, ANAL, VENTRAL):
        paint(img, shape, *fin)
    body = paint(img, BODY, (52, 52, 60), (10, 10, 12), (4, 4, 6))
    hl = Image.new("L", SIZE, 0)
    ImageDraw.Draw(hl).ellipse([c * S for c in (70, 24, 150, 34)], fill=70)
    hl = hl.filter(ImageFilter.GaussianBlur(3 * S))
    img.paste(
        Image.new("RGBA", SIZE, (120, 120, 134, 255)),
        (0, 0),
        ImageChops.multiply(hl, body),
    )
    paint(
        img, rotate(PECT, pect_deg, PECT_PIVOT), (62, 62, 70), (22, 22, 26), (6, 6, 8)
    )
    d = ImageDraw.Draw(img)
    ex, ey, er = 176, 49, 6.5
    (x0, y0), (x1, y1) = tx([(ex - er, ey - er), (ex + er, ey + er)])
    d.ellipse([x0, y0, x1, y1], fill=(236, 236, 236, 255))
    (x0, y0), (x1, y1) = tx([(ex - 2.8, ey - 3.6), (ex + 4.4, ey + 3.6)])
    d.ellipse([x0, y0, x1, y1], fill=(6, 6, 8, 255))
    (x0, y0), (x1, y1) = tx([(ex + 0.2, ey - 2.6), (ex + 2.0, ey - 0.8)])
    d.ellipse([x0, y0, x1, y1], fill=(255, 255, 255, 255))
    return img


def make_glyph():
    font = ImageFont.truetype(FONT, round(64 * K * S))
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (SIZE[0] / 2, SIZE[1] / 2),
        "?",
        font=font,
        fill=(250, 250, 250, 255),
        anchor="mm",
    )
    box = layer.getbbox()
    glyph = layer.crop(box)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # centre of the mark on the body, in canvas pixels
    return (
        glyph,
        (CX0 + 4 * K) * S + (cx - SIZE[0] / 2),
        (CY0 + 2 * K) * S + (cy - SIZE[1] / 2),
    )


def wave(x, phase, amp):
    """Vertical shift (1x pixels) at column x: a wave, larger at the tail."""
    w = min(1.0, max(0.0, (HEAD_X - x) / (HEAD_X - TAIL_X)))
    return amp * (0.06 + 0.94 * w**1.7) * math.sin(phase - 1.6 * w)


def warp(img, phase, amp):
    arr = np.asarray(img)
    out = np.zeros_like(arr)
    h = arr.shape[0]
    for xs in range(arr.shape[1]):
        d = round(wave(xs / S, phase, amp) * S)
        if d == 0:
            out[:, xs] = arr[:, xs]
        elif d > 0:
            out[d:, xs] = arr[: h - d, xs]
        else:
            out[: h + d, xs] = arr[-d:, xs]
    return Image.fromarray(out, "RGBA")


GLYPH, GX, GY = make_glyph()


def frame(phase, amp, pect_deg, theta=0.0):
    """One frame: the fish at a tail phase, turned by `theta` (0..pi)."""
    fish = warp(draw_base(pect_deg), phase, amp)
    cos = math.cos(theta)
    wf = math.hypot(cos, 0.2 * math.sin(theta))
    new_w = max(2, round(SIZE[0] * wf))
    layer = fish.resize((new_w, SIZE[1]), Image.LANCZOS)
    if cos < 0:
        layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    canvas.alpha_composite(layer, ((SIZE[0] - new_w) // 2, 0))
    # the mark is painted on both flanks: it is never mirrored
    if abs(cos) > 0.12:
        gw = max(2, round(GLYPH.width * abs(cos)))
        glyph = GLYPH.resize((gw, GLYPH.height), Image.LANCZOS)
        sign = 1 if cos >= 0 else -1
        gx = SIZE[0] / 2 + sign * (GX - SIZE[0] / 2) * wf
        gy = GY + wave(GX / S, phase, amp) * S
        canvas.alpha_composite(
            glyph, (round(gx - gw / 2), round(gy - glyph.height / 2))
        )
    return canvas.convert("RGBa").resize((W, H), Image.LANCZOS).convert("RGBA")


def smooth(u):
    return u * u * (3 - 2 * u)


def build():
    frames, _names = [], []
    n = CLIPS["swim"][0]
    for i in range(n):
        p = 2 * math.pi * 2 * i / n
        frames.append(frame(p, 7.0, 7 * math.sin(2 * math.pi * 2 * i / n)))
    n = CLIPS["idle"][0]
    for i in range(n):
        p = 2 * math.pi * i / n
        frames.append(frame(p, 2.2, 9 * math.sin(2 * math.pi * 5 * i / n)))
    n = CLIPS["turn"][0]
    for i in range(n):
        u = i / (n - 1)
        frames.append(
            frame(
                2 * math.pi * 1.5 * u,
                5.0 * math.sin(math.pi * u),
                6 * math.sin(2 * math.pi * 2 * u),
                math.pi * smooth(u),
            )
        )
    return frames


def thumbnail(fr, size=(96, 64)):
    box = fr.getchannel("A").getbbox()
    sub = fr.crop(box)
    k = min(size[0] / sub.width, size[1] / sub.height)
    small = sub.convert("RGBa").resize(
        (max(1, round(sub.width * k)), max(1, round(sub.height * k))), Image.LANCZOS
    )
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(
        small.convert("RGBA"),
        ((size[0] - small.width) // 2, (size[1] - small.height) // 2),
    )
    return out


def main(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    frames = build()
    rows = math.ceil(len(frames) / COLS)
    sheet = Image.new("RGBA", (COLS * W, rows * H), (0, 0, 0, 0))
    for i, fr in enumerate(frames):
        sheet.paste(fr, ((i % COLS) * W, (i // COLS) * H))
    entry = folder.name
    sheet.save(
        folder / f"{entry}.webp", format="WEBP", quality=80, alpha_quality=90, method=6
    )
    thumbnail(frames[0]).save(
        folder / "thumb.webp", format="WEBP", quality=85, method=6
    )
    swim = frames[: CLIPS["swim"][0]]
    widths = [b[2] - b[0] for b in (f.getchannel("A").getbbox() for f in swim)]
    length = float(np.median(widths))
    start, clips = 0, {}
    for name, (count, fps, loop) in CLIPS.items():
        clips[name] = {"from": start, "to": start + count - 1, "fps": fps, "loop": loop}
        start += count
    # light descriptor: only the picture (the species brings its own names and
    # behaviour)
    species = {
        "kind": "fish",
        "frame": [W, H],
        "columns": COLS,
        "clips": clips,
        "atlas": {"1x": f"{entry}.webp"},
        "facing": "right",
        "length_frac": round(length / W, 4),
        "thumbnail": "thumb.webp",
    }
    (folder / "species.json").write_text(
        json.dumps(species, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
