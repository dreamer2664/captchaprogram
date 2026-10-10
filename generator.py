"""
Captcha generator v3 — calibrated on the REAL page screenshot
(uploads/immagine_2026-10-09_211228957.png, 2026-10-09).

What the real thing does (measured from the screenshot + user reports):
  * card ~172x70 native, but the IMAGE SIZE CHANGES every round
  * pale card backgrounds (khaki (236,224,167) in the shot) that change
  * glyph ink dark & saturated (navy/purple (82,64,197) in the shot),
    DIFFERENT COLOR PER CHARACTER, different FONT per character,
    different SIZE per character, rotated
  * noise: 1px scribble lines in random mid hues at partial alpha,
    plus scattered 1px speckle dots over the whole card
  * dual mode: up to two cards at once

generate()            -> (img, text)
generate(boxes=True)  -> (img, text, [(xc, baseline, em), ...])
"""
import math
import random
import string

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 172, 70
CHARS = string.ascii_letters + string.digits

# --- palettes measured / extrapolated from the real page -------------------
CARD_BGS = [
    (236, 224, 167), (243, 239, 158),          # the khakis from the screenshot
    (214, 228, 240), (240, 219, 222),          # pale blue / pink rounds
    (219, 235, 209), (248, 248, 246),          # pale green / near-white
    (228, 222, 240), (250, 229, 197),          # pale lavender / peach
    (222, 236, 236), (240, 240, 210),
]
INK_COLORS = [
    (40, 40, 150), (82, 64, 197), (30, 30, 30), (140, 30, 35),
    (25, 105, 45), (170, 80, 20), (110, 30, 140), (0, 105, 120),
    (60, 60, 60), (35, 90, 160),
]
NOISE_COLORS = [
    (60, 180, 90), (220, 110, 85), (110, 107, 157), (230, 180, 60),
    (90, 200, 200), (200, 90, 180), (150, 200, 120), (210, 140, 100),
]
_FONT_CANDIDATES = [
    # system families (paths differ per distro / sandbox generation)
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # matplotlib's bundled faces (always present wherever mpl is)
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneral.ttf",
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/STIXNonUni.ttf",
    "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerifDisplay.ttf",
    "C:/Windows/Fonts/times.ttf",      # user's Windows box, if training there
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/georgia.ttf",
]
import os as _os
FONT_FILES = [p for p in _FONT_CANDIDATES if _os.path.exists(p)] or \
             [_FONT_CANDIDATES[4]]
_font_cache: dict = {}


def _font(path: str, size: int):
    key = (path, size)
    f = _font_cache.get(key)
    if f is None:
        f = ImageFont.truetype(path, size)
        _font_cache[key] = f
    return f


LAST_BOXES: list = []


def generate(length: int | None = None, boxes: bool = False):
    """One captcha card, real-page style."""
    n = length or 6
    bg = random.choice(CARD_BGS)
    card = Image.new("RGBA", (WIDTH, HEIGHT), bg + (255,))
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))

    # per-character: font, size, color, rotation  (the hard-captcha axes)
    adv_total = 0
    specs = []
    for _ in range(n):
        size = random.randint(26, 44)
        specs.append(dict(
            ch=random.choice(CHARS),
            font=random.choice(FONT_FILES),
            size=size,
            color=random.choice(INK_COLORS),
            rot=random.uniform(-18, 18),
        ))
    # lay out with small random gaps, then center the run; shrink until it
    # fits with a margin like the real card (never clips at the edges)
    widths = []
    for s in specs:
        widths.append(_font(s["font"], s["size"]).getlength(s["ch"]))
    gaps = [random.randint(-2, 3) for _ in range(n)]
    for _shrink in range(6):
        run = sum(widths) + sum(gaps)
        if run <= WIDTH - 14:
            break
        for i, s in enumerate(specs):
            s["size"] = max(18, int(s["size"] * 0.88))
            widths[i] = _font(s["font"], s["size"]).getlength(s["ch"])
    run = sum(widths) + sum(gaps)
    slack = max(0.0, (WIDTH - 14 - run) / 2)
    x = 7 + slack / 2 + random.uniform(-min(4, slack / 2), min(4, slack / 2))

    char_boxes = []
    text = ""
    for i, s in enumerate(specs):
        f = _font(s["font"], s["size"])
        pad = s["size"] * 2
        glyph = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((pad // 2, pad // 2), s["ch"], font=f,
                                   fill=s["color"] + (255,), anchor="mm")
        glyph = glyph.rotate(s["rot"], resample=Image.BICUBIC, expand=True)
        yc = HEIGHT // 2 + random.randint(-6, 6)
        layer.alpha_composite(glyph, (int(x - glyph.width / 2),
                                      int(yc - glyph.height / 2)))
        asc, _desc = f.getmetrics()
        char_boxes.append((x + widths[i] / 2, yc - s["size"] / 2 + asc,
                           s["size"]))
        text += s["ch"]
        x += widths[i] + gaps[i]

    card = Image.alpha_composite(card, layer)

    # --- noise ON TOP, exactly like the shot: thin scribbles + speckle -----
    d = ImageDraw.Draw(card, "RGBA")
    for _ in range(random.randint(3, 5)):
        c = random.choice(NOISE_COLORS)
        a = random.randint(80, 150)
        x0, y0 = random.randint(-10, WIDTH // 2), random.randint(0, HEIGHT)
        x1, y1 = random.randint(WIDTH // 2, WIDTH + 10), random.randint(0, HEIGHT)
        d.line([(x0, y0), (x1, y1)], fill=c + (a,), width=1)
    for _ in range(random.randint(25, 55)):
        c = random.choice(NOISE_COLORS + INK_COLORS)
        a = random.randint(90, 200)
        px, py = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
        d.point((px, py), fill=c + (a,))

    img = card.convert("RGB")
    LAST_BOXES.clear()
    LAST_BOXES.extend(char_boxes)
    if boxes:
        return img, text, list(char_boxes)
    return img, text
