"""
realpage.py — pixel-faithful clone of the REAL captcha page
(measured from uploads/immagine_2026-10-09_211228957.png).

render_page(dual=False, seed=None) -> PIL.Image 1026x425, plus the captcha
rects and texts via render_page(..., meta=True) -> (img, meta) with
meta = {"boxes": [(x,y,w,h)], "texts": [str], "textbox": (x,y,w,h),
        "buttons": [(x,y,w,h)], "submit": (x,y,w,h)}

Layout constants copied from the screenshot:
  outer bg 241, panels 243 / white 255, grey row 237,
  red textbox ring ~2px at (210,196,785,45),
  grey buttons (230,230,230), yellow SUBMIT AND PAUSE (231,231,169),
  captcha card centred in row 1 (~y 20), native ~172x70 (size varies/round).
"""
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 1026, 425
_F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fc = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(_FB if bold else _F, size)
    return _fc[key]


def render_page(dual=False, seed=None, meta=False, scale_range=(0.8, 1.5)):
    import generator
    rng = random.Random(seed)

    page = Image.new("RGB", (W, H), (241, 241, 241))
    d = ImageDraw.Draw(page)

    # --- panel rows (y ranges from the screenshot) -------------------------
    d.rectangle([6, 8, W - 7, 110], fill=(243, 243, 243), outline=(225, 225, 225))
    d.rectangle([6, 114, W - 7, 166], fill=(237, 237, 237), outline=(225, 225, 225))
    d.rectangle([6, 170, 200, 262], fill=(243, 243, 243), outline=(225, 225, 225))
    d.rectangle([204, 170, W - 7, 262], fill=(255, 255, 255), outline=(225, 225, 225))
    d.rectangle([6, 266, W - 7, 348], fill=(255, 255, 255), outline=(225, 225, 225))
    d.rectangle([6, 352, W - 7, 417], fill=(243, 243, 243), outline=(225, 225, 225))

    # --- row 2: the 14-second notice ---------------------------------------
    d.text((20, 128), "You have to solve the image in ", font=font(17),
           fill=(70, 70, 70))
    w1 = d.textlength("You have to solve the image in ", font=font(17))
    d.text((20 + w1, 128), "14", font=font(17, True), fill=(40, 40, 40))
    w2 = d.textlength("14", font=font(17, True))
    d.text((20 + w1 + w2, 128), " seconds", font=font(17), fill=(70, 70, 70))

    # --- row 3: labels + red textbox ---------------------------------------
    d.text((20, 196), "Text:", font=font(15), fill=(60, 60, 60))
    d.text((20, 218), "CASE SENSITIVE", font=font(13, True), fill=(200, 30, 30))
    TB = (210, 196, 785, 45)
    d.rectangle([TB[0], TB[1], TB[0] + TB[2], TB[1] + TB[3]],
                fill=(255, 255, 255), outline=(200, 30, 30), width=2)

    # --- row 4: buttons -----------------------------------------------------
    B1 = (218, 285, 134, 40)
    B2 = (360, 285, 112, 40)
    B3 = (845, 285, 128, 40)
    for b, label, fill in ((B1, "SUBMIT ANSWER", (230, 230, 230)),
                           (B2, "DON'T KNOW", (230, 230, 230)),
                           (B3, "SUBMIT AND PAUSE", (231, 231, 169))):
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], fill=fill,
                    outline=(150, 150, 150), width=1)
        tw = d.textlength(label, font=font(12, True))
        d.text((b[0] + (b[2] - tw) / 2, b[1] + 12), label, font=font(12, True),
               fill=(60, 60, 60))

    # --- row 5: dual mode ---------------------------------------------------
    d.text((20, 372), "Dual Mode:", font=font(15), fill=(60, 60, 60))
    d.rectangle([218, 372, 231, 385], outline=(90, 90, 90), width=1,
                fill=(255, 255, 255))
    if dual:
        d.line([220, 378, 224, 383], fill=(30, 30, 30), width=2)
        d.line([224, 383, 229, 374], fill=(30, 30, 30), width=2)
    d.text((238, 372), "(get up to 2 images at once)", font=font(15),
           fill=(90, 90, 90))

    # --- the captcha card(s) in row 1 --------------------------------------
    boxes, texts = [], []
    n = 2 if dual else 1
    cards = []
    for i in range(n):
        cap, truth = generator.generate()
        s = rng.uniform(*scale_range)
        cap = cap.resize((max(40, int(cap.width * s)),
                          max(24, int(cap.height * s))), Image.LANCZOS)
        cards.append((cap, truth))
    gap = rng.randint(14, 60) if n == 2 else 0
    total = sum(c.width for c, _ in cards) + gap
    x = (W - total) // 2 + rng.randint(-40, 40)
    for i, (cap, truth) in enumerate(cards):
        y = rng.randint(16, 100 - min(70, cap.height)) + 6
        page.paste(cap, (x, y))
        boxes.append((x, y, cap.width, cap.height))
        texts.append(truth)
        x += cap.width + gap

    if not meta:
        return page
    return page, {"boxes": boxes, "texts": texts, "textbox": TB,
                  "buttons": [B1, B2], "submit": B3}
