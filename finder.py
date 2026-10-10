"""
Auto-detection inside a BIG calibrated region.

The real page changes the captcha image size every round and sometimes shows
TWO captchas side by side (answer = "text1 text2"), so a tight fixed box is
useless.  The user calibrates one generous region containing the image area
AND the answer text box; this module finds everything per round:

  detect_all(region_img) -> (boxes, textbox_center)
      boxes: [(x,y,w,h), ...] 1 or 2 captcha image rects, left-to-right
      textbox_center: (cx, cy) of the answer input, or None

Robust to the two regimes real pages produce:
  * card bg contrasts with page bg -> the whole card is one solid blob
  * card bg ~ page bg             -> only glyph ink shows; a bigger
    morphological close bridges the glyph gaps into one cluster blob
Text lines (header/labels), timer bars and buttons are rejected by
aspect/fill/interior-ink rules; the text box (widest UNIFORM rectangle in
the lower region) is found first and anchors the captcha search above it.
"""
import hashlib

import cv2
import numpy as np


def _bg_median(a: np.ndarray) -> np.ndarray:
    flat = a.reshape(-1, 3)
    if len(flat) > 200_000:
        flat = flat[:: len(flat) // 200_000]
    return np.median(flat, axis=0)


def _components(mask, raw, min_area, h_min=22, w_min=55):
    """candidates from a closed mask; rules use BOTH the closed fill and the
    RAW ink fill (morphological closing fattens sparse text rows and would
    otherwise let header/label lines masquerade as captcha cards)."""
    n, lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    cands = []
    for i in range(1, n):
        x, y, w, h, ar = (int(stats[i, 0]), int(stats[i, 1]),
                          int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4]))
        if ar < min_area or w < w_min or h < h_min:
            continue
        if w >= 0.97 * W and h >= 0.97 * H:
            continue                          # whole region lit: bad page bg
        fill = ar / float(w * h)
        raw_fill = float(raw[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].sum()) / (w * h)
        asp = w / max(h, 1)
        if asp > 4.5 and raw_fill < 0.55 and h < 55:
            continue                          # text line (possibly + thin bar)
        if asp > 6.0 and raw_fill < 0.8:
            continue                          # wide sparse row: labels, headers
        if fill > 0.9 and asp > 6.0:
            continue                          # solid strip: timer/progress bar
        score = ar * (1.4 if fill > 0.75 else 1.0)
        if fill < 0.95 and asp > 4.2:
            score *= 0.35                     # button-like: solid-ish + elongated
        if 1.9 <= asp <= 3.5:
            score *= 1.3                      # captcha-card aspect
        cands.append({"box": (x, y, w, h), "score": score, "fill": fill})
    return cands


def _iou_box(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def _merge_near(cands, gap=8):
    boxes = [list(c["box"]) for c in cands]
    scores = [c["score"] for c in cands]
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(boxes)), key=lambda i: -scores[i])
        boxes = [boxes[i] for i in order]
        scores = [scores[i] for i in order]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if _iou_box(a, b) > 0 or (
                        b[0] - gap <= a[0] + a[2] and a[0] - gap <= b[0] + b[2]
                        and b[1] - gap <= a[1] + a[3] and a[1] - gap <= b[1] + b[3]
                        and min(a[3], b[3]) / max(a[3], b[3]) > 0.6):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    scores[i] = max(scores[i], scores[j])
                    boxes.pop(j)
                    scores.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def _split_dual(box, mask):
    """a box with two-captcha aspect may be two cards merged by the close:
    split at the emptiest column in the middle band"""
    x, y, w, h = box
    if w / max(h, 1) < 3.6:
        return [box]
    col = mask[y:y + h, x:x + w].sum(axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col[None], (15, 1), 0)[0]
    lo, hi = int(w * 0.30), int(w * 0.70)
    cut = lo + int(np.argmin(col[lo:hi]))
    if col[cut] > 0.25 * h:
        return [box]                          # no clear gap: single wide card
    out = []
    for xx, ww in ((x, cut), (x + cut, w - cut)):
        sub = mask[y:y + h, xx:xx + ww]
        cols = np.where(sub.sum(axis=0) > 0)[0]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(cols) < 30 or len(rows) < 12:
            continue
        pw, ph = int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)
        if ph < 30 or pw > 4.5 * ph:
            continue                    # sliver / text row, not a card
        out.append((xx + int(cols[0]), y + int(rows[0]), pw, ph))
    if len(out) == 2:
        h_sim = min(out[0][3], out[1][3]) / max(out[0][3], out[1][3])
        if h_sim < 0.5:
            return [box]                # not two side-by-side cards
    return out if len(out) == 2 else [box]


def find_captcha_boxes(img, min_area=1600, thresh=24, max_boxes=2):
    """-> [(x,y,w,h), ...] up to max_boxes captcha rects, left-to-right.

    Two detection regimes are unioned (dual mode can mix one solid card with
    one pale card, so BOTH always run):
      A  solid card bg   -> small close; the whole card is one blob
      B  card bg ~ page  -> bigger close bridges glyph gaps into a cluster
    Dedup uses true IoU.  Candidates are ranked by score, the top one seeds,
    and a second is accepted only if it is side-by-side with the seed.
    """
    a = np.array(img.convert("RGB")).astype(np.float32)
    H, W = a.shape[:2]
    page_bg = _bg_median(a)
    diff = np.abs(a - page_bg).max(axis=2)
    # thresh 24: pale captcha cards (e.g. (214,228,240) on a (243,243,243)
    # panel) differ by ~29 and become SOLID blobs; every page-chrome colour
    # pair on the real layout stays under 24 (panels 237/243/255, borders 225)
    mask = (diff > thresh).astype(np.uint8)

    # PRE-ERASE text rows ("You have to solve...", labels): wide, sparse,
    # short bands are page furniture, and leaving them in lets the close
    # operation fuse them into captcha cards a few pixels away.
    mA0 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(mA0, 8)
    for i in range(1, n0):
        x, y, w, h = st0[i, cv2.CC_STAT_LEFT], st0[i, cv2.CC_STAT_TOP], \
                     st0[i, cv2.CC_STAT_WIDTH], st0[i, cv2.CC_STAT_HEIGHT]
        sub = mask[y:y + h, x:x + w]
        fill = float(sub.mean()) if sub.size else 0.0
        if w > 4.0 * h and fill < 0.5 and h <= 35:
            # erase ONLY this component's pixels: a captcha card overlapping
            # the text row's bbox must keep its own ink
            comp = (lab0[y:y + h, x:x + w] == i)
            sub[comp] = 0

    maskA = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    candsA = [c for c in _components(maskA, mask, min_area) if c["fill"] > 0.75]
    # horizontal-only close: chains glyphs of one captcha into a single cluster
    # but NEVER bridges the vertical gap to a text row below/above it (that
    # fusion was mutilating cards on the real page layout)
    maskB = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    candsB = _components(maskB, mask, 150, h_min=20)

    cands = list(candsA)
    for c in candsB:
        if not any(_iou_box(c["box"], a2["box"]) > 0.45 for a2 in candsA):
            cands.append(c)

    # Page furniture (yellow SUBMIT AND PAUSE, red CASE SENSITIVE label, the
    # red ring itself) is solid enough to out-score a pale captcha card.  On
    # the real page the captcha always sits ABOVE the answer row, so once the
    # text box is known, anything at or below its top edge is furniture.
    tb = _red_textbox_rect(img)
    if tb is not None:
        limit = tb[1] - 6
        above = [c for c in cands if c["box"][1] + c["box"][3] <= limit + 40
                 and c["box"][1] < limit]
        cands = above or cands
    if not cands:
        return []

    # (a) wide sparse rows are text lines (the "14 seconds" notice), not cards
    def _is_textrow(c):
        x, y, w, h = c["box"]
        sub = mask[max(0, y):y + h, max(0, x):x + w]
        fill = float(sub.mean()) if sub.size else 0.0
        return w > 4.0 * h and fill < 0.5 and h < 55
    cands = [c for c in cands if not _is_textrow(c)] or cands

    # (b) the morphological close can bridge a card to the notice row a few px
    # below it; the RAW mask still has empty rows there — split on them
    split = []
    for c in cands:
        x, y, w, h = c["box"]
        sub = mask[max(0, y):y + h, max(0, x):x + w]
        rows = sub.sum(axis=1)
        bands, start = [], None
        for i, v in enumerate(rows):
            if v > 0 and start is None:
                start = i
            elif v == 0 and start is not None:
                if i - start >= 2:
                    bands.append((start, i))
                start = None
        if start is not None:
            bands.append((start, len(rows)))
        bands = [b for b in bands if b[1] - b[0] >= 12]
        if len(bands) > 1:
            for b0, b1 in bands:
                piece = dict(c)
                piece["box"] = (x, y + b0, w, b1 - b0)
                piece["score"] = c["score"] * (b1 - b0) / h
                split.append(piece)
        else:
            split.append(c)
    cands = [c for c in split if not _is_textrow(c)] or cands
    if not cands:
        return []

    # (c) pre-split: a cluster that swallowed TWO cards (horizontal bridge >=
    # the dual gap) is cut at a real empty gutter: >=8 consecutive near-zero
    # ink columns inside the middle band.  Word gaps inside one captcha are
    # <=6 columns, so a single card never gets cut.
    presplit = []
    for c in cands:
        x, y, w, h = c["box"]
        cols = mask[y:y + h, x:x + w].sum(axis=0).astype(np.float32)
        lo, hi = int(w * 0.25), int(w * 0.75)
        best_cut, best_run = None, 0
        run = 0
        for i in range(lo, hi):
            if cols[i] < 0.05 * h:
                run += 1
                if run > best_run:
                    best_run, best_cut = run, i - run // 2
            else:
                run = 0
        if best_run >= 8 and best_cut is not None:
            pieces = []
            for xx, ww in ((x, best_cut), (x + best_cut, w - best_cut)):
                sub = mask[y:y + h, xx:xx + ww]
                cs = np.where(sub.sum(axis=0) > 0)[0]
                rs = np.where(sub.sum(axis=1) > 0)[0]
                if len(cs) < 30 or len(rs) < 12:
                    continue
                pw, ph = int(cs[-1] - cs[0] + 1), int(rs[-1] - rs[0] + 1)
                if ph < 30 or pw > 4.5 * ph:
                    continue
                pieces.append(dict(box=(xx + int(cs[0]), y + int(rs[0]), pw, ph),
                                   score=c["score"] * 0.5, fill=c["fill"]))
            if len(pieces) == 2:
                h0, h1 = pieces[0]["box"][3], pieces[1]["box"][3]
                h_sim = min(h0, h1) / max(h0, h1)
                if h_sim > 0.5:
                    presplit.extend(pieces)
                    continue
        presplit.append(c)
    cands = presplit

    cands.sort(key=lambda c: -c["score"])
    seed = list(cands[0]["box"])
    out = [seed]
    for c in cands[1:]:
        b = list(c["box"])
        vov = min(seed[1] + seed[3], b[1] + b[3]) - max(seed[1], b[1])
        h_sim = min(seed[3], b[3]) / max(seed[3], b[3])
        side = (b[0] >= seed[0] + seed[2] - 8) or (seed[0] >= b[0] + b[2] - 8)
        if vov > 0.25 * min(seed[3], b[3]) and h_sim > 0.40 and side:
            out.append(b)
            if len(out) >= max_boxes:
                break

    # un-merge two cards the close fused into one blob
    final = []
    for b in out:
        final.extend(_split_dual(tuple(b), maskB))
    final = final[:max_boxes]
    final.sort(key=lambda b: b[0])

    # pad by regime, then re-bound on real ink: a solid card already carries
    # its own bg margin (padding it would pull page bg inside the crop, where
    # it reads as "ink" and fuses every glyph into full-height strips); an ink
    # cluster gets breathing room, then both are tightened to the ink they
    # actually contain (recovers partial pale-card clusters, trims slack).
    res = []
    for x, y, w, h in final:
        sub = mask[max(0, y):y + h, max(0, x):x + w]
        p = 0 if (sub.size and float(sub.mean()) >= 0.6) else 8
        x0, y0 = max(0, x - p), max(0, y - p)
        bx, by, bw, bh = x0, y0, min(W - x0, w + 2 * p), min(H - y0, h + 2 * p)
        wx0, wy0 = max(0, bx - 10), max(0, by - 10)
        wx1, wy1 = min(W, bx + bw + 10), min(H, by + bh + 10)
        win = mask[wy0:wy1, wx0:wx1]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(win, 8)
        xs0, ys0, xs1, ys1 = wx1, wy1, wx0, wy0
        found = False
        for i in range(1, n):
            cx, cy, cw, ch, area = stats[i]
            if area < 12:
                continue
            ccx, ccy = wx0 + cx + cw / 2, wy0 + cy + ch / 2
            if not (bx - 10 <= ccx <= bx + bw + 10 and by - 10 <= ccy <= by + bh + 10):
                continue
            xs0 = min(xs0, wx0 + cx); ys0 = min(ys0, wy0 + cy)
            xs1 = max(xs1, wx0 + cx + cw); ys1 = max(ys1, wy0 + cy + ch)
            found = True
        if found and (xs1 - xs0) * (ys1 - ys0) >= 0.5 * bw * bh:
            bx, by = max(0, xs0 - 2), max(0, ys0 - 2)
            bw, bh = min(W - bx, xs1 - xs0 + 4), min(H - by, ys1 - ys0 + 4)
        res.append((bx, by, bw, bh))
    return res


def _red_textbox_rect(img):
    """The real page draws the answer input with a saturated RED ring.
    Chroma-keying it is exact and immune to everything else on the page.
    -> (x, y, w, h) or None."""
    a = np.array(img.convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    red = ((r > 130) & (r - g > 60) & (r - b > 60)).astype(np.uint8)
    if red.sum() < 200:
        return None
    red = cv2.dilate(red, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = red.shape
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.20 * W or h < 16 or h > 90 or w / h < 2.2:
            continue
        ring = red[y:y + h, x:x + w]
        fill = float(ring.mean())
        if fill > 0.5:                       # solid red block = label text
            continue
        if best is None or w * h > best[0]:
            best = (w * h, x, y, w, h)
    return (best[1], best[2], best[3], best[4]) if best else None


def find_textbox(img, cap_boxes=None):
    """-> (cx, cy) center of the answer input, or None.  Tries the real page's
    red ring first; falls back to the widest UNIFORM rectangle in the lower
    part of the region (text lines and buttons fail the interior-ink check)."""
    rr = _red_textbox_rect(img)
    if rr is not None:
        x, y, w, h = rr
        return (x + w // 2, y + h // 2)
    g = np.array(img.convert("L")).astype(np.uint8)
    H, W = g.shape[:2]
    ymin = int(H * 0.35)
    if cap_boxes:
        ymin = max(ymin, max(b[1] + b[3] for b in cap_boxes) + 4)
    if ymin >= H - 16:
        ymin = int(H * 0.35)
    sub = g[ymin:H, :]
    edges = cv2.Canny(sub, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.20 * W or w > W - 2:
            continue
        if h < 16 or h > 90 or w / h < 2.2:
            continue
        inner = sub[y + 3:y + h - 3, x + 3:x + w - 3]
        if inner.size and float(np.std(inner)) > 32:
            continue                          # ink inside: label/button, not input
        score = w * h
        if best is None or score > best[0]:
            best = (score, x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    return (x + w // 2, ymin + y + h // 2)


def find_textbox_rect(img, cap_boxes=None):
    """-> (x, y, w, h) of the answer input in region coords, or None"""
    rr = _red_textbox_rect(img)
    if rr is not None:
        return rr
    g = np.array(img.convert("L")).astype(np.uint8)
    H, W = g.shape[:2]
    ymin = int(H * 0.35)
    if cap_boxes:
        ymin = max(ymin, max(b[1] + b[3] for b in cap_boxes) + 4)
    if ymin >= H - 16:
        ymin = int(H * 0.35)
    sub = g[ymin:H, :]
    edges = cv2.dilate(cv2.Canny(sub, 50, 150), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.20 * W or w > W - 2:
            continue
        if h < 16 or h > 90 or w / h < 2.2:
            continue
        inner = sub[y + 3:y + h - 3, x + 3:x + w - 3]
        if inner.size and float(np.std(inner)) > 32:
            continue
        if best is None or w * h > best[0]:
            best = (w * h, x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    return (x, ymin + y, w, h)


def detect_all(img):
    """the one call the runner uses: captcha boxes first (aspect/fill rules
    reject furniture rows), then the textbox strictly BELOW the cards —
    a pale solid card can otherwise masquerade as the input field."""
    boxes = find_captcha_boxes(img)
    tb = find_textbox(img, cap_boxes=boxes)
    return boxes, tb


# ------------------------------------------------- round / furniture helpers
def crop_hash(img, b):
    """content fingerprint of a candidate box (tiny thumbnail: immune to
    sub-pixel jitter, fast on big regions)"""
    crop = img.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))
    return hashlib.md5(np.array(crop.convert("L").resize((24, 12))).tobytes()).digest()


def round_key(hashes):
    h = hashlib.md5()
    for x in hashes:
        h.update(x)
    return h.digest()


def box_iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def dynamic_boxes(img, cands, prev=None):
    """STATIC-FURNITURE FILTER: page decoration (headers, labels, buttons)
    keeps identical content round after round; captchas change every round.
    prev = [(box, crop_hash), ...] from the previous poll.
    -> [(box, hash, is_static), ...]"""
    prev = prev or []
    out = []
    for b in cands:
        h = crop_hash(img, b)
        static = any(box_iou(b, pb) > 0.7 and h == ph for pb, ph in prev)
        out.append((b, h, static))
    return out
