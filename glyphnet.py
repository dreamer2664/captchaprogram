"""
Per-glyph CNN classifier — the trained "reads" model (cross-entropy).

Input: 2 channels x 56 x 56
  ch0: baseline-aligned ABSOLUTE contrast patch   (case/size cues)
  ch1: square-normalized glyph shape              (identity cues)

Train:  python3 glyphnet.py train [epochs] [samples]
"""
import shutil
import os
import string
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    torch.set_num_threads(2)      # small batches: fewer threads = lower latency
except Exception:
    pass

from crnn import clean_mask, contrast_field
from generator import generate

ALPHABET = string.ascii_letters + string.digits
P = 56
WID = 84          # wide patch: neighbours visible => relative case cues
UP, DOWN, SIDE = 40, 16, 42


def _shape_channel(mask_win: np.ndarray) -> np.ndarray:
    """largest component of window, square-normalized to 28, centred"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_win.astype(np.uint8), 8)
    canv = np.zeros((P, WID), np.float32)
    if n <= 1:
        return canv
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    m = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    q = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_LINEAR)
    canv[14:42, (WID-28)//2:(WID-28)//2+28] = q.astype(np.float32) / 255.0
    return canv


def patch(dom: np.ndarray, xc: float, base: float) -> np.ndarray:
    """2-channel 56x56 patch around (xc, baseline)."""
    H, W = dom.shape
    x0, y0 = int(round(xc)) - SIDE, int(round(base)) - UP
    ch0 = np.zeros((P, WID), np.float32)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + WID), min(H, y0 + P)
    ch0[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = dom[ys0:ys1, xs0:xs1]
    wx0, wy0 = int(round(xc)) - 16, int(round(base)) - UP
    win = np.zeros((54, 32), np.float32)
    wxs0, wys0 = max(0, wx0), max(0, wy0)
    wxs1, wys1 = min(W, wx0 + 32), min(H, wy0 + 54)
    win[wys0 - wy0:wys1 - wy0, wxs0 - wx0:wxs1 - wx0] = dom[wys0:wys1, wxs0:wxs1]
    m = (win > 0.78).astype(np.uint8)  # lines cap at dom 0.75: never pass
    ch1 = _shape_channel(m)
    return np.stack([ch0, ch1])


class GlyphNet(nn.Module):
    def __init__(self, n: int = len(ALPHABET)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 10, 192), nn.ReLU(True), nn.Dropout(0.25),
            nn.Linear(192, n),
        )

    def forward(self, x):
        return self.net(x)


_model = None
_HERE = os.path.dirname(os.path.abspath(__file__))


def get_model(path: str = "glyphnet.pt"):
    """Load the trained weights.  The default path resolves NEXT TO THIS
    MODULE, not next to the current working directory: launching the runner
    from a different folder (or via a shortcut) must not silently produce a
    model-less process that answers every captcha with an empty string."""
    global _model
    if _model is not None:
        return _model
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_HERE, path)
        if os.path.exists(candidate):
            path = candidate
    try:
        m = GlyphNet()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        _model = m
        return m
    except Exception as e:
        print(f"[glyphnet] WARNING: could not load weights from {path!r}: {e}. "
              f"Local reads will be EMPTY; put your Gemini keys in "
              f"gemini_keys.txt so the second opinion carries the rounds.",
              flush=True)
        return None


def classify(dom: np.ndarray, xc: float, base: float):
    """-> (char, confidence)"""
    m = get_model()
    if m is None:
        return "", 0.0
    x = torch.from_numpy(patch(dom, xc, base)[None]).float()
    with torch.no_grad():
        p = m(x).softmax(-1)[0]
    i = int(p.argmax())
    return ALPHABET[i], float(p[i])


def classify_many(dom: np.ndarray, xcs, base: float):
    """ONE batched forward for all glyphs -> [(char, confidence), ...]"""
    m = get_model()
    if m is None or len(xcs) == 0:
        return [("", 0.0)] * len(xcs)
    x = torch.from_numpy(np.stack([patch(dom, xc, base) for xc in xcs])).float()
    with torch.no_grad():
        p = m(x).softmax(-1)
    idx = p.argmax(1)
    conf = p.gather(1, idx[:, None])[:, 0]
    return [(ALPHABET[int(i)], float(c)) for i, c in zip(idx, conf)]


def train(epochs: int = 8, samples: int = 15000, batch: int = 64) -> None:
    import os
    model = GlyphNet()
    # SAFETY FLOOR: a bad epoch-1 initialization must never overwrite good
    # weights.  Read the current model's accuracy first and only save when the
    # new one actually beats it (this burned three retrains already).
    floor = 0.0
    if os.path.exists("glyphnet.pt"):
        try:
            prev = GlyphNet()
            prev.load_state_dict(torch.load("glyphnet.pt", map_location="cpu",
                                            weights_only=True))
            prev.eval()
            shutil.copy("glyphnet.pt", "glyphnet_prev.pt")
            floor = -1.0            # marker: floor measured after dataset load
            del prev
        except Exception:
            floor = 0.0
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    crit = nn.CrossEntropyLoss()
    cache = "glyph_train_wide.npy"
    if not os.path.exists(cache):
        Xu = np.zeros((samples, 2, P, WID), np.uint8)
        Yu = np.zeros(samples, np.int64)
        i = 0
        print("generating dataset…", flush=True)
        rng = np.random.default_rng(0)
        while i < samples:
            img, truth, boxes = generate(boxes=True)
            dom = contrast_field(img)
            for ch, (xc, base, _sz) in zip(truth, boxes):
                # jitter = inference reality: seg centroids/baselines are off
                # by ~N(0,1)px in x and ~N(-1.4,4.2)px in baseline
                xc_j = xc + rng.uniform(-4.0, 4.0)
                base_j = base + rng.uniform(-6.0, 6.0)
                Xu[i] = np.clip(patch(dom, xc_j, base_j) * 255, 0, 255).astype(np.uint8)
                Yu[i] = ALPHABET.index(ch)
                i += 1
        np.save(cache, Xu)
        np.save("glyph_train_wide_y.npy", Yu)
        del Xu
    M = np.load(cache, mmap_mode="r")
    Y = torch.from_numpy(np.load("glyph_train_wide_y.npy"))
    n = M.shape[0]
    print(f"dataset: {n} patches (memmap)", flush=True)

    if floor < 0:                     # measure the weights we are protecting
        try:
            prev = GlyphNet()
            prev.load_state_dict(torch.load("glyphnet.pt", map_location="cpu",
                                            weights_only=True))
            prev.eval()
            c = 0
            with torch.no_grad():
                for i in range(0, min(n, 3000), 500):
                    idx = torch.arange(i, min(i + 500, n))
                    c += int((prev(get(idx)).argmax(-1) == Y[idx]).sum())
            floor = c / min(n, 3000)
            print(f"safety floor: existing weights score {floor:.2%} on this "
                  f"dataset — new weights must beat it", flush=True)
            del prev
        except Exception:
            floor = 0.0

    def get(idx):
        return torch.from_numpy(np.array(M[idx], dtype=np.float32) / 255.0)

    best = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = crit(model(get(idx)), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n, 500):
                correct += int((model(get(torch.arange(i, min(i + 500, n)))).argmax(-1)
                                == Y[i:i + 500]).sum())
        acc = correct / n
        print(f"epoch {ep+1}/{epochs} loss {tot/n:.4f} train-acc {acc:.2%} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            if acc > floor:
                torch.save(model.state_dict(), "glyphnet.pt")
                print(f"  saved ({acc:.2%} > floor {floor:.2%})", flush=True)
            else:
                print(f"  NOT saved: {acc:.2%} <= floor {floor:.2%} "
                      f"(existing weights kept)", flush=True)
    Xv, Yv = [], []
    rng = np.random.default_rng(123)
    for _ in range(400):
        img, truth, boxes = generate(boxes=True)
        dom = contrast_field(img)
        for ch, (xc, base, _sz) in zip(truth, boxes):
            Xv.append(patch(dom, xc + rng.uniform(-4.0, 4.0),
                            base + rng.uniform(-6.0, 6.0)))   # inference-like
            Yv.append(ALPHABET.index(ch))
    Xv = torch.from_numpy(np.stack(Xv)).float(); Yv = torch.tensor(Yv)
    model.eval()
    cv = 0
    with torch.no_grad():
        for i in range(0, Xv.shape[0], 500):
            cv += int((model(Xv[i:i+500]).argmax(-1) == Yv[i:i+500]).sum())
    print(f"held-out glyph accuracy: {cv / Xv.shape[0]:.2%}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
              int(sys.argv[3]) if len(sys.argv) > 3 else 15000)
    else:
        print("model loaded:", get_model() is not None)
