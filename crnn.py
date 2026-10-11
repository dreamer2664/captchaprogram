"""
Shared image frontend + dormant CRNN/CTC reader.

contrast_field(): COLOR-INVARIANT frontend — measures each pixel's signed
luminance distance from the page background, so any bg/text color combo works
(dark-on-light and light-on-dark alike).  Replaces the old blue-dominance
field that only worked for indigo-on-khaki.

The CRNN+CTC reader is kept as a dormant fallback (it never converged on this
2-core box); glyphnet.py is the trained reader that ships.
"""
import string

import cv2
import numpy as np
from PIL import Image

ALPHABET = string.ascii_letters + string.digits          # 62 chars
BLANK = len(ALPHABET)
IMG_H, IMG_W = 70, 170


def contrast_field(img) -> np.ndarray:
    """PIL RGB image -> float32 (H,W) in [0,1]; glyphs bright, bg/lines ~0."""
    a = np.array(img.convert("RGB")).astype(np.float32)
    bg = np.median(a.reshape(-1, 3), axis=0)
    lum = a.mean(axis=2)
    bg_lum = float(bg.mean())
    diff = np.abs(a - bg).max(axis=2)
    mask0 = (diff > 50).astype(np.uint8)
    opened = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if opened.sum() > 20:
        inside_lum = float(np.median(lum[opened > 0]))
    elif mask0.sum() > 20:
        inside_lum = float(np.median(lum[mask0 > 0]))
    else:
        inside_lum = bg_lum - 1
    sign = 1.0 if inside_lum < bg_lum else -1.0
    return np.clip(sign * (bg_lum - lum), 0, 80) / 80.0


dom_field = contrast_field        # legacy alias


def clean_mask(m: np.ndarray) -> np.ndarray:
    """remove 1 px noise lines / speckles, keep strokes >= 2 px:
    erode 2x2 kills single-pixel structures, dilate 3x3 restores strokes"""
    e = cv2.erode(m, np.ones((2, 2), np.uint8))
    return cv2.dilate(e, np.ones((3, 3), np.uint8))


# ---------------------------------------------------------------- dormant CTC
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """column-profile encoder + BiLSTM + CTC head (kept for experiments)"""

    def __init__(self, n_classes: int = BLANK + 1):
        super().__init__()
        self.col = nn.Conv2d(1, 96, kernel_size=(IMG_H, 5), stride=(1, 2),
                             padding=(0, 2))                 # B x 96 x 1 x 85
        self.seq = nn.Sequential(
            nn.Conv1d(96, 128, 5, stride=2, padding=2), nn.ReLU(True),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(128, 96, bidirectional=True, num_layers=1,
                           batch_first=True)
        self.out = nn.Linear(192, n_classes)

    def forward(self, x):
        f = torch.relu(self.col(x)).squeeze(2)
        f = self.seq(f)
        f = f.permute(0, 2, 1)
        f, _ = self.rnn(f)
        return self.out(f)


def encode(text: str) -> list[int]:
    return [ALPHABET.index(ch) for ch in text]


def greedy_decode(logits: torch.Tensor) -> str:
    out = []
    for row in logits.argmax(-1):
        prev = None
        for t in row:
            t = int(t)
            if t != BLANK and t != prev:
                out.append(ALPHABET[t])
            prev = t
    return "".join(out)


_model = None


def get_model(path: str = "crnn_best.pt"):
    global _model
    if _model is not None:
        return _model
    try:
        m = CRNN()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        _model = m
        return m
    except FileNotFoundError:
        return None


def read_crnn(img, expected_len: int | None = None) -> str:
    m = get_model()
    if m is None:
        return ""
    x = torch.from_numpy(contrast_field(img)[None, None]).float()
    with torch.no_grad():
        return greedy_decode(m(x))


def train(epochs: int = 6, per_epoch: int = 5000, batch: int = 64) -> None:
    from generator import generate
    model = CRNN()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for _b in range(per_epoch // batch):
            xs, ys, yl = [], [], []
            for _ in range(batch):
                img, truth = generate()
                xs.append(contrast_field(img))
                e = encode(truth)
                ys.extend(e); yl.append(len(e))
            X = torch.from_numpy(np.stack(xs)[:, None]).float()
            logits = model(X)
            T = logits.shape[1]
            lp = logits.log_softmax(-1).permute(1, 0, 2)
            loss = crit(lp, torch.tensor(ys), torch.full((batch,), T),
                        torch.tensor(yl))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        hits = n = 0
        with torch.no_grad():
            for _ in range(200):
                img, truth = generate()
                x = torch.from_numpy(contrast_field(img)[None, None]).float()
                hits += greedy_decode(model(x)) == truth; n += 1
        acc = hits / n
        print(f"epoch {ep+1}/{epochs} val exact {acc:.1%}", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "crnn_best.pt")
    print(f"done, best {best:.1%}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
