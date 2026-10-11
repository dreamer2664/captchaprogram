"""
Gemini-vision reader with KEY ROTATION — the second opinion for hard rounds.

Keys are loaded from (in order):
  1. env var GEMINI_API_KEYS   (comma or whitespace separated)
  2. file gemini_keys.txt next to this module (one key per line, # comments)

Model: env var GEMINI_MODEL, default gemini-2.5-flash.
Never paste keys into chat — put them in gemini_keys.txt or the env var.
"""
import base64
import io
import os
import re
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TIMEOUT = 8

_lock = threading.Lock()
_keys: list[str] = []
_idx = 0
_loaded = False

PROMPT = ("This is a captcha image containing letters and digits. "
          "Respond with ONLY the exact characters you see, case sensitive, "
          "no spaces, no explanation.")


def load_keys() -> list[str]:
    global _keys, _loaded
    with _lock:
        if _loaded:
            return _keys
        raw = os.environ.get("GEMINI_API_KEYS", "")
        keys = re.split(r"[,\s]+", raw.strip()) if raw.strip() else []
        if not keys:
            path = os.path.join(HERE, "gemini_keys.txt")
            if os.path.exists(path):
                for line in open(path):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        keys.append(line)
        _keys = [k for k in keys if k]
        _loaded = True
        return _keys


def has_keys() -> bool:
    return bool(load_keys())


def _next_key() -> str | None:
    global _idx
    ks = load_keys()
    if not ks:
        return None
    with _lock:
        k = ks[_idx % len(ks)]
        _idx += 1
        return k


def _ask_once(url, payload, key, timeout):
    r = requests.post(url, json=payload, timeout=(min(3.0, timeout), timeout),
                      headers={"x-goog-api-key": key})
    if r.status_code != 200:
        return ""
    parts = r.json()["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts)
    return re.sub(r"[^A-Za-z0-9]", "", text)


def read_gemini(img, expected_len: int | None = None, model: str | None = None,
                timeout: float = TIMEOUT, deadline: float | None = None,
                max_attempts: int = 4) -> str:
    """OCR via Gemini vision; rotates keys on failure; '' if unusable.

    deadline = absolute time.time() by which we MUST be done: each request's
    timeout shrinks to fit the remaining budget, so the 15s round law holds.

    With expected_len set this asks up to max_attempts keys and returns the
    MAJORITY answer among length-correct reads (ties broken by arrival).  One
    call is enough to know the shape is right, but captchas routinely produce
    a single wrong character, so 2-3 agreeing reads are worth far more than
    the first non-empty string.  Returns early as soon as two attempts agree.
    """
    ks = load_keys()
    if not ks:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    model = model or DEFAULT_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32},
    }
    votes: dict[str, int] = {}
    best_wrong = ""
    n_att = min(max_attempts, len(ks))
    for _attempt in range(n_att):
        if deadline is not None:
            left = deadline - time.time()
            if left < 0.8:
                break
            # never let one request consume the whole remaining budget: each of
            # the attempts still to come gets an even slice, so a slow network
            # cannot blow the round's deadline on call one
            remaining_att = max(1, n_att - _attempt)
            timeout = min(timeout, max(0.8, left / remaining_att))
        try:
            clean = _ask_once(url, payload, _next_key(), timeout)
        except Exception:
            continue
        if not clean:
            continue
        if expected_len and len(clean) != expected_len:
            best_wrong = best_wrong or clean       # keep as last resort
            continue
        votes[clean] = votes.get(clean, 0) + 1
        if votes[clean] >= 2:
            return clean                            # two keys agree: done early
    if votes:
        return max(votes.items(), key=lambda kv: kv[1])[0]
    return best_wrong
