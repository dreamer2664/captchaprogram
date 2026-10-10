"""
LIVE PC RUNNER (Windows) — finds the captcha(s) in a BIG zone, reads, types.

    python local_runner.py calibrate     # one-time: big region + options
    python local_runner.py run           # the bot: watch -> find -> read -> type
    python local_runner.py once          # single grab + detect + read (no typing)

The page changes image size every round and sometimes shows TWO captchas
(dual mode) — so instead of framing the image, you calibrate ONE generous
region that always contains the image area AND the answer text box.  Each
round the runner:
  1. grabs the region (mss, ~10 ms)
  2. finder.find_captcha_boxes -> 1 or 2 captcha rects (any size)
  3. detects a NEW round by hashing the crop thumbnails (immune to timer
     bars/animations elsewhere in the region)
  4. reads each crop: GlyphNet locally (~0.1 s); Gemini only when confidence
     is low AND the 15 s budget survives a hard-deadline call
  5. answer = "text1" or "text1 text2" (SPACE between dual captchas)
  6. clicks the text box (auto-detected; calibrated point as fallback),
     types with human jitter, submits (Enter or click)

Mouse to a screen corner = pyautogui failsafe = abort.
`once` saves last_capture.png (raw) + last_debug.png (green = captcha boxes
with reads, blue = detected text box). Send both when something looks wrong.
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import mss
import pyautogui
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solver  # noqa: E402
import finder  # noqa: E402

CONFIG = os.path.join(HERE, "runner_config.json")
ROUND_SECONDS = 14.0          # page shows 15; stay conservative
POLL = 0.2
pyautogui.FAILSAFE = True
_MSS = getattr(mss, "MSS", None) or mss.mss   # mss>=10 renamed mss() -> MSS()


def _grab(sct, box):
    shot = sct.grab(box)
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")





def _load_cfg():
    cfg = json.load(open(CONFIG))
    if "region" not in cfg and "image_box" in cfg:
        cfg["region"] = cfg.pop("image_box")     # old tight-box config: reuse
    return cfg


def calibrate():
    print("ONE big region this time: frame the area that ALWAYS contains\n"
          "the captcha image(s) AND the answer text box, wherever they move.\n"
          "Hover the mouse, press ENTER in this console for each point.\n")
    cfg = {}
    input("Mouse to TOP-LEFT of the big region…")
    x1, y1 = pyautogui.position()
    input("Mouse to BOTTOM-RIGHT of the big region…")
    x2, y2 = pyautogui.position()
    cfg["region"] = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
    n = input("Characters PER captcha? [6]: ").strip()
    cfg["chars"] = int(n) if n.isdigit() and int(n) > 0 else 6
    cfg["textbox_fallback"] = None
    if input("Calibrate a FALLBACK text-box point (used only if auto-detect\n"
             "fails)? (y/N): ").strip().lower() == "y":
        input("Mouse to the CENTER of the answer text box…")
        cfg["textbox_fallback"] = list(pyautogui.position())
    cfg["submit"] = None
    cfg["submit_mode"] = "enter"
    if input("Calibrate the SUBMIT button and submit by CLICK? (y/N, "
             "Enter is faster): ").strip().lower() == "y":
        input("Mouse to the SUBMIT ANSWER button…")
        cfg["submit"] = list(pyautogui.position())
        cfg["submit_mode"] = "click"
    json.dump(cfg, open(CONFIG, "w"), indent=2)
    print(f"\nsaved {CONFIG}:")
    print(json.dumps(cfg, indent=2))


def _type(text):
    for ch in text:                      # write() handles capitals + space
        pyautogui.write(ch)
        time.sleep(random.uniform(0.03, 0.07))


def _detect(img, max_boxes=2):
    boxes = finder.find_captcha_boxes(img, max_boxes=max_boxes)
    tb = finder.find_textbox(img, cap_boxes=boxes)
    return boxes, tb





def _read_all(img, boxes, chars, t0, time_left_total):
    """-> (answer, parts) where parts = [(text, src), ...] per captcha"""
    parts = []
    for i, b in enumerate(boxes):
        crop = img.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))
        remaining = time_left_total - (time.time() - t0)
        type_est = 0.06 * (chars * len(boxes) + len(boxes) - 1) + 0.5
        text, src = solver.solve_budget(
            crop, chars, time_left=max(remaining, 1.0), type_time=type_est)[:2]
        parts.append((text, src))
    answer = " ".join(p[0] for p in parts if p[0])
    return answer, parts


def once():
    cfg = _load_cfg()
    chars = cfg.get("chars", 6)
    region = cfg["region"]
    print(f"region: {region['width']}x{region['height']} px, chars/captcha: {chars}")
    with _MSS() as sct:
        img = _grab(sct, region)
    img.save(os.path.join(HERE, "last_capture.png"))
    boxes, tb = _detect(img)
    t0 = time.time()
    answer, parts = _read_all(img, boxes, chars, t0, ROUND_SECONDS - 1.0)
    dt = time.time() - t0
    ann = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for b, (text, src) in zip(boxes, parts):
        cv2.rectangle(ann, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 2)
        cv2.putText(ann, f"{text or '-'} ({src})", (b[0] + 3, max(14, b[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if tb:
        cv2.circle(ann, (int(tb[0]), int(tb[1])), 7, (255, 128, 0), 2)
        cv2.putText(ann, "textbox", (int(tb[0]) + 10, int(tb[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1)
    cv2.imwrite(os.path.join(HERE, "last_debug.png"), ann)
    print(f"captcha boxes found: {len(boxes)}  {[tuple(b) for b in boxes]}")
    print(f"text box: {tb if tb else 'NOT FOUND (auto-detect failed)'}")
    for i, (text, src) in enumerate(parts):
        flag = "" if len(text) == chars else f"  WARN: {len(text)} chars!"
        print(f"  captcha {i+1}: {text!r} via {src}{flag}")
    print(f"ANSWER: {answer!r}  (read in {dt:.2f}s)")
    print("saved last_capture.png + last_debug.png — send both if wrong")


def run():
    cfg = _load_cfg()
    chars = cfg.get("chars", 6)
    region = cfg["region"]
    print(f"region: {region['width']}x{region['height']} px, chars/captcha: {chars}, "
          f"submit: {cfg['submit_mode']}")
    print("bot running — mouse to a screen corner to abort.\n")
    last_key = None
    prev = []
    tb_warned = False
    with _MSS() as sct:
        while True:
            img = _grab(sct, region)
            cands, tb = _detect(img, max_boxes=4)
            if not cands:
                time.sleep(POLL)
                continue
            dyn = finder.dynamic_boxes(img, cands, prev)
            moving = sorted([(b, h) for b, h, s in dyn if not s], key=lambda t: t[0][0])
            prev = [(b, h) for b, h, _s in dyn]
            if not moving:
                time.sleep(POLL)
                continue
            key = finder.round_key([h for _b, h in moving])
            if key == last_key:
                time.sleep(POLL)
                continue
            last_key = key
            boxes = [b for b, _h in moving[:2]]
            t0 = time.time()
            answer, parts = _read_all(img, boxes, chars, t0, ROUND_SECONDS - 1.0)
            t_read = time.time() - t0
            if not answer:
                print(f"[{time.strftime('%H:%M:%S')}] no read — waiting for next round")
                time.sleep(0.5)
                continue
            if time.time() - t0 > ROUND_SECONDS - 3.0:
                print(f"[{time.strftime('%H:%M:%S')}] read took {t_read:.1f}s — "
                      "round expired, skipping submit")
                continue
            target = tb or (tuple(cfg["textbox_fallback"]) if cfg.get("textbox_fallback") else None)
            if target is None:
                if not tb_warned:
                    print("WARN: text box not found — set a fallback point via calibrate")
                    tb_warned = True
                time.sleep(0.5)
                continue
            warn = ""
            exp_len = chars * len(parts) + len(parts) - 1
            if len(answer.replace(" ", "")) != chars * len(parts):
                warn = f"  WARN: expected {len(parts)}x{chars} chars"
            pyautogui.click(*target)
            time.sleep(0.08)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            _type(answer)
            if cfg["submit_mode"] == "click" and cfg.get("submit"):
                pyautogui.click(*cfg["submit"])
            else:
                pyautogui.press("enter")
            dt = time.time() - t0
            srcs = "+".join(s for _, s in parts)
            print(f"[{time.strftime('%H:%M:%S')}] {answer!r} via {srcs} "
                  f"({len(parts)} captcha{'' if len(parts)==1 else 's'}, "
                  f"read {t_read:.1f}s, submitted at {dt:.1f}s){warn}")
            time.sleep(0.8)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"calibrate": calibrate, "run": run, "once": once}[cmd]()
