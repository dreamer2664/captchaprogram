# LIVE RUNNER — Windows setup (read + type on YOUR pc)

## 1. Install
- Python 3.11+ from python.org (tick "Add to PATH")
- In the project folder:
      pip install -r requirements.txt
  (smaller torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first)
- `glyphnet.pt` (trained weights) must sit next to local_runner.py — it's in the zip.

## 2. Calibrate (once) — ONE BIG REGION
      python local_runner.py calibrate
- Frame a GENEROUS area that always contains the captcha image(s) AND the
  answer text box — the software now FINDS everything itself every round:
    * image size changes? handled (auto-detected + rescaled to model size)
    * image moves around? handled
    * TWO captchas (dual mode)? handled — answer is typed "text1 text2"
      with the space, exactly as the page expects
    * page furniture (titles, labels, timer bar, buttons)? ignored — the
      runner learns static content after round 1 and skips it
- Optional fallback text-box point (only used if auto-detect ever fails).
- Submit: Enter (default, fastest) or calibrated button click.
- It asks "Characters PER captcha? [6]" — enforced everywhere; wrong-length
  reads trigger the fallbacks instead of being typed.
- Saves `runner_config.json` (keep it when re-extracting new zips!).

## 3. Test without typing
      python local_runner.py once
- Prints: boxes found, text-box position, per-captcha read + source,
  final ANSWER (space-joined in dual mode) + latency.
- Saves `last_capture.png` (raw grab) and `last_debug.png`
  (green boxes = detected captchas with their reads, blue circle = detected
  text box).
- **When a read is wrong, send me those two PNGs** — that's how I tune the
  model to YOUR exact captcha (noise strength, fonts, scale).

## 4. Gemini keys (optional second opinion)
- Keys one per line in `gemini_keys.txt` (or env `GEMINI_API_KEYS=k1,k2,...`).
- Model: env `GEMINI_MODEL` (default `gemini-2.5-flash`).
- The 15 s law: Gemini is called ONLY when the local confidence is low AND
  remaining time > its timeout + typing + margin. Hard deadline: each
  request's timeout shrinks to fit the budget (even-sliced across attempts,
  so one slow call can't eat the round); if it stalls, the local answer
  goes in anyway. Rounds are never gambled, and blank/expired regions
  never reach the network.
- With keys loaded, the second opinion VOTES: 2-3 reads, length-checked,
  majority wins, early exit on the first agreement. Wrong-length reads are
  discarded. Measured worst case (20 s-latency network, dual mode): the
  whole solve stays inside 13 s — the 15 s round law holds.

## 5. Run the bot
      python local_runner.py run
- Polls the region 5×/s, detects new rounds by image hash,
  reads (~0.1 s typical), clicks the box, Ctrl+A/Del, types with 30-70 ms
  human jitter, submits (Enter or click per calibration).
- If a read somehow takes longer than the round allows, the submit is
  SKIPPED (the round is already dead — no point burning time typing).
- Console logs per round: text, reader, read-time, submit-time, warnings.
- ABORT ANY TIME: mouse into a screen corner (pyautogui failsafe).

## Accuracy bars & how they are measured
- `python3 suite_read.py 100`  — local reader only, three suites:
    easy  = clean binary cards (white/black, black/white)      bar: 92% exact
    hard  = designed backgrounds (objects, patterns), mixed
            fonts/colors/sizes, 4-8 chars                      bar: 85% exact
    real  = replica of your real page's cards
- `python3 suite_detect.py 100` — finder on the full real-page clone:
    captcha boxes (single+dual), red-ring textbox, 14s-notice immunity.
- `python3 suite_e2e.py 60`     — detect + read + space-join on whole pages.
- The reader is a multi-task CNN (36-way case-folded identity + 3-way case
  head) trained on all three styles at once with segmentation jitter.

## Troubleshooting
- Wrong boxes detected on the FIRST round only: normal — the static-
  furniture filter needs one round of history; from round 2 it's clean.
- Reads full of junk chars: captcha noise stronger than the model knows —
  send last_capture.png + last_debug.png for a retrain on your exact style.
- "no read — waiting": region doesn't contain the captcha right now
  (page scrolled/moved) — re-calibrate the big region.
- Slow first round only: torch model load (~2-5 s once per process).
- Dual mode IS implemented: both captchas are found, read, and
  joined with a SPACE between them ("abc123 xyz789").
- After a page restyle: re-calibrate, and send a screenshot for retraining.

## Accuracy log (sandbox suites, local reader only — no Gemini keys here)

| date | change | easy exact | hard exact | real exact | e2e exact |
|---|---|---|---|---|---|
| oct10 am | v4 weights, broken generator | 7 | 0 | 0 | - |
| oct10 pm | RGBA-alpha fix + anchor fix + dom-ink seg | 30 | 13 | 13 | 3 |
| oct10 pm | + dom sign-after-ring fix (REAL pages were polarity-dead!) | - | - | - | rising |
| oct10 eve | v5 multitask (14 ep, 24k) | 12 | 5 | 7 | 2 |
| oct10 night | v7 on-the-fly 30ep + geometry cues | 72 | 27 | 30 | 8 |
| oct10 night | v8 wide fc384, 40 ep; geometric post-passes OFF for multitask | 93 | 63 | 69 | - |
| oct10 night | + uniform card sizes, gaps 1-5, no-ink speckle, loose valley | 93 | 63 | 69 | - |
| oct11 | v9 hard-mix 50 ep + STYLE ROUTING (see below) | 93 | 76 | 82 | 67 |
| training | v10 hard specialist (hard:real 4:1, erosion aug), 60 ep | ? | ? | ? | ? |

v8 held-out per-char: easy 99.3 / hard 89.4 / real 88.2.
v9 held-out per-char: easy 99.6 / hard 98.5 / real 97.5.

STYLE ROUTING: the solver sniffs each card with a PCA test (are all pixels on
one line in RGB = exactly two inks?).  Binary black/white cards go to
`glyphnet_easy.pt` (v8 easy specialist, exact 93 >= bar 92); everything else
goes to `glyphnet.pt` (hard-mix weights).  Both files must sit next to
glyphnet.py; the runner finds them whichever folder you launch it from.

Geometric case/descender/O-0 post-passes run ONLY for legacy v4 weights --
on v8/v9 the multitask case head beats every hand rule (A/B: 92 vs 75 easy).

Bugs fixed oct10 (all silently catastrophic before):
1. ImageDraw on RGBA writes alpha raw -> designs/noise rendered SOLID (looked
   like ink). Now drawn on transparent overlays + true alpha_composite.
2. Glyph paste off by half advance vs recorded anchor -> train/infer mismatch.
3. dom sign computed BEFORE frame-ring strip -> any loose crop (page bg ring)
   flipped polarity -> dead dom -> '111111' garbage. THIS killed real runs.
4. Legacy v4 weights (net.* keys) wouldn't load into v5 class -> remap added.
