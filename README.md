# ocr-typer — CAPTCHA reader + test harness + (soon) live PC runner

Rebuilt 2026-10-09 after total loss. Reads generated captchas (mixed-case
letters + digits, CASE SENSITIVE; backgrounds/fonts/sizes/colors all vary)
and types them into the test page within the time limit.

## Layout
- generator.py     synthetic generator, full random difficulty
- crnn.py          color-invariant contrast frontend + dormant CTC reader
- glyphnet.py      TRAINED per-glyph CNN (the main reader)  -> glyphnet.pt
- gemini_reader.py Gemini-vision second opinion, KEY ROTATION
                   keys: env GEMINI_API_KEYS or gemini_keys.txt (one/line)
                   model: env GEMINI_MODEL (default gemini-2.5-flash)
- solver.py        chain: glyphnet -> gemini (low confidence) -> templates
                   -> tesseract -> rapidocr; color-invariant segmentation
- templates.py     glyph template bank (heuristic fallback)
- harness_app.py   test page: 15 s bar, CASE SENSITIVE box, AUTOPILOT,
                   100-round test            -> python3 harness_app.py 5000
- benchmark.py     headless scorer           -> python3 benchmark.py 100
- local_runner.py  (coming) live screen read + type on YOUR pc

## Train
    python3 glyphnet.py train 8 15000
