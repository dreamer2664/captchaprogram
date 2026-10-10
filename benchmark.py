"""
Headless batch benchmark:  python3 benchmark.py [N] [--gemini]

Scores every reader engine: exact match %, per-char %, time per image.
--gemini also queries Gemini vision (needs keys; slow, uses quota).
"""
import sys
import time

from generator import generate
import solver

ENGINES = ["glyphnet", "template", "tesseract", "rapidocr"]


def run(n: int = 50, use_gemini: bool = False) -> None:
    engines = ENGINES + (["gemini"] if use_gemini else [])
    hits = {e: 0 for e in engines}
    times = {e: 0.0 for e in engines}
    any_hits = 0
    char_sum = 0.0
    segok = 0
    misses = []

    for i in range(n):
        img, truth = generate()
        expected = len(truth)
        segok += len(solver.glyph_components(img)) == expected
        guesses = {}
        for e in engines:
            t0 = time.perf_counter()
            guesses[e] = getattr(solver, f"read_{e}")(img, expected)
            times[e] += time.perf_counter() - t0
            hits[e] += guesses[e] == truth
        ok = any(g == truth for g in guesses.values())
        any_hits += ok
        best = next((g for g in guesses.values() if g == truth),
                    guesses[engines[0]])
        char_sum += solver.char_accuracy(truth, best)
        if not ok:
            misses.append((truth, guesses))

    print(f"\n=== benchmark: {n} images (segmentation clean: {segok}/{n}) ===")
    for e in engines:
        print(f"{e:10s} exact: {hits[e]}/{n}  ({hits[e]/n:.0%})"
              f"   avg {times[e]/n:.2f}s/img")
    print(f"any engine:  {any_hits}/{n}  ({any_hits/n:.0%})")
    print(f"per-char accuracy (best engine): {char_sum/n:.1%}")
    if misses:
        print(f"\n--- first {min(10, len(misses))} misses ---")
        for truth, g in misses[:10]:
            detail = "  /  ".join(f"{e}:{g[e] or '—'}" for e in engines)
            print(f"  {truth}  ->  {detail}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(int(args[0]) if args else 50, "--gemini" in sys.argv)
