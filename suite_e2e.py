"""End-to-end suite: whole real-page clones -> detect -> read -> space-join.
Scores EXACT page answers the way the game does (dual = 'abc xyz')."""
import random, sys, time
import numpy as np
import finder, realpage, solver


def run(N=60):
    det = ex = 0
    ex_s = n_s = ex_d = n_d = 0
    lat = []
    bad = []
    for seed in range(N):
        random.seed(8080 + seed)
        dual = seed % 2 == 1
        page, meta = realpage.render_page(dual=dual, seed=seed, meta=True)
        t0 = time.time()
        boxes, tb = finder.detect_all(page)
        parts = []
        for b in boxes:
            crop = page.crop((b[0], b[1], b[0] + b[2], b[1] + b[3]))
            parts.append(solver.solve(crop, None))
        ans = " ".join(p for p in parts if p)
        lat.append(time.time() - t0)
        want = " ".join(meta["texts"])
        ious = [max(finder.box_iou(b, r) for r in meta["boxes"]) for b in boxes] \
            if boxes else [0]
        det += len(boxes) == len(meta["boxes"]) and min(ious) > 0.55
        ok = ans == want
        ex += ok
        if dual:
            n_d += 1
            ex_d += ok
        else:
            n_s += 1
            ex_s += ok
        if not ok and len(bad) < 6:
            bad.append((seed, dual, ans, want))
    print(f"e2e pages: detect {det}/{N}   EXACT {ex}/{N} = {ex/N:.0%}   "
          f"(single {ex_s}/{n_s}, dual {ex_d}/{n_d})")
    print(f"latency mean {np.mean(lat)*1000:.0f}ms  max {np.max(lat)*1000:.0f}ms")
    for b in bad:
        print(f"   miss seed{b[0]} dual={b[1]}: got {b[2]!r} want {b[3]!r}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
