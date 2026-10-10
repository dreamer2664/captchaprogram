"""Local-reader benchmark on the three bars: easy (92 target), hard (85), real."""
import random, sys, time
import numpy as np
import generator, solver

def run(N=100, reader=None):
    reader = reader or (lambda img, n: solver.solve(img, n))
    for style, target in (("easy", 92), ("hard", 85), ("real", None)):
        ex = ch = tot = 0; t0 = time.time()
        for seed in range(N):
            random.seed(4242 + seed * 3 + len(style))
            img, truth = generator.generate(style=style)
            guess = reader(img, None)          # length-agnostic on purpose
            ex += guess == truth
            for a, b in zip(truth, guess):
                tot += 1; ch += a == b
            if len(guess) != len(truth):
                tot += abs(len(guess) - len(truth))
                # length mismatch: missing/extra chars count as wrong
        dt = (time.time() - t0) / N
        tgt = f"  target {target}" if target else ""
        print(f"{style:5s}: exact {ex:3d}/{N} = {ex/N:5.1%}{tgt}   "
              f"per-char {ch/tot:5.1%}   {dt*1000:.0f}ms/img")

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
