"""Experiment 3: is an f2 similarity decision stable at n = 12 vessels?

f2 is computed from the mean profile of a small number of vessels, and a
release specification turns on whether it clears 50. This experiment fixes a
true difference between a reference and a test formulation and asks how often
a twelve vessel study lands on the correct side of the threshold.
"""
from __future__ import annotations
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dissokin.similarity import f2_similarity, f2_with_uncertainty
from dissokin.simulate import GRIDS, truth_curve

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "f2_study.json")

TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0], dtype=float)
REF_PARAMS = ("Weibull", (1.2, 0.95))
OFFSETS = [0.0, 4.0, 8.0, 10.0, 12.0, 15.0]   # percent, test slower than reference
VESSEL_SDS = [2.0, 4.0, 6.0]
N_VESSELS = [6, 12, 24]
N_MC = 2000


def main():
    ref_true = truth_curve(*REF_PARAMS, TIMES)
    rows = []
    rng = np.random.default_rng(4242)
    for off in OFFSETS:
        test_true = np.clip(ref_true - off, 0.0, 100.0)
        f2_true = f2_similarity(ref_true, test_true)
        for sd in VESSEL_SDS:
            for nv in N_VESSELS:
                vals = np.empty(N_MC)
                for i in range(N_MC):
                    R = np.clip(ref_true + rng.normal(0, sd, (nv, TIMES.size)), 0, 100)
                    T = np.clip(test_true + rng.normal(0, sd, (nv, TIMES.size)), 0, 100)
                    vals[i] = f2_similarity(R.mean(axis=0), T.mean(axis=0))
                correct = np.mean((vals >= 50.0) == (f2_true >= 50.0))
                rows.append({
                    "offset": off, "vessel_sd": sd, "n_vessels": nv,
                    "f2_true": float(f2_true),
                    "f2_mean": float(vals.mean()),
                    "f2_bias": float(vals.mean() - f2_true),
                    "f2_sd": float(vals.std(ddof=1)),
                    "p_pass": float(np.mean(vals >= 50.0)),
                    "decision_correct": float(correct),
                    "n_mc": N_MC,
                })
                print(f"off={off:5.1f} sd={sd} n={nv:2d}  f2_true={f2_true:5.1f} "
                      f"f2_mean={vals.mean():5.1f} sd={vals.std(ddof=1):4.2f} "
                      f"P(pass)={np.mean(vals>=50):.3f}", flush=True)

    # one worked example with a reported bootstrap interval
    R = np.clip(ref_true + rng.normal(0, 4.0, (12, TIMES.size)), 0, 100)
    T = np.clip(np.clip(ref_true - 10.0, 0, 100) + rng.normal(0, 4.0, (12, TIMES.size)), 0, 100)
    worked = f2_with_uncertainty(R, T, n_boot=5000, seed=1)
    worked["times_h"] = TIMES.tolist()
    worked["ref_mean"] = R.mean(axis=0).round(2).tolist()
    worked["test_mean"] = T.mean(axis=0).round(2).tolist()
    print("\nworked example:", json.dumps(worked, indent=1))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"times_h": TIMES.tolist(), "ref_params": list(REF_PARAMS[1]),
                   "ref_model": REF_PARAMS[0], "n_mc": N_MC,
                   "rows": rows, "worked_example": worked}, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
