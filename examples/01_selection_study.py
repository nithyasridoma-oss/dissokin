"""Experiment 1: does the selection rule matter?

Profiles are generated from a known release model, then all eight candidate
models are fitted and ranked by four rules in common use: raw R, adjusted
R, AICc and BIC. Because the true model is known, the recovery rate of each
rule can be measured directly.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dissokin.fitting import fit_all
from dissokin.simulate import GRIDS, SCENARIOS, simulate_profile

GRID_NAMES = ["log6", "log8", "log10", "log16"]
SIGMAS = [1.0, 2.0, 3.0, 5.0]
N_REP = 400
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "selection_study.json")


def one_condition(args):
    truth_name, grid_name, sigma, n_rep, seed = args
    model_name, params = SCENARIOS[truth_name]
    t = np.array(GRIDS[grid_name].times, dtype=float)
    rng = np.random.default_rng(seed)
    rec = {"r2": 0, "r2_adj": 0, "aicc": 0, "bic": 0}
    disagree_r2_aicc = 0
    npar_r2, npar_aicc = [], []
    r2_best_all, r2_true_all, r2_gap = [], [], []
    picks_r2, picks_aicc = {}, {}
    n_valid = 0
    for _ in range(n_rep):
        q = simulate_profile(model_name, params, t, sigma, rng)
        res = [r for r in fit_all(t, q) if r.converged and np.isfinite(r.aicc)]
        if len(res) < 4:
            continue
        n_valid += 1
        by = {r.model: r for r in res}
        best_r2 = max(res, key=lambda r: r.r2)
        best_adj = max(res, key=lambda r: (r.r2_adj if np.isfinite(r.r2_adj) else -9e9))
        best_aicc = min(res, key=lambda r: r.aicc)
        best_bic = min(res, key=lambda r: r.bic)
        rec["r2"] += best_r2.model == truth_name
        rec["r2_adj"] += best_adj.model == truth_name
        rec["aicc"] += best_aicc.model == truth_name
        rec["bic"] += best_bic.model == truth_name
        disagree_r2_aicc += best_r2.model != best_aicc.model
        npar_r2.append(best_r2.n_params)
        npar_aicc.append(best_aicc.n_params)
        picks_r2[best_r2.model] = picks_r2.get(best_r2.model, 0) + 1
        picks_aicc[best_aicc.model] = picks_aicc.get(best_aicc.model, 0) + 1
        r2_best_all.append(best_r2.r2)
        if truth_name in by:
            r2_true_all.append(by[truth_name].r2)
            r2_gap.append(best_r2.r2 - by[truth_name].r2)
    if n_valid == 0:
        return None
    return {
        "truth": truth_name, "grid": grid_name, "n_points": len(t),
        "sigma": sigma, "n_valid": n_valid,
        "recovery_r2": rec["r2"] / n_valid,
        "recovery_r2_adj": rec["r2_adj"] / n_valid,
        "recovery_aicc": rec["aicc"] / n_valid,
        "recovery_bic": rec["bic"] / n_valid,
        "disagree_r2_aicc": disagree_r2_aicc / n_valid,
        "mean_npar_r2": float(np.mean(npar_r2)),
        "mean_npar_aicc": float(np.mean(npar_aicc)),
        "mean_r2_best": float(np.mean(r2_best_all)),
        "mean_r2_true": float(np.mean(r2_true_all)) if r2_true_all else float("nan"),
        "mean_r2_gap": float(np.mean(r2_gap)) if r2_gap else float("nan"),
        "picks_r2": picks_r2, "picks_aicc": picks_aicc,
    }


def main():
    jobs, seed = [], 20260907
    for truth in SCENARIOS:
        for g in GRID_NAMES:
            for s in SIGMAS:
                seed += 1
                jobs.append((truth, g, s, N_REP, seed))
    print(f"{len(jobs)} conditions x {N_REP} replicates "
          f"= {len(jobs)*N_REP} simulated profiles", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(one_condition, jobs), 1):
            if r:
                rows.append(r)
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s elapsed, "
                      f"~{el/i*(len(jobs)-i):.0f}s left", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"n_rep": N_REP, "rows": rows}, fh, indent=1)
    print(f"wrote {OUT}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
