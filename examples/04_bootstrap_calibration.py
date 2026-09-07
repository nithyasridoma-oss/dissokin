"""Experiment 4: is the bootstrap interval for n actually a 95 percent interval?

Experiment 2 reported coverage well below nominal, and the shortfall tracked
the number of points rather than the noise level, which is the signature of a
degrees of freedom problem rather than a noise problem. Fitted residuals are
shrunk relative to the true errors by roughly sqrt((n - k) / n), and with six
points and two parameters that is a quarter of the spread. This experiment
measures coverage with and without the standard sqrt(n / (n - k)) rescaling,
and against the Monte Carlo sampling distribution of the estimator itself,
which is the target the interval is supposed to cover.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dissokin.models import MODELS_BY_NAME
from dissokin.fitting import fit_model, first_60_percent
from dissokin.simulate import GRIDS, simulate_profile

KP = MODELS_BY_NAME["Korsmeyer-Peppas"]
N_TRUE, SIGMA = 0.60, 2.0
GRID_NAMES = ["log6", "log8", "log12"]
N_REP, N_BOOT = 200, 250
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "bootstrap_calibration.json")


def one_cell(args):
    grid_name, rescale, seed = args
    kkp = 90.0 / 12.0 ** N_TRUE
    t_full = np.array(GRIDS[grid_name].times, dtype=float)
    rng = np.random.default_rng(seed)
    widths, covered, n_hats, kept = [], 0, [], []
    for _ in range(N_REP):
        q_full = simulate_profile(KP, (kkp, N_TRUE), t_full, SIGMA, rng)
        t, q = first_60_percent(t_full, q_full)
        if t.size < 4:
            continue
        fr = fit_model(t, q, KP)
        if not fr.converged:
            continue
        p_hat = np.array([fr.params["kKP"], fr.params["n"]])
        n_hats.append(p_hat[1]); kept.append(t.size)
        fitted = KP.func(t, *p_hat)
        resid = q - fitted
        if rescale:
            resid = resid * np.sqrt(t.size / (t.size - KP.k))
        draws = []
        for _ in range(N_BOOT):
            qb = fitted + rng.choice(resid, size=t.size, replace=True)
            try:
                popt, _ = curve_fit(KP.func, t, qb, p0=p_hat, bounds=KP.bounds,
                                    maxfev=100000)
                draws.append(popt[1])
            except Exception:
                continue
        if len(draws) < 20:
            continue
        lo, hi = np.percentile(draws, [2.5, 97.5])
        widths.append(hi - lo)
        covered += (lo <= N_TRUE <= hi)
    mc_sd = float(np.std(n_hats, ddof=1))
    return {
        "grid": grid_name, "n_points_full": int(t_full.size),
        "rescaled": bool(rescale), "n_rep": len(widths),
        "mean_points_kept": float(np.mean(kept)),
        "coverage": covered / max(len(widths), 1),
        "mean_ci_width": float(np.mean(widths)),
        "mc_sd_of_nhat": mc_sd,
        "oracle_width": float(2 * 1.959964 * mc_sd),
        "width_ratio": float(np.mean(widths) / (2 * 1.959964 * mc_sd)),
    }


def main():
    jobs = [(g, r, 5150 + i * 7 + int(r))
            for i, g in enumerate(GRID_NAMES) for r in (False, True)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(one_cell, jobs):
            rows.append(r)
            print(f"  {r['grid']:6s} rescaled={str(r['rescaled']):5s} "
                  f"kept={r['mean_points_kept']:.1f} coverage={r['coverage']:.3f} "
                  f"width={r['mean_ci_width']:.3f} oracle={r['oracle_width']:.3f} "
                  f"ratio={r['width_ratio']:.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"n_true": N_TRUE, "sigma": SIGMA, "n_rep": N_REP,
                   "n_boot": N_BOOT, "rows": rows}, fh, indent=1)
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
