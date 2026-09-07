"""Experiment 2: how reliable is a mechanism read off the Peppas exponent?

Profiles are generated from the Korsmeyer-Peppas law with a known exponent n.
Each replicate is truncated at 60 percent released, as the derivation of the
power law requires, and n is then estimated two ways: by nonlinear least
squares on percent released, and by the log-log linear regression that most
published analyses use. A residual bootstrap gives an interval for the
nonlinear estimate. Ground truth is known, so bias, interval coverage and
the reliability of the resulting mechanistic label can all be measured.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dissokin.models import MODELS_BY_NAME, peppas_mechanism
from dissokin.fitting import fit_model, fit_kp_loglog, first_60_percent
from dissokin.simulate import GRIDS, simulate_profile

KP = MODELS_BY_NAME["Korsmeyer-Peppas"]
N_TRUE = [0.35, 0.45, 0.60, 0.75, 0.95]
GRID_NAMES = ["log6", "log8", "log12"]
SIGMAS = [1.0, 2.0, 3.0, 5.0]
N_REP = 150
N_BOOT = 200
GEOM = "slab"
MIN_POINTS = 4
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "mechanism_study.json")


def _k_for(n_true, q_end=90.0, t_end=12.0):
    """Scale constant giving q_end percent released at t_end for this n."""
    return q_end / (t_end ** n_true)


def _boot_n(t, q, p_hat, n_boot, rng):
    fitted = KP.func(t, *p_hat)
    resid = q - fitted
    out = []
    for _ in range(n_boot):
        qb = fitted + rng.choice(resid, size=t.size, replace=True)
        try:
            popt, _ = curve_fit(KP.func, t, qb, p0=p_hat, bounds=KP.bounds,
                                maxfev=100000)
            out.append(popt[1])
        except Exception:
            continue
    return np.array(out)


def one_condition(args):
    n_true, grid_name, sigma, n_rep, n_boot, seed = args
    kkp = _k_for(n_true)
    t_full = np.array(GRIDS[grid_name].times, dtype=float)
    rng = np.random.default_rng(seed)
    true_class = peppas_mechanism(n_true, GEOM)

    n_nls, n_lin, widths, kept = [], [], [], []
    covered = cls_correct_nls = cls_correct_lin = 0
    ci_single_class = methods_agree = 0
    valid = 0
    for _ in range(n_rep):
        q_full = simulate_profile(KP, (kkp, n_true), t_full, sigma, rng)
        t, q = first_60_percent(t_full, q_full)
        if t.size < MIN_POINTS:
            continue
        fr = fit_model(t, q, KP)
        if not fr.converged:
            continue
        lin = fit_kp_loglog(t, q)
        if lin is None:
            continue
        valid += 1
        kept.append(t.size)
        nh = fr.params["n"]
        n_nls.append(nh)
        n_lin.append(lin["n"])
        draws = _boot_n(t, q, np.array([fr.params["kKP"], nh]), n_boot, rng)
        if draws.size >= 20:
            lo, hi = np.percentile(draws, [2.5, 97.5])
            widths.append(hi - lo)
            covered += (lo <= n_true <= hi)
            classes = {peppas_mechanism(v, GEOM) for v in draws}
            ci_single_class += (len(classes) == 1)
        cls_correct_nls += peppas_mechanism(nh, GEOM) == true_class
        cls_correct_lin += peppas_mechanism(lin["n"], GEOM) == true_class
        methods_agree += (peppas_mechanism(nh, GEOM)
                          == peppas_mechanism(lin["n"], GEOM))
    if valid == 0:
        return None
    nb = max(len(widths), 1)
    return {
        "n_true": n_true, "true_class": true_class, "grid": grid_name,
        "n_points_full": int(t_full.size), "sigma": sigma,
        "kKP_true": kkp, "n_valid": valid,
        "mean_points_kept": float(np.mean(kept)),
        "bias_nls": float(np.mean(n_nls) - n_true),
        "bias_loglog": float(np.mean(n_lin) - n_true),
        "sd_nls": float(np.std(n_nls, ddof=1)) if valid > 1 else float("nan"),
        "sd_loglog": float(np.std(n_lin, ddof=1)) if valid > 1 else float("nan"),
        "rmse_nls": float(np.sqrt(np.mean((np.array(n_nls) - n_true) ** 2))),
        "rmse_loglog": float(np.sqrt(np.mean((np.array(n_lin) - n_true) ** 2))),
        "ci_coverage": covered / nb,
        "mean_ci_width": float(np.mean(widths)) if widths else float("nan"),
        "ci_single_class": ci_single_class / nb,
        "class_correct_nls": cls_correct_nls / valid,
        "class_correct_loglog": cls_correct_lin / valid,
        "methods_agree": methods_agree / valid,
        "n_ci_replicates": len(widths),
    }


def main():
    jobs, seed = [], 990001
    for n_true in N_TRUE:
        for g in GRID_NAMES:
            for s in SIGMAS:
                seed += 1
                jobs.append((n_true, g, s, N_REP, N_BOOT, seed))
    print(f"{len(jobs)} conditions x {N_REP} replicates x {N_BOOT} bootstrap",
          flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(one_condition, jobs), 1):
            if r:
                rows.append(r)
            el = time.time() - t0
            print(f"  {i}/{len(jobs)}  {el:.0f}s, ~{el/i*(len(jobs)-i):.0f}s left",
                  flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"n_rep": N_REP, "n_boot": N_BOOT, "geometry": GEOM,
                   "rows": rows}, fh, indent=1)
    print(f"wrote {OUT} ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
