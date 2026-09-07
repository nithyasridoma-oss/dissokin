"""Build every figure in the report from the stored simulation results."""
from __future__ import annotations
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

from dissokin.models import MODELS, MODELS_BY_NAME, peppas_mechanism
from dissokin.fitting import fit_all, fit_model, bootstrap_params, first_60_percent
from dissokin.simulate import GRIDS, simulate_profile, truth_curve

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 160,
})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
      "#E69F00", "#56B4E9", "#7F7F7F", "#442288"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  figures/" + name + ".pdf")


# ---------------------------------------------------------------- figure 1
def fig1_identifiability():
    """All eight models fitted to one noisy Higuchi profile."""
    rng = np.random.default_rng(2024)
    t = np.array(GRIDS["log8"].times)
    q = simulate_profile("Higuchi", (26.0,), t, 2.0, rng)
    res = fit_all(t, q)
    tt = np.linspace(0.05, 12.5, 400)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.8),
                                  constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    for i, r in enumerate(res):
        m = MODELS_BY_NAME[r.model]
        ax.plot(tt, m.func(tt, *r.params.values()), lw=1.1, color=CB[i % len(CB)],
                label=f"{r.model} ($R^2$={r.r2:.4f})")
    ax.plot(t, q, "o", ms=4.5, mfc="white", mec="black", mew=1.0, zorder=5,
            label="simulated observations")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative release (%)")
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 105)
    ax.legend(fontsize=5.6, loc="upper left", ncol=1,
              handlelength=1.4, labelspacing=0.28,
              frameon=True, framealpha=0.88, edgecolor="none")
    ax.set_title("a  Eight models, one profile", loc="left", fontsize=8.5, weight="bold")

    names = [r.model for r in res][::-1]
    r2s = [r.r2 for r in res][::-1]
    ws = [r.akaike_weight for r in res][::-1]
    y = np.arange(len(names))
    ax2.barh(y - 0.2, r2s, height=0.38, color="#B0B0B0", label="$R^2$")
    ax2.barh(y + 0.2, ws, height=0.38, color="#0072B2", label="Akaike weight")
    for yi, v in zip(y, r2s):
        ax2.text(min(v, 1.0) + 0.012, yi - 0.2, f"{v:.4f}", va="center", fontsize=5.6)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=6.4)
    ax2.set_xlim(0, 1.28); ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.axvline(1.0, color="k", lw=0.5, ls=":")
    ax2.set_xlabel("Value")
    ax2.legend(fontsize=6.4, loc="lower right",
               frameon=True, framealpha=0.88, edgecolor="none")
    ax2.set_title("b  What each criterion says", loc="left", fontsize=8.5, weight="bold")
    save(fig, "fig1_identifiability")
    return res


# ---------------------------------------------------------------- figure 2
def fig2_recovery():
    rows = json.load(open(os.path.join(RES, "selection_study.json")))["rows"]
    rules = [("recovery_r2", "$R^2$"), ("recovery_r2_adj", "adj. $R^2$"),
             ("recovery_aicc", "AICc"), ("recovery_bic", "BIC")]
    npts = sorted({r["n_points"] for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6),
                             constrained_layout=True)
    ax = axes[0]
    means = [np.mean([r[k] for r in rows]) for k, _ in rules]
    ax.bar(range(4), means, color=["#B0B0B0", "#8C8C8C", "#0072B2", "#009E73"], width=0.62)
    for i, v in enumerate(means):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels([lab for _, lab in rules], fontsize=7)
    ax.set_ylabel("P(true model recovered)"); ax.set_ylim(0, max(means) * 1.35)
    ax.set_title("a  Pooled", loc="left", fontsize=8.5, weight="bold")

    ax = axes[1]
    for (k, lab), c in zip(rules, ["#B0B0B0", "#8C8C8C", "#0072B2", "#009E73"]):
        ax.plot(npts, [np.mean([r[k] for r in rows if r["n_points"] == n]) for n in npts],
                "o-", ms=3.5, lw=1.2, color=c, label=lab)
    ax.set_xlabel("Sampling points"); ax.set_ylabel("P(true model recovered)")
    ax.set_xticks(npts); ax.legend(fontsize=6.4)
    ax.set_title("b  Schedule density", loc="left", fontsize=8.5, weight="bold")

    ax = axes[2]
    for (k, lab), c in zip(rules, ["#B0B0B0", "#8C8C8C", "#0072B2", "#009E73"]):
        ax.plot(sigmas, [np.mean([r[k] for r in rows if r["sigma"] == s]) for s in sigmas],
                "o-", ms=3.5, lw=1.2, color=c, label=lab)
    ax.set_xlabel("Assay noise $\\sigma$ (% released)")
    ax.set_xticks(sigmas)
    ax.set_title("c  Assay noise", loc="left", fontsize=8.5, weight="bold")
    save(fig, "fig2_recovery")


# ---------------------------------------------------------------- figure 3
def fig3_r2_gap():
    rows = json.load(open(os.path.join(RES, "selection_study.json")))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7),
                             constrained_layout=True)

    ax = axes[0]
    gaps = [r["mean_r2_gap"] for r in rows if np.isfinite(r["mean_r2_gap"])]
    ax.hist(gaps, bins=28, color="#0072B2", alpha=0.85)
    ax.axvline(np.median(gaps), color="#D55E00", lw=1.2,
               label=f"median = {np.median(gaps):.4f}")
    ax.set_xlabel("$R^2$(best) $-$ $R^2$(true model)")
    ax.set_ylabel("Conditions")
    ax.legend(fontsize=6.6)
    ax.set_title("a  The margin R$^2$ decides on", loc="left", fontsize=8.5, weight="bold")

    ax = axes[1]
    truths = sorted({r["truth"] for r in rows})
    x = np.arange(len(truths))
    p_r2 = [np.mean([r["mean_npar_r2"] for r in rows if r["truth"] == tn]) for tn in truths]
    p_ai = [np.mean([r["mean_npar_aicc"] for r in rows if r["truth"] == tn]) for tn in truths]
    ktrue = [MODELS_BY_NAME[tn].k for tn in truths]
    ax.bar(x - 0.26, p_r2, 0.25, color="#B0B0B0", label="selected by $R^2$")
    ax.bar(x, p_ai, 0.25, color="#0072B2", label="selected by AICc")
    ax.bar(x + 0.26, ktrue, 0.25, color="#009E73", label="true model")
    ax.set_xticks(x)
    short = {"Baker-Lonsdale": "Baker-\nLonsdale", "First order": "First\norder",
             "Higuchi": "Higuchi", "Hixson-Crowell": "Hixson-\nCrowell",
             "Korsmeyer-Peppas": "Korsmeyer-\nPeppas", "Weibull": "Weibull",
             "Zero order": "Zero\norder"}
    ax.set_xticklabels([short[t] for t in truths], fontsize=5.4, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("Number of free parameters")
    ax.set_ylim(0, 2.4); ax.legend(fontsize=6.2, ncol=1)
    ax.set_title("b  Complexity of the chosen model", loc="left", fontsize=8.5, weight="bold")
    save(fig, "fig3_r2_gap")


# ---------------------------------------------------------------- figure 4
def fig4_mechanism():
    d = json.load(open(os.path.join(RES, "mechanism_study.json")))
    rows = d["rows"]
    sigmas = sorted({r["sigma"] for r in rows})
    npts = sorted({r["n_points_full"] for r in rows})
    ntrues = sorted({r["n_true"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6),
                             constrained_layout=True)

    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6, ls=":")
    for meth, c, lab in [("bias_nls", "#0072B2", "nonlinear least squares"),
                         ("bias_loglog", "#D55E00", "log-log regression")]:
        ax.plot(ntrues, [np.mean([r[meth] for r in rows if r["n_true"] == n]) for n in ntrues],
                "o-", ms=3.5, lw=1.2, color=c, label=lab)
    ax.set_xlabel("True exponent $n$"); ax.set_ylabel("Bias in $\\hat{n}$")
    ax.legend(fontsize=6.0, loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="none")
    ax.set_title("a  Estimator bias", loc="left", fontsize=8.5, weight="bold")

    ax = axes[1]
    for np_, c in zip(npts, CB):
        ax.plot(sigmas, [np.mean([r["mean_ci_width"] for r in rows
                                  if r["sigma"] == s and r["n_points_full"] == np_])
                         for s in sigmas], "o-", ms=3.5, lw=1.2, color=c,
                label=f"{np_} points")
    ax.axhline(0.44, color="#7F7F7F", lw=0.9, ls="--")
    ax.text(sigmas[0], 0.405, "anomalous band width (0.44)",
            fontsize=5.6, color="#555555")
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Assay noise $\\sigma$ (% released)")
    ax.set_ylabel("95% interval width for $n$")
    ax.set_xticks(sigmas); ax.legend(fontsize=6.2)
    ax.set_title("b  How well is $n$ pinned down?", loc="left", fontsize=8.5, weight="bold")

    ax = axes[2]
    ax.axvspan(0.43, 0.47, color="#7F7F7F", alpha=0.16)
    ax.axvspan(0.87, 0.91, color="#7F7F7F", alpha=0.16)
    for key, c, lab in [("class_correct_nls", "#0072B2", "point label correct"),
                        ("ci_single_class", "#D55E00", "interval fits one class")]:
        ax.plot(ntrues, [np.mean([r[key] for r in rows if r["n_true"] == n])
                         for n in ntrues], "o-", ms=3.5, lw=1.2, color=c, label=lab)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True exponent $n$")
    ax.set_ylabel("Proportion of profiles")
    ax.legend(fontsize=6.0, loc="lower left")
    ax.set_title("c  Distance to a boundary decides", loc="left",
                 fontsize=8.5, weight="bold")
    save(fig, "fig4_mechanism")


# ---------------------------------------------------------------- figure 5
def fig5_worked_example():
    """One profile taken through the full reporting pipeline."""
    rng = np.random.default_rng(99)
    t_full = np.array(GRIDS["log8"].times)
    n_true, kkp = 0.60, 90.0 / 12.0 ** 0.60
    q_full = simulate_profile("Korsmeyer-Peppas", (kkp, n_true), t_full, 3.0, rng)
    t, q = first_60_percent(t_full, q_full)
    kp = MODELS_BY_NAME["Korsmeyer-Peppas"]
    fr = fit_model(t, q, kp)
    ci = bootstrap_params(t, q, kp, n_boot=3000, seed=99)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6),
                             constrained_layout=True)
    ax = axes[0]
    tt = np.linspace(0.05, t_full.max() * 1.05, 300)
    ax.plot(tt, kp.func(tt, kkp, n_true), color="#7F7F7F", lw=1.0, ls="--",
            label=f"truth ($n$={n_true:.2f})")
    ax.plot(tt, kp.func(tt, *fr.params.values()), color="#0072B2", lw=1.3,
            label=f"fit ($\\hat{{n}}$={fr.params['n']:.3f})")
    ax.plot(t_full, q_full, "o", ms=4, mfc="white", mec="#B0B0B0", mew=0.9)
    ax.plot(t, q, "o", ms=4.5, mfc="white", mec="black", mew=1.1,
            label="points used ($\\leq$60%)")
    ax.axhline(60, color="#D55E00", lw=0.8, ls=":")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative release (%)")
    ax.legend(fontsize=6.2, loc="lower right")
    ax.set_title("a  Fit under the 60% rule", loc="left", fontsize=8.5, weight="bold")

    ax = axes[1]
    rngb = np.random.default_rng(99)
    fitted = kp.func(t, *fr.params.values())
    # same residual rescaling the package applies, so the histogram matches
    # the interval reported alongside it
    resid = (q - fitted) * np.sqrt(t.size / (t.size - kp.k))
    from scipy.optimize import curve_fit
    draws = []
    for _ in range(3000):
        qb = fitted + rngb.choice(resid, size=t.size, replace=True)
        try:
            popt, _ = curve_fit(kp.func, t, qb, p0=list(fr.params.values()),
                                bounds=kp.bounds, maxfev=100000)
            draws.append(popt[1])
        except Exception:
            pass
    draws = np.array(draws)
    ax.hist(draws, bins=45, color="#0072B2", alpha=0.85)
    ax.axvspan(0.0, 0.45, color="#D55E00", alpha=0.10)
    ax.axvspan(0.45, 0.89, color="#009E73", alpha=0.10)
    ax.axvspan(0.89, 2.0, color="#CC79A7", alpha=0.10)
    ax.axvline(0.45, color="k", lw=0.7, ls="--")
    ax.axvline(0.89, color="k", lw=0.7, ls="--")
    ax.axvline(fr.params["n"], color="#D55E00", lw=1.3)
    lo, hi = ci["n"]["ci_low"], ci["n"]["ci_high"]
    shares = {}
    for v in draws:
        c = peppas_mechanism(v)
        shares[c] = shares.get(c, 0) + 1
    top = max(shares.values()) / len(draws)
    ax.set_xlim(max(0, lo - 0.25), hi + 0.25)
    ax.set_xlabel("Bootstrap draws of $n$")
    ax.set_ylabel("Count")
    ax.set_title(f"b  95% CI [{lo:.2f}, {hi:.2f}], {top:.0%} in one class",
                 loc="left", fontsize=8.5, weight="bold")
    save(fig, "fig5_worked_example")
    return {"n_hat": fr.params["n"], "ci": [lo, hi],
            "shares": {k: v / len(draws) for k, v in shares.items()},
            "n_points_used": int(t.size), "n_points_total": int(t_full.size)}


# ---------------------------------------------------------------- figure 6
def fig6_f2():
    d = json.load(open(os.path.join(RES, "f2_study.json")))
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6),
                             constrained_layout=True)
    offsets = sorted({r["offset"] for r in rows})
    sds = sorted({r["vessel_sd"] for r in rows})

    ax = axes[0]
    for sd, c in zip(sds, CB):
        sub = [r for r in rows if r["vessel_sd"] == sd and r["n_vessels"] == 12]
        sub.sort(key=lambda r: r["offset"])
        ax.plot([r["offset"] for r in sub], [r["p_pass"] for r in sub],
                "o-", ms=3.5, lw=1.2, color=c, label=f"vessel SD = {sd}%")
    ft = {r["offset"]: r["f2_true"] for r in rows}
    cross = [o for o in offsets if ft[o] < 50]
    if cross:
        ax.axvline(min(cross), color="#7F7F7F", lw=0.8, ls="--")
    ax.set_xlabel("True offset between profiles (% released)")
    ax.set_ylabel("P(study declares similar)")
    ax.set_ylim(-0.03, 1.03); ax.legend(fontsize=6.2)
    ax.set_title("a  12 vessels, decision at $f_2$ = 50", loc="left",
                 fontsize=8.5, weight="bold")

    ax = axes[1]
    for nv, c in zip(sorted({r["n_vessels"] for r in rows}), CB):
        sub = [r for r in rows if r["n_vessels"] == nv and r["vessel_sd"] == 4.0]
        sub.sort(key=lambda r: r["offset"])
        ax.plot([r["offset"] for r in sub], [r["f2_sd"] for r in sub],
                "o-", ms=3.5, lw=1.2, color=c, label=f"{nv} vessels")
    ax.set_xlabel("True offset between profiles (% released)")
    ax.set_ylabel("SD of the reported $f_2$")
    ax.legend(fontsize=6.2)
    ax.set_title("b  Sampling error in $f_2$", loc="left", fontsize=8.5, weight="bold")
    save(fig, "fig6_f2")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    print("building figures")
    if "1" in which: fig1_identifiability()
    if "2" in which: fig2_recovery()
    if "3" in which: fig3_r2_gap()
    if "4" in which: fig4_mechanism()
    if "5" in which: print("   ", fig5_worked_example())
    if "6" in which: fig6_f2()
