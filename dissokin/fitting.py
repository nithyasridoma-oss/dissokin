"""Fitting, information-criterion model selection and bootstrap inference."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from scipy.optimize import curve_fit
from typing import Optional, Sequence
from .models import Model, MODELS, peppas_mechanism


@dataclass
class FitResult:
    model: str
    params: dict
    n_params: int
    n_points: int
    rss: float
    r2: float
    r2_adj: float
    aic: float
    aicc: float
    bic: float
    rmse: float
    converged: bool
    delta_aicc: float = float("nan")
    akaike_weight: float = float("nan")
    param_ci: Optional[dict] = None

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("param_ci", None)
        d["params"] = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return d


def _aic_gaussian(rss: float, n: int, k: int) -> float:
    """AIC for least squares with unknown, constant variance.

    The variance counts as an estimated parameter, hence k + 1.
    """
    return n * np.log(rss / n) + 2.0 * (k + 1)


def fit_model(t, q, model: Model, sigma=None) -> FitResult:
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    n = t.size
    try:
        popt, _ = curve_fit(
            model.func, t, q, p0=model.p0, bounds=model.bounds,
            maxfev=200000, sigma=sigma,
        )
        converged = True
    except Exception:
        popt = np.array(model.p0, dtype=float)
        converged = False

    resid = q - model.func(t, *popt)
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((q - q.mean()) ** 2))
    k = model.k
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    # adjusted R2 penalises parameter count
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    rss_safe = max(rss, 1e-12)
    aic = _aic_gaussian(rss_safe, n, k)
    kk = k + 1  # + variance
    denom = n - kk - 1
    aicc = aic + (2 * kk * (kk + 1)) / denom if denom > 0 else float("inf")
    bic = n * np.log(rss_safe / n) + np.log(n) * kk
    rmse = float(np.sqrt(rss / n))
    return FitResult(
        model=model.name,
        params={pn: float(pv) for pn, pv in zip(model.param_names, popt)},
        n_params=k, n_points=n, rss=rss, r2=r2, r2_adj=r2_adj,
        aic=aic, aicc=aicc, bic=bic, rmse=rmse, converged=converged,
    )


def fit_all(t, q, models: Sequence[Model] = MODELS, sigma=None) -> list:
    results = [fit_model(t, q, m, sigma=sigma) for m in models]
    ok = [r for r in results if r.converged and np.isfinite(r.aicc)]
    if ok:
        best = min(r.aicc for r in ok)
        weights = {}
        for r in results:
            r.delta_aicc = r.aicc - best if np.isfinite(r.aicc) else float("inf")
            weights[r.model] = np.exp(-0.5 * r.delta_aicc) if np.isfinite(r.delta_aicc) else 0.0
        tot = sum(weights.values())
        for r in results:
            r.akaike_weight = weights[r.model] / tot if tot > 0 else float("nan")
    results.sort(key=lambda r: (not r.converged, r.aicc))
    return results


def bootstrap_params(t, q, model: Model, n_boot: int = 2000, seed: int = 0,
                     method: str = "residual", rescale_residuals: bool = True) -> dict:
    """Bootstrap confidence intervals for the parameters of one model.

    method 'residual' resamples fit residuals (assumes homoscedasticity);
    method 'case' resamples (t, q) pairs.

    Fitted residuals are shrunk relative to the true errors, because the fit
    has already absorbed k degrees of freedom. With six points and two
    parameters that is a third of them, and a plain residual bootstrap then
    produces intervals that are visibly too narrow. rescale_residuals
    multiplies them by sqrt(n / (n - k)) before resampling, which is the
    standard correction. See examples/04_bootstrap_calibration.py for the
    coverage this buys.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    base = fit_model(t, q, model)
    if not base.converged:
        return {}
    p_hat = np.array(list(base.params.values()))
    resid = q - model.func(t, *p_hat)
    n = t.size
    if rescale_residuals and n > model.k:
        resid = resid * np.sqrt(n / (n - model.k))
    draws = []
    for _ in range(n_boot):
        if method == "residual":
            qb = model.func(t, *p_hat) + rng.choice(resid, size=n, replace=True)
            tb = t
        else:
            idx = rng.integers(0, n, size=n)
            tb, qb = t[idx], q[idx]
        try:
            popt, _ = curve_fit(model.func, tb, qb, p0=p_hat,
                                bounds=model.bounds, maxfev=200000)
            draws.append(popt)
        except Exception:
            continue
    if not draws:
        return {}
    arr = np.array(draws)
    out = {}
    for i, pn in enumerate(model.param_names):
        lo, hi = np.percentile(arr[:, i], [2.5, 97.5])
        out[pn] = {
            "estimate": float(p_hat[i]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "boot_sd": float(arr[:, i].std(ddof=1)),
            "n_success": int(arr.shape[0]),
        }
    return out


def mechanism_with_uncertainty(t, q, n_boot: int = 2000, seed: int = 0,
                               geometry: str = "slab") -> dict:
    """Classify Korsmeyer-Peppas mechanism and report classification stability.

    Returns the point classification plus the share of bootstrap replicates
    falling in each mechanistic class.
    """
    from .models import MODELS_BY_NAME
    kp = MODELS_BY_NAME["Korsmeyer-Peppas"]
    ci = bootstrap_params(t, q, kp, n_boot=n_boot, seed=seed)
    if not ci:
        return {}
    rng = np.random.default_rng(seed)
    base = fit_model(t, q, kp)
    n_hat = base.params["n"]
    # recover the bootstrap n draws by re-running (cheap: reuse sd assumption)
    # instead, redo the bootstrap capturing classes directly
    p_hat = np.array(list(base.params.values()))
    resid = q - kp.func(np.asarray(t, float), *p_hat)
    nn = len(t)
    if nn > kp.k:
        resid = resid * np.sqrt(nn / (nn - kp.k))
    classes = []
    ns = []
    for _ in range(n_boot):
        qb = kp.func(np.asarray(t, float), *p_hat) + rng.choice(resid, size=len(t), replace=True)
        try:
            popt, _ = curve_fit(kp.func, np.asarray(t, float), qb, p0=p_hat,
                                bounds=kp.bounds, maxfev=200000)
        except Exception:
            continue
        ns.append(popt[1])
        classes.append(peppas_mechanism(popt[1], geometry))
    if not classes:
        return {}
    uniq, counts = np.unique(np.array(classes), return_counts=True)
    shares = {u: float(c) / len(classes) for u, c in zip(uniq, counts)}
    ns = np.array(ns)
    return {
        "n_estimate": float(n_hat),
        "n_ci": [float(np.percentile(ns, 2.5)), float(np.percentile(ns, 97.5))],
        "point_classification": peppas_mechanism(n_hat, geometry),
        "class_shares": shares,
        "agreement": float(shares.get(peppas_mechanism(n_hat, geometry), 0.0)),
        "n_bootstrap": int(len(classes)),
    }


def first_60_percent(t, q, cutoff: float = 60.0):
    """Restrict a profile to the portion the Korsmeyer-Peppas law applies to.

    The power law is derived for the early stage of release and is
    conventionally fitted only to points below about 60 percent released
    (Ritger and Peppas, 1987). Applying the rule honestly is important,
    because it is what leaves a two parameter model with very few points.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    below = q <= cutoff
    # keep the leading run only, so that a noisy late point cannot re-enter
    if (~below).any():
        first_above = int(np.argmax(~below))
        keep = np.zeros_like(below)
        keep[:first_above] = True
    else:
        keep = below
    return t[keep], q[keep]


def fit_kp_loglog(t, q):
    """Korsmeyer-Peppas by ordinary least squares on log Q against log t.

    This linearisation is the form most often reported in the literature.
    It is not equivalent to nonlinear least squares on Q: taking logs
    reweights the points, so if the assay error is roughly constant in
    percent released the linearised estimate is biased.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = (t > 0) & (q > 0)
    if ok.sum() < 3:
        return None
    x = np.log(t[ok])
    y = np.log(q[ok])
    n_slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + n_slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": float(n_slope),
        "kKP": float(np.exp(intercept)),
        "r2_log": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(ok.sum()),
    }
