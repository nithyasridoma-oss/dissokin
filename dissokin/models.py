"""Release kinetic models for in vitro dissolution profiles.

Each model maps time (h) and parameters to cumulative percent released.
Parameter bounds follow the physical meaning of each constant.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class Model:
    name: str
    func: Callable[..., np.ndarray]
    param_names: tuple
    p0: tuple
    bounds: tuple
    mechanism_note: str = ""

    @property
    def k(self) -> int:
        """Number of free parameters."""
        return len(self.param_names)


def _zero_order(t, k0):
    return k0 * t


def _first_order(t, k1):
    # Q = 100 (1 - exp(-k1 t))
    return 100.0 * (1.0 - np.exp(-k1 * t))


def _higuchi(t, kh):
    return kh * np.sqrt(t)


def _korsmeyer_peppas(t, kkp, n):
    return kkp * np.power(t, n)


def _hixson_crowell(t, khc):
    # 100^(1/3) - Q_remaining^(1/3) = k t  ->  Q = 100 (1 - (1 - k t)^3)
    x = np.clip(1.0 - khc * t, 0.0, None)
    return 100.0 * (1.0 - x ** 3)


def _weibull(t, alpha, beta):
    # Q = 100 (1 - exp(-(t^beta)/alpha))
    with np.errstate(over="ignore"):
        return 100.0 * (1.0 - np.exp(-(np.power(t, beta)) / alpha))


def _hopfenberg(t, kho, n):
    x = np.clip(1.0 - kho * t, 0.0, None)
    return 100.0 * (1.0 - x ** n)


def _baker_lonsdale(t, kbl):
    """Baker-Lonsdale, solved numerically for Q given t.

    3/2 [1 - (1 - Q/100)^(2/3)] - Q/100 = k t

    The left hand side is monotone increasing in Q on [0, 1], so a vectorised
    bisection converges to machine-useful precision in 60 halvings.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    target = kbl * t
    lo = np.zeros_like(t)
    hi = np.ones_like(t)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = 1.5 * (1.0 - np.power(1.0 - mid, 2.0 / 3.0)) - mid
        left = val < target
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    return 100.0 * 0.5 * (lo + hi)


MODELS: tuple = (
    Model("Zero order", _zero_order, ("k0",), (5.0,), ([1e-9], [1e4]),
          "Constant rate; membrane-controlled or saturated reservoir."),
    Model("First order", _first_order, ("k1",), (0.2,), ([1e-9], [1e3]),
          "Rate proportional to remaining load; typical of porous matrices."),
    Model("Higuchi", _higuchi, ("kH",), (20.0,), ([1e-9], [1e4]),
          "Fickian diffusion from a planar matrix; Q proportional to sqrt(t)."),
    Model("Korsmeyer-Peppas", _korsmeyer_peppas, ("kKP", "n"), (20.0, 0.5),
          ([1e-9, 0.0], [1e4, 2.0]),
          "Semi-empirical; n classifies the transport mechanism."),
    Model("Hixson-Crowell", _hixson_crowell, ("kHC",), (0.05,), ([1e-9], [10.0]),
          "Erosion or dissolution with changing surface area."),
    Model("Weibull", _weibull, ("alpha", "beta"), (10.0, 1.0),
          ([1e-9, 1e-3], [1e6, 10.0]),
          "Empirical; beta relates to the shape of the release curve."),
    Model("Hopfenberg", _hopfenberg, ("kHB", "n"), (0.05, 1.0),
          ([1e-9, 0.5], [10.0, 3.0]),
          "Surface-eroding devices; n encodes geometry (1 slab, 2 cylinder, 3 sphere)."),
    Model("Baker-Lonsdale", _baker_lonsdale, ("kBL",), (0.05,), ([1e-9], [10.0]),
          "Diffusion-controlled release from a spherical matrix."),
)

MODELS_BY_NAME = {m.name: m for m in MODELS}


def peppas_mechanism(n: float, geometry: str = "slab") -> str:
    """Classify transport mechanism from the Korsmeyer-Peppas exponent.

    Boundaries depend on device geometry (Ritger and Peppas, 1987).
    """
    bounds = {
        "slab": (0.45, 0.89),
        "cylinder": (0.45, 0.89),
        "sphere": (0.43, 0.85),
    }
    lo, hi = bounds.get(geometry, bounds["slab"])
    if n < lo:
        return "Quasi-Fickian diffusion"
    if np.isclose(n, lo, atol=1e-9) or n < hi:
        return "Anomalous (non-Fickian) transport"
    if np.isclose(n, hi, atol=1e-9):
        return "Case II transport (zero order)"
    return "Super Case II transport"
