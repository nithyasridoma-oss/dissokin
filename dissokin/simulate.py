"""Data generating processes for the simulation study.

The point of this module is to produce release profiles whose true model and
true parameters are known, so that the behaviour of model selection rules and
of mechanistic classification can be measured against ground truth rather than
argued about.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .models import Model, MODELS_BY_NAME


@dataclass(frozen=True)
class TimeGrid:
    name: str
    times: tuple

    @property
    def n(self) -> int:
        return len(self.times)


def _log_grid(t_min: float, t_max: float, n: int) -> tuple:
    return tuple(np.round(np.geomspace(t_min, t_max, n), 4))


def _lin_grid(t_min: float, t_max: float, n: int) -> tuple:
    return tuple(np.round(np.linspace(t_min, t_max, n), 4))


# Sampling schedules in hours. The 8-point log schedule is close to the
# 5/10/15/30/45/60/90/120 min pattern used in immediate release testing;
# the 12 h window matches a sustained release monograph.
GRIDS = {
    "log6":  TimeGrid("log6",  _log_grid(0.25, 12.0, 6)),
    "log8":  TimeGrid("log8",  _log_grid(0.25, 12.0, 8)),
    "log10": TimeGrid("log10", _log_grid(0.25, 12.0, 10)),
    "log12": TimeGrid("log12", _log_grid(0.25, 12.0, 12)),
    "log16": TimeGrid("log16", _log_grid(0.25, 12.0, 16)),
    "lin6":  TimeGrid("lin6",  _lin_grid(1.0, 12.0, 6)),
    "lin8":  TimeGrid("lin8",  _lin_grid(1.0, 12.0, 8)),
    "lin12": TimeGrid("lin12", _lin_grid(1.0, 12.0, 12)),
}


def simulate_profile(model: Model | str, params, times, sigma: float,
                     rng: np.random.Generator,
                     noise: str = "additive",
                     clip: bool = True) -> np.ndarray:
    """One noisy realisation of a release profile.

    sigma is in percent released for additive noise, and is the relative
    standard deviation for proportional noise. Clipping to [0, 100] reflects
    that an assay cannot report a negative or above label release; it is on
    by default because leaving it off would flatter the fitting routines.
    """
    m = MODELS_BY_NAME[model] if isinstance(model, str) else model
    t = np.asarray(times, dtype=float)
    truth = m.func(t, *params)
    if noise == "additive":
        obs = truth + rng.normal(0.0, sigma, size=t.size)
    elif noise == "proportional":
        obs = truth * (1.0 + rng.normal(0.0, sigma, size=t.size))
    else:
        raise ValueError(f"unknown noise model: {noise}")
    if clip:
        obs = np.clip(obs, 0.0, 100.0)
    return obs


def truth_curve(model: Model | str, params, times) -> np.ndarray:
    m = MODELS_BY_NAME[model] if isinstance(model, str) else model
    return m.func(np.asarray(times, dtype=float), *params)


# Ground truth scenarios. Parameters are chosen so that each profile reaches
# a plausible extent of release (roughly 70 to 95 percent) over the 12 h
# window, which is what makes the models genuinely hard to tell apart.
SCENARIOS = {
    "Zero order":       ("Zero order", (7.0,)),
    "First order":      ("First order", (0.22,)),
    "Higuchi":          ("Higuchi", (26.0,)),
    "Korsmeyer-Peppas": ("Korsmeyer-Peppas", (28.0, 0.42)),
    "Hixson-Crowell":   ("Hixson-Crowell", (0.055,)),
    "Weibull":          ("Weibull", (6.0, 0.85)),
    "Baker-Lonsdale":   ("Baker-Lonsdale", (0.030,)),
}
