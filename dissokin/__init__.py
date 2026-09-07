"""dissokin: release kinetics model fitting with honest uncertainty reporting.

The package fits the standard in vitro drug release models, ranks them by
information criteria rather than by R alone, and attaches bootstrap
confidence intervals to the quantities that get interpreted mechanistically.
"""
from .models import MODELS, MODELS_BY_NAME, Model, peppas_mechanism
from .fitting import (
    FitResult, fit_model, fit_all, bootstrap_params, mechanism_with_uncertainty,
)
from .similarity import f1_difference, f2_similarity, f2_with_uncertainty
from .simulate import simulate_profile, TimeGrid, GRIDS

__version__ = "0.1.0"

__all__ = [
    "MODELS", "MODELS_BY_NAME", "Model", "peppas_mechanism",
    "FitResult", "fit_model", "fit_all", "bootstrap_params",
    "mechanism_with_uncertainty",
    "f1_difference", "f2_similarity", "f2_with_uncertainty",
    "simulate_profile", "TimeGrid", "GRIDS",
]
