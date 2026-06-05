"""
Ensemble Module.

This module provides ensemble generation and judging capabilities
for improved response quality through voting and selection.
"""

from .ensemble_generator import EnsembleGenerator
from .judge import ResponseJudge

__all__ = ["EnsembleGenerator", "ResponseJudge"]
