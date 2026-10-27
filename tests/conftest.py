"""
Shared pytest configuration for the IsoFATE test suite.

The repository root is added to sys.path so the tests run against the local
checkout without needing `pip install -e .` first.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isofate.constants import (  # noqa: E402
    Me,
    M_D,
    M_H,
    M_He,
    M_C,
    M_O,
    R_gas,
)


@pytest.fixture
def scale_heights():
    """Return a function giving per-species scale heights [m] for a given T [K] and g [m/s2]."""

    def _scale_heights(T=800.0, g=10.0):
        return {
            "H": R_gas * T / (M_H * g),
            "He": R_gas * T / (M_He * g),
            "D": R_gas * T / (M_D * g),
            "O": R_gas * T / (M_O * g),
            "C": R_gas * T / (M_C * g),
        }

    return _scale_heights


@pytest.fixture
def earth_mass():
    return Me
