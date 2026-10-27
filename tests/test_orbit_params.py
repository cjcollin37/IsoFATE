"""Tests for isofate.orbit_params using well-known Solar System values."""

import pytest

from isofate.constants import Fe, Ls, Ms, Rs, au2m, s2yr
from isofate.orbit_params import EqTemp, HabZone, Insolation, Luminosity, Period, SemiMajor

T_SUN = 5772.0  # IAU nominal solar effective temperature [K]
YEAR_S = 1 / s2yr


def test_luminosity_of_sun():
    assert Luminosity(Rs, T_SUN) == pytest.approx(Ls, rel=1e-2)


def test_earth_semimajor_axis_from_one_year():
    assert SemiMajor(Ms, YEAR_S) == pytest.approx(au2m, rel=5e-3)


def test_earth_period_from_one_au():
    assert Period(au2m, Ms) == pytest.approx(YEAR_S, rel=5e-3)


@pytest.mark.parametrize("P_days", [1.0, 5.0, 37.0, 365.0])
def test_semimajor_and_period_are_inverses(P_days):
    P = P_days * 86400
    M = 0.3 * Ms
    assert Period(SemiMajor(M, P), M) == pytest.approx(P)


def test_insolation_at_earth():
    assert Insolation(Ls, au2m) == pytest.approx(Fe, rel=1e-2)


def test_insolation_inverse_square():
    assert Insolation(Ls, 2 * au2m) == pytest.approx(Insolation(Ls, au2m) / 4)


def test_earth_equilibrium_temperature():
    # zero albedo ~278 K, Bond albedo 0.3 ~255 K
    assert EqTemp(Fe) == pytest.approx(278.6, abs=1.0)
    assert EqTemp(Fe, A=0.3) == pytest.approx(254.9, abs=1.0)


def test_equilibrium_temperature_scales_with_quarter_power():
    assert EqTemp(16 * Fe) == pytest.approx(2 * EqTemp(Fe))


def test_runaway_greenhouse_limit_for_sun():
    # Kopparapu et al. 2013: Seff = 1.0385 for the Sun -> d = 1/sqrt(1.0385) au
    assert HabZone(Ls, 5780) / au2m == pytest.approx(1.0385**-0.5, rel=1e-6)


def test_habitable_zone_limits_are_ordered():
    zones = ["recent venus", "runaway greenhouse", "moist greenhouse", "max greenhouse", "early mars"]
    distances = [HabZone(Ls, 5780, HZ=z) for z in zones]
    assert distances == sorted(distances)


def test_habitable_zone_scales_with_sqrt_luminosity():
    assert HabZone(4 * Ls, 5780) == pytest.approx(2 * HabZone(Ls, 5780))


@pytest.mark.xfail(strict=True, reason="HabZone raises UnboundLocalError on an unknown HZ name instead of a clear ValueError")
def test_habitable_zone_rejects_unknown_zone():
    with pytest.raises(ValueError):
        HabZone(Ls, 5780, HZ="not a zone")
