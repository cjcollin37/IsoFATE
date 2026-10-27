"""
Tests for the incident XUV flux models in isofate.isofunks.

Times passed to the flux functions are in seconds; t0, t_sat and t_pms are in years.
"""

import numpy as np
import pytest

from isofate.constants import Rs, au2m, cgs2si_flux, s2yr
from isofate.isofunks import Fxuv, Fxuv_a, Fxuv_hazmat, Fxuv_Ribas, Fxuv_SF, phi_E


def yr(t_years):
    """Convert years to seconds."""
    return t_years / s2yr


F0 = 10.0  # [W/m2]
T_SAT = 5e8  # [yr]


# ---------------------------------------------------------------- Fxuv


@pytest.mark.parametrize("age", [1e6, 1e7, 1e8, 4.99e8])
def test_flux_is_saturated_before_t_sat(age):
    assert Fxuv(yr(age), F0) == pytest.approx(F0)


@pytest.mark.parametrize("factor", [1.0, 2.0, 10.0])
def test_flux_follows_power_law_after_t_sat(factor):
    beta = -1.23
    assert Fxuv(yr(factor * T_SAT), F0, beta=beta) == pytest.approx(F0 * factor**beta)


def test_flux_is_continuous_at_t_sat():
    eps = 1e-6
    before = Fxuv(yr(T_SAT * (1 - eps)), F0)
    after = Fxuv(yr(T_SAT * (1 + eps)), F0)
    assert after == pytest.approx(before, rel=1e-5)


def test_flux_decreases_monotonically_after_saturation():
    ages = np.logspace(np.log10(T_SAT), 10, 50)
    fluxes = np.array([Fxuv(yr(a), F0) for a in ages])
    assert np.all(np.diff(fluxes) < 0)


def test_steeper_beta_decays_faster():
    age = yr(5e9)
    assert Fxuv(age, F0, beta=-2.0) < Fxuv(age, F0, beta=-1.0)


def test_custom_saturation_time():
    assert Fxuv(yr(1e8), F0, t_sat=5e7) == pytest.approx(F0 * 2**-1.23)


def test_step_function_returns_final_flux_after_t_sat():
    assert Fxuv(yr(1e9), F0, step_fn=True, F_final=0.17) == pytest.approx(0.17)
    assert Fxuv(yr(1e8), F0, step_fn=True, F_final=0.17) == pytest.approx(F0)


class TestPreMainSequence:
    T_PMS = 1e8
    PMS_FACTOR = 100.0

    def flux(self, age):
        return Fxuv(yr(age), F0, t0=1e6, t_pms=self.T_PMS, pms_factor=self.PMS_FACTOR)

    def test_starts_at_boosted_flux(self):
        assert self.flux(1e6) == pytest.approx(F0 * self.PMS_FACTOR)

    def test_joins_saturated_flux_at_end_of_pms(self):
        assert self.flux(self.T_PMS * (1 - 1e-9)) == pytest.approx(F0, rel=1e-6)
        assert self.flux(self.T_PMS) == pytest.approx(F0)

    def test_decreases_during_pms(self):
        ages = np.logspace(6, np.log10(self.T_PMS) - 1e-6, 30)
        fluxes = np.array([self.flux(a) for a in ages])
        assert np.all(np.diff(fluxes) < 0)

    def test_is_log_linear_between_t0_and_t_pms(self):
        # Halfway in log-time should be halfway in log-flux
        mid = np.sqrt(1e6 * self.T_PMS)
        assert self.flux(mid) == pytest.approx(np.sqrt(F0 * F0 * self.PMS_FACTOR))

    def test_no_pms_phase_by_default(self):
        assert Fxuv(yr(1e6), F0) == pytest.approx(F0)


# ---------------------------------------------------------------- Fxuv_a


def test_vectorised_flux_matches_scalar_flux():
    t = yr(np.logspace(6, 10, 40))
    expected = [Fxuv(ti, F0) for ti in t]
    np.testing.assert_allclose(Fxuv_a(t, F0), expected, rtol=1e-12)


# ---------------------------------------------------------------- other flux models


def test_ribas_luminosity_at_one_gyr():
    expected = 29.7 * cgs2si_flux * 4 * np.pi * au2m**2
    assert Fxuv_Ribas(yr(1e9)) == pytest.approx(expected)


def test_ribas_power_law_slope():
    assert Fxuv_Ribas(yr(2e9)) / Fxuv_Ribas(yr(1e9)) == pytest.approx(2**-1.23)


def test_sanz_forcada_at_one_gyr():
    assert Fxuv_SF(yr(1e9)) == pytest.approx(10**22.12)


class TestHazmat:
    @pytest.mark.parametrize("activity", ["low", "medium", "high"])
    def test_inverse_square_with_distance(self, activity):
        t = yr(1e9)
        assert Fxuv_hazmat(t, 2 * au2m, activity) == pytest.approx(Fxuv_hazmat(t, au2m, activity) / 4)

    def test_value_at_stellar_reference_radius(self):
        # Flux tabulated at the stellar surface of a 0.515 Rsun star
        assert Fxuv_hazmat(yr(5e6), 0.515 * Rs, "high") == pytest.approx(45400)

    @pytest.mark.parametrize("age_myr", [5, 100, 500, 5000])
    def test_activity_quartiles_are_ordered(self, age_myr):
        t, d = yr(age_myr * 1e6), au2m
        assert Fxuv_hazmat(t, d, "high") >= Fxuv_hazmat(t, d, "medium") >= Fxuv_hazmat(t, d, "low")

    @pytest.mark.xfail(
        strict=True,
        reason="flux_45myr_med = 3661 is below the lower quartile (30288); likely a typo for 36661",
    )
    def test_activity_quartiles_are_ordered_at_45_myr(self):
        t, d = yr(30e6), au2m
        assert Fxuv_hazmat(t, d, "medium") >= Fxuv_hazmat(t, d, "low")

    def test_unknown_activity_returns_none(self):
        assert Fxuv_hazmat(yr(1e9), au2m, "extreme") is None


# ---------------------------------------------------------------- energy-limited escape


def test_energy_limited_mass_flux_power_law():
    t, eps, Vpot = yr(1e9), 0.15, 1e8
    expected = eps * Fxuv(t, F0) / (4 * Vpot)
    assert phi_E(t, eps, Vpot, au2m, F0) == pytest.approx(expected)


def test_energy_limited_mass_flux_phoenix():
    t, eps, Vpot, d = yr(1e9), 0.15, 1e8, 0.05 * au2m
    expected = eps * Fxuv_hazmat(t, d, "high") / (4 * Vpot)
    assert phi_E(t, eps, Vpot, d, F0, activity="high", flux_model="phoenix") == pytest.approx(expected)


def test_energy_limited_mass_flux_passes_flux_options_through():
    t = yr(1e9)
    got = phi_E(t, 0.1, 1e8, au2m, F0, t_sat=1e8, beta=-2.0)
    assert got == pytest.approx(0.1 * F0 * 10**-2.0 / 4e8)
