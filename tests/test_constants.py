"""Sanity checks on physical constants and unit conversions in isofate.constants."""

import numpy as np
import pytest

from isofate import constants as c


def test_time_conversions_are_consistent():
    assert 1 / c.s2day == pytest.approx(86400)
    assert 1 / c.s2yr == pytest.approx(86400 * 365)
    assert c.s2yr / c.s2day == pytest.approx(1 / 365)


@pytest.mark.parametrize(
    "molar, particle",
    [
        (c.M_H, c.mu_H),
        (c.M_D, c.mu_D),
        (c.M_He, c.mu_He),
        (c.M_O, c.mu_O),
        (c.M_C, c.mu_C),
        (c.M_N, c.mu_N),
        (c.M_S, c.mu_S),
        (c.M_Fe, c.mu_Fe),
        (c.M_H2, c.mu_H2),
        (c.M_HD, c.mu_HD),
    ],
)
def test_particle_masses_are_molar_mass_over_avogadro(molar, particle):
    assert particle == pytest.approx(molar / c.avogadro)


def test_molecular_molar_masses_match_atoms():
    assert c.M_H2 == pytest.approx(2 * c.M_H, rel=1e-3)
    assert c.M_HD == pytest.approx(c.M_H + c.M_D, rel=1e-3)
    assert c.M_H2O == pytest.approx(2 * c.M_H + c.M_O, rel=1e-3)


def test_species_mass_ordering():
    assert c.mu_H < c.mu_D < c.mu_He < c.mu_C < c.mu_O


def test_jupiter_to_earth_ratios():
    assert c.J2E_mass == pytest.approx(317.8, rel=1e-3)
    assert c.J2E_rad == pytest.approx(11.2, rel=1e-2)
    assert c.J2E_mass == pytest.approx(c.Mjup / c.Me, rel=1e-3)


def test_boltzmann_consistent_with_gas_constant():
    assert c.kb * c.avogadro == pytest.approx(c.R_gas, rel=1e-2)


def test_mass_ratios_derived_from_mole_ratios():
    assert c.DtoH_solar_mass == pytest.approx(c.DtoH_solar * c.mu_D / c.mu_H)
    assert c.OtoH_protosolar_mass == pytest.approx(c.OtoH_protosolar * c.mu_O / c.mu_H)
    assert c.CtoH_protosolar_mass == pytest.approx(c.CtoH_protosolar * c.mu_C / c.mu_H)


def test_earth_xuv_flux_from_ribas():
    # Ribas et al. 2005: 3.88 erg/cm2/s at 1 au today
    assert c.Fe_xuv == pytest.approx(3.88e-3)


def test_solar_D_to_H_is_below_ocean_value():
    # Earth's oceans are enriched in D relative to the protosolar nebula
    assert c.DtoH_solar < c.vsmow


@pytest.mark.parametrize(
    "fn",
    [c.b_H_D, c.b_H_He, c.b_He_D, c.b_H_O, c.b_He_O, c.b_H_C, c.b_He_C, c.b_H_N, c.b_He_N, c.b_H_S, c.b_He_S],
)
def test_binary_diffusion_coefficients_positive_and_increasing(fn):
    T = np.array([100.0, 500.0, 1000.0, 3000.0])
    b = fn(T)
    assert np.all(b > 0)
    assert np.all(np.diff(b) > 0)


def test_binary_diffusion_power_law_exponent():
    # b_H_O = 4.8e19 * T**0.75 -> doubling T scales by 2**0.75
    assert c.b_H_O(2000.0) / c.b_H_O(1000.0) == pytest.approx(2**0.75)
    assert c.b_H_He(2000.0) / c.b_H_He(1000.0) == pytest.approx(2**0.732)
