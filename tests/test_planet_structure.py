"""
Tests for planetary structure, envelope and miscellaneous helper functions in
isofate.isofunks.
"""

import numpy as np
import pytest

from isofate import constants as c
from isofate import isofunks as f

Me, Re, Ms = c.Me, c.Re, c.Ms
GYR = 1e9 / c.s2yr  # 1 Gyr in seconds


# ---------------------------------------------------------------- Lopez & Fortney 2014 radii


def test_core_radius_of_earth_mass_planet():
    assert f.R_core(Me) == pytest.approx(Re)


def test_core_radius_quarter_power_scaling():
    assert f.R_core(16 * Me) == pytest.approx(2 * Re)


def test_envelope_radius_reference_point():
    # By construction: 1 Me, 5% envelope, Earth insolation, 5 Gyr -> 2.06 Re
    assert f.R_env(Me, 0.05, c.Fe, 5 * GYR) == pytest.approx(2.06 * Re)


def test_envelope_radius_non_thermal_matches_thermal_at_5_gyr():
    args = (5 * Me, 0.02, 30 * c.Fe)
    assert f.R_env(*args, 5 * GYR, thermal=False) == pytest.approx(f.R_env(*args, 5 * GYR))


def test_envelope_contracts_with_age():
    args = (5 * Me, 0.02, 30 * c.Fe)
    assert f.R_env(*args, 0.1 * GYR) > f.R_env(*args, 1 * GYR) > f.R_env(*args, 5 * GYR)


def test_envelope_radius_increases_with_envelope_fraction():
    radii = [f.R_env(5 * Me, fe, c.Fe, GYR) for fe in (0.001, 0.01, 0.1)]
    assert radii == sorted(radii)


def test_radiative_atmosphere_is_nine_scale_heights():
    Teq, Mp, Rc, Renv, mu = 800.0, 5 * Me, f.R_core(5 * Me), Re, c.mu_solar
    g = c.G * Mp / (Rc + Renv) ** 2
    assert f.R_atm(Teq, Mp, Rc, Renv, mu) == pytest.approx(9 * c.kb * Teq / (g * mu))


@pytest.mark.parametrize("thermal", [True, False])
@pytest.mark.parametrize("f_true", [0.005, 0.02, 0.1])
def test_envelope_fraction_inverts_radius_relation(thermal, f_true):
    Mp, Fp, Teq, mu, age = 6 * Me, 50 * c.Fe, 700.0, c.mu_solar, 2 * GYR
    Rc = f.R_core(Mp)
    Renv = f.R_env(Mp, f_true, Fp, age, thermal=thermal)
    Rp = Rc + Renv + f.R_atm(Teq, Mp, Rc, Renv, mu)
    assert f.f_env(Rc, Renv, Rp, Mp, Teq, mu, Fp, age, thermal=thermal) == pytest.approx(f_true, rel=1e-9)


# ---------------------------------------------------------------- characteristic radii


def test_bondi_radius_formula():
    Mp, mu, Teq, gamma = 5 * Me, c.mu_solar, 800.0, 7 / 5
    expected = (gamma - 1) * c.G * Mp * mu / (gamma * c.kb * Teq)
    assert f.R_Bondi(Mp, mu, Teq) == pytest.approx(expected)


def test_earth_hill_radius():
    # Earth's Hill sphere is ~1.5 million km
    assert f.R_Hill(Me, Ms, c.au2m) == pytest.approx(1.5e9, rel=0.02)


def test_tidal_reduction_factor_approaches_one_far_from_star():
    assert f.V_reduction(Me, Ms, 100 * c.au2m, Re) == pytest.approx(1.0, abs=1e-4)


def test_tidal_reduction_factor_decreases_close_in():
    far = f.V_reduction(5 * Me, 0.2 * Ms, 0.1 * c.au2m, 2 * Re)
    near = f.V_reduction(5 * Me, 0.2 * Ms, 0.01 * c.au2m, 2 * Re)
    assert 0 < near < far < 1


def test_primordial_radius_is_min_of_three_limits_for_arrays():
    Mp = np.array([2, 5, 10]) * Me
    args = dict(f_atm=0.02, Fp=50 * c.Fe, t0=1e6 / c.s2yr, T=700.0, mu=c.mu_solar, M_star=0.3 * Ms, d=0.05 * c.au2m)
    Rp = f.Rp_prim(Mp, **args)
    Rc = f.R_core(Mp)
    Renv = f.R_env(Mp, args["f_atm"], args["Fp"], args["t0"])
    R_LF = Rc + Renv + f.R_atm(args["T"], Mp, Rc, Renv, args["mu"])
    expected = np.minimum.reduce([R_LF, f.R_Bondi(Mp, args["mu"], args["T"]), f.R_Hill(Mp, args["M_star"], args["d"])])
    np.testing.assert_allclose(Rp, expected)


@pytest.mark.xfail(strict=True, reason="Rp_prim checks type(R_LF) != 'int' (always True) and so calls len() on scalars")
def test_primordial_radius_accepts_scalar_mass():
    args = (0.02, 50 * c.Fe, 1e6 / c.s2yr, 700.0, c.mu_solar, 0.3 * Ms, 0.05 * c.au2m)
    Rp = f.Rp_prim(5 * Me, *args)
    assert np.ndim(Rp) == 0
    assert Rp == pytest.approx(f.Rp_prim(np.array([5 * Me]), *args)[0])


# ---------------------------------------------------------------- surface conditions


def test_surface_pressure_is_weight_of_atmosphere():
    Mp, fatm = 5 * Me, 0.01
    Rc = f.R_core(Mp)
    expected = (c.G * Mp / Rc**2) * (Mp * fatm) / (4 * np.pi * Rc**2)
    assert f.P_surf(Mp, fatm) == pytest.approx(expected)


def test_surface_pressure_linear_in_envelope_fraction():
    assert f.P_surf(5 * Me, 0.02) == pytest.approx(2 * f.P_surf(5 * Me, 0.01))


def test_surface_temperature_is_dry_adiabat_from_rcb():
    Teq, Mp, fatm = 700.0, 5 * Me, 0.01
    k = (c.kb / c.mu_H2) / 14514
    expected = Teq * (f.P_surf(Mp, fatm) / 1e4) ** k
    assert f.T_surf(Teq, Mp, fatm) == pytest.approx(expected)


def test_surface_temperature_equals_teq_when_surface_is_rcb():
    Mp, Teq = 5 * Me, 700.0
    fatm = 1e4 / f.P_surf(Mp, 1.0)  # chosen so P_surf == 1e4 Pa (= Peq)
    assert f.T_surf(Teq, Mp, fatm) == pytest.approx(Teq)


@pytest.mark.xfail(
    strict=True,
    reason="R_rcb's correction term (k Ps^k / Tsurf)/(Prcb^k - Ps^k) is negative whenever Ps > Prcb, "
    "so the RCB lands below the core surface",
)
@pytest.mark.parametrize("fatm", [1e-5, 1e-3, 1e-2])
def test_rcb_radius_above_core(fatm):
    Mp = 5 * Me
    assert f.R_rcb(Mp, fatm, c.mu_H2, 2000.0) > f.R_core(Mp)


def test_radius_grid_isothermal_thin_atmosphere_matches_barometric_law():
    Mp, fatm, Teq, mu, n = 5 * Me, 1e-4, 300.0, c.mu_H2, 5000
    grid = f.R_grid(Mp, fatm, Teq, mu, k=0.0, n_tot=n)
    Rc = f.R_core(Mp)
    H = (c.R_gas / (mu * c.avogadro)) * Teq / (c.G * Mp / Rc**2)
    expected_height = H * np.log(f.P_surf(Mp, fatm) / 1e4)
    assert grid.shape == (n,)
    assert grid[0] == pytest.approx(Rc)
    assert np.all(np.diff(grid) > 0)
    assert grid[-1] - Rc == pytest.approx(expected_height, rel=0.1)


class TestAtmosphereDescent:
    args = (700.0, c.mu_H2, 2.5 * Re, 5 * Me, 7 / 5)

    def test_output_modes_are_consistent(self):
        Matm, T, p = f.make_atmosphere_descent(*self.args, 0)
        assert f.make_atmosphere_descent(*self.args, 1) == pytest.approx(Matm)
        Tsurf, psurf = f.make_atmosphere_descent(*self.args, 2)
        assert Tsurf == pytest.approx(T[0])
        assert psurf == pytest.approx(p[0])

    def test_profiles_increase_downward(self):
        Matm, T, p = f.make_atmosphere_descent(*self.args, 0)
        assert Matm > 0
        assert np.all(np.diff(p) < 0)
        assert np.all(np.diff(T) <= 0)
        assert T[-1] == pytest.approx(self.args[0])
        assert p[-1] == pytest.approx(0.2e5)

    def test_larger_planet_radius_holds_more_atmosphere(self):
        Tem, mu, _, Mc, gamma = self.args
        small = f.make_atmosphere_descent(Tem, mu, 2.0 * Re, Mc, gamma, 1)
        large = f.make_atmosphere_descent(Tem, mu, 2.5 * Re, Mc, gamma, 1)
        assert large > small


@pytest.mark.xfail(
    strict=True,
    reason="rplanet_fn uses scipy 'optimize' without importing it; once imported, fsolve's length-1 array "
    "also needs unwrapping before it reaches make_atmosphere_descent",
)
def test_rplanet_fn_inverts_atmosphere_descent():
    Teq, mu, Mc, gamma = 700.0, c.mu_H2, 5 * Me, 7 / 5
    Matm = f.make_atmosphere_descent(Teq, mu, 2.5 * Re, Mc, gamma, 1)
    assert f.rplanet_fn(Teq, mu, Mc, gamma, Matm) == pytest.approx(2.5 * Re, rel=1e-3)


# ---------------------------------------------------------------- melt fraction (packaged data grid)


class TestMeltFraction:
    def test_bounded_between_zero_and_one(self):
        T = np.linspace(10, 16000, 50)
        psi = f.MeltFraction(5 * Me, T)
        assert np.all((psi >= 0) & (psi <= 1))

    def test_solid_when_cold(self):
        assert f.MeltFraction(5 * Me, 300.0) == pytest.approx(0.0)

    def test_non_decreasing_with_temperature(self):
        T = np.linspace(10, 16000, 200)
        psi = f.MeltFraction(5 * Me, T)
        assert np.all(np.diff(psi) >= -1e-12)

    def test_accepts_arrays_of_masses(self):
        psi = f.MeltFraction(np.array([1, 5, 10]) * Me, 3000.0)
        assert psi.shape == (3,)

    @pytest.mark.parametrize("Mp, T", [(0.5 * Me, 3000.0), (25 * Me, 3000.0), (5 * Me, 20000.0)])
    def test_outside_grid_raises(self, Mp, T):
        # Grid covers 1-21 Earth masses and 10-16000 K
        with pytest.raises(ValueError):
            f.MeltFraction(Mp, T)


# ---------------------------------------------------------------- inventories and conversions


def test_terrestrial_oceans_to_hydrogen_atoms():
    N_H, n_TO = f.TO(5 * Me, n_TO=2.0)
    assert n_TO == 2.0
    assert N_H == pytest.approx(2 * 2.0 * c.n_OperTO * c.avogadro)


def test_envelope_fraction_to_terrestrial_oceans_round_trip():
    N_H_from_fatm, n_TO = f.TO(5 * Me, f_atm=0.01)
    N_H_from_TO, _ = f.TO(5 * Me, n_TO=n_TO)
    assert N_H_from_fatm == pytest.approx(N_H_from_TO)


def test_terrestrial_oceans_requires_an_argument(capsys):
    assert f.TO(5 * Me) is None
    assert "Error" in capsys.readouterr().out


def test_water_mass_fraction_for_one_earth_ocean():
    earth_ocean_kg = 1.4e21
    assert f.WMF(Me, earth_ocean_kg / Me) == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize(
    "N1, N2, expected",
    [
        (1, 0, 0.0),
        (0, 1, 1.0),
        (1, 1, c.M_He / (c.M_H + c.M_He)),
    ],
)
def test_mole_to_mass_fraction(N1, N2, expected):
    assert f.NtoM(N1, N2, c.M_H, c.M_He) == pytest.approx(expected)


class TestSpecificHeat:
    def test_h2_at_room_temperature(self):
        # NIST Shomate value for H2 at 298 K is 28.84 J/mol/K
        assert f.SpecHeatCap(298.15) == pytest.approx(28.84, abs=0.05)

    @pytest.mark.parametrize("T_boundary", [1000.0, 2500.0])
    def test_continuous_across_shomate_ranges(self, T_boundary):
        assert f.SpecHeatCap(T_boundary - 1e-6) == pytest.approx(f.SpecHeatCap(T_boundary), rel=1e-3)

    def test_constant_above_6000K(self):
        assert f.SpecHeatCap(7000.0) == 42
        assert f.SpecHeatCap(6000.0) == pytest.approx(42, rel=1e-2)


# ---------------------------------------------------------------- envelope predictions (Ginzburg et al. 2016)


def test_envelope_predictions_at_reference_values():
    assert f.f_atm_pred(Me) == pytest.approx(0.05)
    assert f.f_atm_pred2(Me, 1000.0) == pytest.approx(0.02)
    assert f.f_atm_pred3(Me, 1000.0) == pytest.approx(0.01)


def test_envelope_prediction_from_orbit_matches_teq_version():
    from isofate.orbit_params import EqTemp, Insolation, Luminosity, SemiMajor

    Mc, P, T_star, M_star, R_star = 5 * Me, 10 * 86400, 3500.0, 0.4 * Ms, 0.4 * c.Rs
    Teq = EqTemp(Insolation(Luminosity(R_star, T_star), SemiMajor(M_star, P)))
    assert f.f_atm_pred2_alt(Mc, P, T_star, M_star, R_star) == pytest.approx(f.f_atm_pred2(Mc, Teq))


def test_envelope_prediction_distribution_is_clipped_and_seedable():
    args = (5 * Me, 10 * 86400, 3500.0, 0.4 * Ms, 0.4 * c.Rs)
    np.random.seed(0)
    draws = [f.f_atm_pred2_alt(*args, distribution=True, sigma_fraction=5.0) for _ in range(200)]
    np.random.seed(0)
    again = [f.f_atm_pred2_alt(*args, distribution=True, sigma_fraction=5.0) for _ in range(200)]
    assert min(draws) >= 0.001
    assert draws == again


# ---------------------------------------------------------------- misc


def test_heating_efficiency_reference_point():
    # v_esc = 15 km/s -> eps = 0.1
    Mp = 5 * Me
    Rp = 2 * c.G * Mp / 15e3**2
    assert f.epsilon(Mp, Rp) == pytest.approx(0.1)


def test_radius_valley_membership():
    upper, lower = (0.0, np.log10(3.0)), (0.0, np.log10(1.5))  # flat valley between 1.5 and 3 Re
    assert f.radius_valley(10.0, 2.0, upper, lower) is True
    assert f.radius_valley(10.0, 1.0, upper, lower) is False
    assert f.radius_valley(10.0, 4.0, upper, lower) is False


def test_transmission_spectroscopy_metric():
    assert f.TSM(1.0, 1.0, 1.0, 1.0, 0.0) == pytest.approx(1.26)
    # TSM ~ Rp^3 Teq / (Mp Rstar^2) * 10^(-mJ/5)
    assert f.TSM(2.0, 500.0, 4.0, 0.5, 5.0) == pytest.approx(1.26 * 8 * 500 / (4 * 0.25) / 10)


def test_phi_kill_removes_atmosphere_tenfold_over_age():
    M_atm, Rp, age = 1e22, 2 * Re, GYR
    phi = f.phi_kill(M_atm, Rp, age)
    assert phi * 4 * np.pi * Rp**2 * age == pytest.approx(10 * M_atm)


@pytest.mark.xfail(strict=True, reason="phi_RMC passes t_sat positionally into Fxuv's t0 slot, so t_sat is ignored")
def test_phi_rmc_respects_saturation_time():
    t, F0, Rp = 1e9 / c.s2yr, 10.0, 2 * Re
    # With t_sat = 2 Gyr the flux at 1 Gyr is still saturated (F = F0)
    expected = 4e9 * np.sqrt(F0 / (5e5 * c.cgs2si_flux)) / (4 * np.pi * Rp**2)
    assert f.phi_RMC(t, F0, 2e9, Rp) == pytest.approx(expected)


class TestAnalyticSolutions:
    def test_subcritical_initial_value_and_growth(self):
        assert f.x2_subcrit(0.01, 10.0, 0.0) == pytest.approx(0.01)
        assert f.x2_subcrit(0.01, 10.0, 5.0) == pytest.approx(0.02)

    def test_supercritical_initial_value(self, capsys):
        assert f.x2_supercrit(0.01, 5 * Me, 2 * Re, 800.0, 1e18, 10.0, 0.0) == pytest.approx(0.01)

    def test_supercritical_enrichment_grows_with_time(self, capsys):
        args = (0.01, 5 * Me, 2 * Re, 800.0, 1e18, 10.0)
        assert f.x2_supercrit(*args, 5.0) > f.x2_supercrit(*args, 1.0) > 0.01
