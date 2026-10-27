"""
Integration tests for isofate.isofate_coupler.isocalc.

These run short simulations with Atmodeller switched off (n_atmodeller=0), so
they exercise the escape/fractionation loop only. isofate_coupler imports
atmodeller at module level, so the whole module is skipped if it isn't installed.
"""

import numpy as np
import pytest

pytest.importorskip("atmodeller")

from isofate import constants as c  # noqa: E402
from isofate.isofate_coupler import isocalc  # noqa: E402
from isofate.isofunks import R_core  # noqa: E402
from isofate.orbit_params import EqTemp, Insolation, SemiMajor  # noqa: E402

SPECIES = ["N_H", "N_He", "N_D", "N_O", "N_C", "N_N", "N_S"]
PARTICLE_MASS = dict(
    N_H=c.mu_H, N_He=c.mu_He, N_D=c.mu_D, N_O=c.mu_O, N_C=c.mu_C, N_N=c.mu_N, N_S=c.mu_S
)
N_STEPS = 1500


def protosolar_inventory(M_atm):
    """Initial atom numbers for a protosolar-composition envelope of mass M_atm (as in sim.py)."""
    He_frac = c.HetoH_protosolar_mass
    heavy = c.DtoH_solar_mass + c.OtoH_protosolar_mass + c.CtoH_protosolar_mass + c.NtoH_protosolar_mass + c.StoH_protosolar_mass
    return dict(
        N_H=(1 - heavy) * M_atm / (1 + He_frac) / c.mu_H,
        N_He=(He_frac / (1 + He_frac)) * M_atm / c.mu_He,
        N_D=c.DtoH_solar_mass * M_atm / (1 + He_frac) / c.mu_D,
        N_O=c.OtoH_protosolar_mass * M_atm / (1 + He_frac) / c.mu_O,
        N_C=c.CtoH_protosolar_mass * M_atm / (1 + He_frac) / c.mu_C,
        N_N=c.NtoH_protosolar_mass * M_atm / (1 + He_frac) / c.mu_N,
        N_S=c.StoH_protosolar_mass * M_atm / (1 + He_frac) / c.mu_S,
    )


def make_planet(f_atm=0.02, Mp=5 * c.Me, P_days=5.0, F0_scale=1e-3):
    """A sub-Neptune around a mid-M dwarf, mirroring the defaults in sim.py."""
    M_star, L = 0.18 * c.Ms, 0.0038 * c.Ls
    a = SemiMajor(M_star, P_days / c.s2day)
    Fp = Insolation(L, a)
    return dict(
        f_atm=f_atm, Mp=Mp, Mstar=M_star, F0=F0_scale * Fp, Fp=Fp, T=EqTemp(Fp), d=a,
        **protosolar_inventory(Mp * f_atm),
    )


def run(planet=None, **overrides):
    planet = dict(planet or make_planet())
    kwargs = dict(time=5e9, mechanism="XUV", rad_evol=True, n_steps=N_STEPS, n_atmodeller=0)
    kwargs.update(overrides)
    return isocalc(**planet, **kwargs)


@pytest.fixture(scope="module")
def planet():
    return make_planet()


@pytest.fixture(scope="module")
def sol(planet):
    return run(planet)


# ---------------------------------------------------------------- output structure


def test_returns_arrays_of_n_steps(sol):
    for key in ["time", "Rp", "Matm", "fatm", "phi", "phic", "Vpot", *SPECIES, "x1", "Phi_H", "Phi_D"]:
        assert sol[key].shape == (N_STEPS,), key


def test_is_deterministic(planet, sol):
    again = run(planet)
    for key in ["Matm", *SPECIES]:
        np.testing.assert_array_equal(sol[key], again[key])


def test_initial_state_matches_inputs(planet, sol):
    assert sol["Matm"][0] == pytest.approx(planet["Mp"] * planet["f_atm"])
    assert sol["fatm"][0] == pytest.approx(planet["f_atm"])
    for s in SPECIES:
        assert sol[s][0] == pytest.approx(planet[s])


def test_time_grid_starts_one_step_after_t0(sol):
    t0 = 1e6 / c.s2yr
    dt = (5e9 / c.s2yr - t0) / N_STEPS
    assert sol["time"][0] == pytest.approx(t0 + dt)
    assert np.all(np.diff(sol["time"]) > 0)


TIME_GRID_BUG = pytest.mark.xfail(
    strict=True,
    reason="t_a uses linspace(1, n+1, n): spacing is dt*n/(n-1) and the run ends one step after `time`",
)


@TIME_GRID_BUG
def test_time_grid_ends_at_requested_time(sol):
    assert sol["time"][-1] == pytest.approx(5e9 / c.s2yr)


@TIME_GRID_BUG
def test_time_grid_is_uniform(sol):
    t0 = 1e6 / c.s2yr
    dt = (5e9 / c.s2yr - t0) / N_STEPS
    np.testing.assert_allclose(np.diff(sol["time"]), dt, rtol=1e-9)


# ---------------------------------------------------------------- physical invariants


def test_abundances_are_non_negative_and_non_increasing(sol):
    for s in SPECIES:
        assert np.all(sol[s] >= 0), s
        assert np.all(np.diff(sol[s]) <= 0), s


def test_atmospheric_mass_decreases(sol):
    assert np.all(np.diff(sol["Matm"]) <= 0)
    assert sol["Matm"][-1] < sol["Matm"][0]


def test_envelope_fraction_tracks_mass(planet, sol):
    np.testing.assert_allclose(sol["fatm"], sol["Matm"] / planet["Mp"], rtol=1e-12)


def test_mass_lost_equals_mass_of_atoms_lost(sol):
    # M_atm is decremented by the summed mass of escaping atoms, so
    # M_atm - sum(N_i * m_i) is constant through the run.
    atom_mass = sum(sol[s] * PARTICLE_MASS[s] for s in SPECIES)
    residual = sol["Matm"] - atom_mass
    assert np.ptp(residual) <= 1e-12 * sol["Matm"][0]


def test_mole_fractions_sum_to_one(sol):
    total = sol["x1"] + sol["x2"] + sol["x3"] + sol["x4"] + sol["x5"] + sol["x6"] + sol["x7"]
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)


def test_radius_bounded_below_by_core(planet, sol):
    assert np.all(sol["Rp"] >= R_core(planet["Mp"]))


def test_escape_flux_is_positive_while_atmosphere_remains(sol):
    alive = sol["Matm"] > 0
    assert np.all(sol["phi"][alive] > 0)


# ---------------------------------------------------------------- fractionation (the point of IsoFATE)


def test_deuterium_is_enriched_relative_to_hydrogen(sol):
    DH = sol["N_D"] / sol["N_H"]
    assert DH[-1] > DH[0]


def test_helium_is_enriched_relative_to_hydrogen(sol):
    HeH = sol["N_He"] / sol["N_H"]
    assert HeH[-1] > HeH[0]


def test_heavier_species_lose_smaller_fraction(sol):
    retained = {s: sol[s][-1] / sol[s][0] for s in ["N_H", "N_D", "N_O"]}
    assert retained["N_H"] < retained["N_D"] < retained["N_O"]


# ---------------------------------------------------------------- options


def test_total_atmospheric_loss_leaves_bare_core():
    planet = make_planet(f_atm=1e-4, Mp=2 * c.Me, P_days=1.0, F0_scale=1e-2)
    sol = run(planet)
    assert sol["Matm"][-1] == 0
    stripped = sol["Matm"] == 0
    assert stripped.any()
    np.testing.assert_allclose(sol["Rp"][stripped], R_core(planet["Mp"]))
    for s in SPECIES:
        assert np.all(sol[s][stripped] == 0)
    assert np.all(sol["phi"][stripped] == 0)


def test_no_radius_evolution_fixes_radius_at_core(planet):
    sol = run(planet, rad_evol=False)
    np.testing.assert_allclose(sol["Rp"], R_core(planet["Mp"]))


def test_radiation_recombination_limit_reduces_flux(planet):
    with_rr = run(planet, RR=True)
    without_rr = run(planet, RR=False)
    assert with_rr["phi"][0] <= without_rr["phi"][0]
    assert with_rr["Matm"][-1] >= without_rr["Matm"][-1]


@pytest.mark.xfail(
    strict=True,
    reason="dynamic_phi uses get_binary_diffusion_coeff('He', 'D'), which returns b_H_He instead of b_He_D; "
    "with that lookup fixed the two paths agree exactly",
)
def test_dynamic_phi_matches_static_when_H_and_He_dominate(planet, sol):
    # With H and He the two most abundant species, the dynamic scheme picks the
    # same binary pair and diffusion coefficient as the static H/He scheme.
    dyn = run(planet, dynamic_phi=True)
    for key in ["Matm", *SPECIES]:
        np.testing.assert_allclose(dyn[key], sol[key], rtol=1e-10, err_msg=key)


def test_core_powered_mass_loss_runs(planet):
    sol = run(planet, mechanism="CPML")
    assert sol["Matm"][-1] < sol["Matm"][0]
    assert np.all(np.diff(sol["N_H"]) <= 0)


def test_save_molecules_returns_arrays(planet):
    sol = run(planet, save_molecules=True)
    for key in ["n_H2_a", "n_H2O_a", "n_N2_a", "n_SO2_a", "fO2_a"]:
        assert isinstance(sol[key], np.ndarray), key
        assert sol[key].shape == (N_STEPS,)
