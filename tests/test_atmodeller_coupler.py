'''
Checks the compiled Atmodeller solve against atmodeller's own, and the numba surface state against isofunks.py.
'''

import numpy as np
import pytest

from isofate import atmodeller_coupler as ac
from isofate import isofunks as f
from isofate.constants import *


def mass_constraints(scale):
    return {'H': 1e20*scale, 'O': 1e19*scale, 'C': 1e18*scale, 'He': 3e19*scale, 'N': 1e17*scale, 'S': 1e9}


def atmodeller_asdict(planet, constraints):
    from atmodeller import InteriorAtmosphere, Planet

    interior_atmosphere = InteriorAtmosphere(ac.isofate_species())
    interior_atmosphere.solve(planet = Planet(**planet), mass_constraints = constraints)
    return interior_atmosphere.output.asdict()


def assert_matches(sol, full):
    for name, entries in sol.items():
        for key, value in entries.items():
            np.testing.assert_allclose(value, full[name][key], rtol = 1e-9, err_msg = f'{name}.{key}')


@pytest.mark.parametrize('temperature, melt_fraction, Mp', [(2500, 0.5, 5*Me), (6000, 1.0, 15*Me), (900, 0.0, 1*Me)])
def test_atmodeller_solver_matches_atmodeller(temperature, melt_fraction, Mp, monkeypatch):
    solver = ac.AtmodellerSolver()
    monkeypatch.setattr(solver._system.interior_atmosphere, 'solve', None) # no fallback
    # a cold start, then warm starts from it as the state drifts
    for step, scale in enumerate((1.0, 0.97, 0.9)):
        planet = dict(surface_temperature = temperature*(1 - 0.01*step), mantle_melt_fraction = melt_fraction,
                      planet_mass = Mp, surface_radius = f.R_core(Mp))
        constraints = mass_constraints(scale)
        sol = solver.solve(*planet.values(), constraints)
        assert_matches(sol, atmodeller_asdict(planet, constraints))


def test_atmodeller_solver_falls_back_to_atmodeller():
    solver = ac.AtmodellerSolver()
    solver._initial = np.full(solver._initial.shape, np.nan)
    Mp = 5*Me
    planet = dict(surface_temperature = 2500, mantle_melt_fraction = 0.5, planet_mass = Mp,
                  surface_radius = f.R_core(Mp))
    sol = solver.solve(*planet.values(), mass_constraints(1.0))
    assert np.all(np.isfinite(solver._initial))
    assert_matches(sol, atmodeller_asdict(planet, mass_constraints(1.0)))


def test_atmodeller_coupler_conserves_atoms():
    # isocalc feeds the atom numbers of each call into the next, so any loss or gain compounds with every call
    Mp = 5*Me
    elements = ('H', 'He', 'O', 'C', 'N', 'S')
    numbers = np.array([1e20/mu_H, 3e19/mu_He, 1e19/mu_O, 1e18/mu_C, 1e17/mu_N, 1e10/mu_S])
    atm, dissolved = numbers, np.zeros(len(elements))
    solver = ac.AtmodellerSolver()
    for _ in range(3):
        results = ac.AtmodellerCoupler(1000, Mp, 2*f.R_core(Mp), 2.3*mu_H, 0.5, False, *atm, *dissolved, solver)[0]
        atm = np.array([results[f'N_{element}_atm'] for element in elements])
        dissolved = np.array([results[f'N_{element}_int'] for element in elements])
        np.testing.assert_allclose(atm + dissolved, numbers, rtol = 1e-12)


def test_surface_temperature_matches_isofunks():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        Mp = rng.uniform(1, 20)*Me
        Teq, mu = rng.uniform(200, 2000), rng.uniform(1, 44)*mu_H
        Rp = f.R_core(Mp)*rng.uniform(1, 40)
        T_surface, P_surface = f.make_atmosphere_descent(Teq, mu, Rp, Mp, 7/5, 2)
        Matm, T, p = ac._atmosphere_descent(Teq, mu, Rp, Mp, f.R_core(Mp), 7/5)
        np.testing.assert_allclose((T[0], p[0]), (T_surface, P_surface), rtol = 1e-13)
        np.testing.assert_allclose(Matm, f.make_atmosphere_descent(Teq, mu, Rp, Mp, 7/5, 1), rtol = 1e-13)


def test_melt_fraction_matches_isofunks():
    rng = np.random.default_rng(0)
    masses, temperatures = np.meshgrid(ac.MELT_MASS_GRID, ac.MELT_TEMP_GRID[::37])
    points = np.concatenate((
        np.stack((masses.ravel(), temperatures.ravel()), axis = 1),
        np.stack((rng.uniform(1, 21, 5000), rng.uniform(10, 16000, 5000)), axis = 1),
        [[1, 10], [21, 16000], [np.nextafter(21, 0), np.nextafter(16000, 0)], [1, np.nextafter(10, 11)]]))
    for m, T in points:
        assert ac.melt_fraction_from_grid(m*Me, T) == pytest.approx(float(f.MeltFraction(m*Me, T)), rel = 1e-14)
    for m, T in [(0.99, 1000), (22, 1000), (5, 9.9), (5, 16001), (5, np.nan)]:
        with pytest.raises(ValueError):
            f.MeltFraction(m*Me, T)
        with pytest.raises(ValueError):
            ac.melt_fraction_from_grid(m*Me, T)
