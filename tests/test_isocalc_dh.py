'''
Checks that isocalc conserves D through its Atmodeller calls and that its D/H converges in n_atmodeller.

Scenario: sim.py's LHS 1140 b at f_atm = 8.5e-4, short runs. Splitting Atmodeller's lumped H with the D/(H + D)
of one step earlier turns D into H and undoes one step of fractionation per call (all of it for
n_atmodeller = 1).
'''

import numpy as np
import pytest

from isofate.constants import *
from isofate.isofate_coupler import isocalc
from isofate.orbit_params import EqTemp, Insolation, Luminosity, SemiMajor

TIME, T0 = 5e9, 1e6 # [yr]


def lhs1140b(n_steps, n_atmodeller, f_atm = 8.5e-4):
    M_star = 0.18*Ms
    L = Luminosity(0.22*Rs, 3096)
    Mp = 5.6*Me
    d = SemiMajor(M_star, 24.74/s2day)
    Fp = Insolation(L, d)
    M_atm = Mp*f_atm
    He = HetoH_protosolar_mass
    OtoH_mass = OtoH_protosolar*mu_O/mu_H
    N = dict(N_He = He/(1 + He)*M_atm/mu_He,
             N_H = (1 - DtoH_solar_mass - OtoH_mass - CtoH_protosolar_mass - StoH_protosolar_mass
                    - NtoH_protosolar_mass)*M_atm/(1 + He)/mu_H,
             N_D = DtoH_solar_mass*M_atm/(1 + He)/mu_D, N_O = OtoH_mass*M_atm/(1 + He)/mu_O,
             N_C = CtoH_protosolar_mass*M_atm/(1 + He)/mu_C, N_N = NtoH_protosolar_mass*M_atm/(1 + He)/mu_N,
             N_S = StoH_protosolar_mass*M_atm/(1 + He)/mu_S)
    sol = isocalc(f_atm, Mp, M_star, Fp*1e-3, Fp, EqTemp(Fp), d, TIME, 'XUV', True, **N, eps = 0.15,
                  t_sat = (5.9 - 15.4*0.18)*1e9, step_fn = True, F_final = 0.17, n_steps = n_steps, t0 = T0,
                  n_atmodeller = n_atmodeller, dynamic_phi = True)
    delta_t = (TIME - T0)/s2yr/n_steps # isocalc's timestep [s] (s2yr converts s to yr)
    return sol, delta_t


def enhancement(sol):
    return sol['N_D'][-1]/sol['N_H'][-1]/(sol['N_D'][0]/sol['N_H'][0])


@pytest.mark.parametrize('n_atmodeller', [1, 7])
def test_D_is_conserved_through_atmodeller_calls(n_atmodeller):
    sol, delta_t = lhs1140b(2000, n_atmodeller)
    assert np.all(sol['fatm'] > 0)
    D = sol['N_D'] + sol['N_D_int']
    escaped = sol['Phi_D']*4*np.pi*sol['Rp']**2*delta_t
    # what escape does not explain, per step; nonzero only at call steps if a call changes the D inventory
    residual = (D[1:] - D[:-1] + escaped[:-1])/D[:-1]
    np.testing.assert_allclose(residual, 0, atol = 1e-10)


def test_DH_converges_in_n_atmodeller():
    e1 = enhancement(lhs1140b(2000, 1)[0])
    e10 = enhancement(lhs1140b(2000, 10)[0])
    e0 = enhancement(lhs1140b(2000, 0)[0]) # no Atmodeller: escape alone
    assert e0 > 2 and e1 > 2 # fractionation happens with and without the interior
    assert abs(e10/e1 - 1) < 0.02
