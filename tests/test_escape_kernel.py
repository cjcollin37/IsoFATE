'''
Checks that the numba ports in escape_kernel.py match isofunks.py.
'''

import numpy as np

from isofate import escape_kernel as ek
from isofate import isofunks as f
from isofate.constants import *

SPECIES = ['H', 'He', 'D', 'O', 'C', 'N', 'S']
RTOL = 1e-10


def random_cases(n = 200):
    rng = np.random.default_rng(0)
    for _ in range(n):
        Mp, T = rng.uniform(1, 15)*Me, rng.uniform(200, 1500)
        F0, t_sat, t_pms = rng.uniform(0.1, 100), rng.uniform(1e8, 1e9), rng.choice([0.0, 1e8])
        p = ek.Params(Mp = Mp, Mstar = rng.uniform(0.1, 1)*Ms, F0 = F0, Fp = 1e3, T = T,
                      d = rng.uniform(0.01, 0.2)*au2m, delta_t = 1.0, t_total = 1.0, t0 = 1e6, t_sat = t_sat,
                      beta = -1.23, step_fn = bool(rng.integers(2)), F_final = 0.17, t_pms = t_pms, pms_factor = 1e2,
                      eps = 0.15, rho_rcb = 1.0, RR = True, thermal = True, rad_evol = True, dynamic_phi = False,
                      mechanism = ek.Mechanism.XUV, flux_model = ek.FluxModel.POWER_LAW, r_core = f.R_core(Mp),
                      r_fixed = f.R_core(Mp), R_B = 1e30, R_H = 1e30)
        N = rng.uniform(0, 1, 7)*10.0**rng.uniform(30, 45, 7)
        yield rng, p, rng.uniform(1e6, 5e9)/s2yr, f.R_core(Mp)*rng.uniform(1, 3), N


def test_escape_rates_match_isofunks():
    for rng, p, t, Rp, N in random_cases():
        Vpot = G*p.Mp/Rp
        np.testing.assert_allclose(
            ek._phi_E(t, Vpot, np.empty(0), 0, p),
            f.phi_E(t, p.eps, Vpot, p.d, p.F0, p.t0, p.t_sat, p.beta, 'medium', 'power law', 'M1', p.step_fn,
                    p.F_final, p.t_pms, p.pms_factor), rtol = RTOL)
        np.testing.assert_allclose(
            ek._phi_RR(Rp, t, p),
            f.phi_RR(Rp, p.Mp, p.T, t, p.F0, p.t0, p.t_sat, p.beta, p.step_fn, p.F_final, p.t_pms, p.pms_factor),
            rtol = RTOL)
        area, mu, R_env = 4*np.pi*Rp**2, rng.uniform(1, 3)*mu_H, rng.uniform(0, 1e7)
        np.testing.assert_allclose(ek._phiE_CP(area, mu, R_env, p),
                                   f.phiE_CP(p.T, p.Mp, p.rho_rcb, p.eps, Vpot, area, mu, R_env), rtol = RTOL)


def test_species_fluxes_match_isofunks():
    for rng, p, t, Rp, N in random_cases():
        g = G*p.Mp/Rp**2
        H = [R_gas*p.T/(M*g) for M in (M_H, M_He, M_D, M_O, M_C, M_N, M_S)]
        N_tot = sum(N)
        phi = 10.0**rng.uniform(-12, -6)

        # H/He binary with the Zahnle et al 1990 minor species, as in isocalc's static mode
        X1, X2 = N[0]/(N[0] + N[1]), N[1]/(N[0] + N[1])
        MU = X1*mu_H + X2*mu_He
        b = b_H_He(p.T)
        Phi_H, phi_c = ek._Phi_1(phi, b, H[0], H[1], mu_H, mu_He, X1, X2, MU)
        np.testing.assert_allclose((Phi_H, phi_c), f.Phi_1(phi, b, H[0], H[1], mu_H, mu_He, X1, X2, MU, output = 1),
                                   rtol = RTOL)
        Phi_He = ek._Phi_2(phi, b, H[0], H[1], mu_H, mu_He, X1, X2, MU)
        np.testing.assert_allclose(Phi_He, f.Phi_2(phi, b, H[0], H[1], mu_H, mu_He, X1, X2, MU), rtol = RTOL)
        Z90 = {2: (f.Phi_D_Z90, b_H_D, b_He_D), 3: (f.Phi_O_Z90, b_H_O, b_He_O), 4: (f.Phi_C_Z90, b_H_C, b_He_C),
               5: (f.Phi_N_Z90, b_H_N, b_He_N), 6: (f.Phi_S_Z90, b_H_S, b_He_S)}
        for i, (Phi_Z90, b_H_X, b_He_X) in Z90.items():
            np.testing.assert_allclose(
                ek._Phi_minor(Phi_H, Phi_He, H[0], H[1], H[i], N[0], N[1], N[i], N_tot, b, b_H_X(p.T), b_He_X(p.T)),
                Phi_Z90(Phi_H, Phi_He, H[0], H[i], H[1], *N, p.T), rtol = RTOL)

        # any light/heavy pair, as in isocalc's dynamic_phi mode
        l, hv = sorted(rng.choice(7, 2, replace = False), key = lambda i: ek.MU[i])
        b_12 = f.get_binary_diffusion_coeff(SPECIES[l], SPECIES[hv], p.T)
        for i in set(range(7)) - {l, hv}:
            np.testing.assert_allclose(
                ek._Phi_minor(Phi_H, Phi_He, H[l], H[hv], H[i], N[l], N[hv], N[i], N_tot, b_12,
                              f.get_binary_diffusion_coeff(SPECIES[l], SPECIES[i], p.T),
                              f.get_binary_diffusion_coeff(SPECIES[hv], SPECIES[i], p.T)),
                f.Phi_minor_species(Phi_H, Phi_He, H[l], H[hv], H[i], list(N), p.T, i, l, hv), rtol = RTOL)

