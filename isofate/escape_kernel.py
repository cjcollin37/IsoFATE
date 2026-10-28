'''
Numba-compiled time step for isocalc
'''

from collections import namedtuple
from enum import IntEnum
from functools import partial

import numba as nb
import numpy as np

from isofate import isofunks
from isofate.constants import *

njit = partial(nb.njit, error_model='numpy')


class Mechanism(IntEnum):
    XUV = 0
    CPML = 1
    PHI_KILL = 2
    XUV_CPML = 3


class FluxModel(IntEnum):
    POWER_LAW = 0
    PHOENIX = 1
    JOHNSTONE = 2


Params = namedtuple('Params', [
    'Mp', 'Mstar', 'F0', 'Fp', 'T', 'd', 'delta_t', 't_total', 't0', 't_sat', 'beta', 'step_fn', 'F_final',
    't_pms', 'pms_factor', 'eps', 'rho_rcb', 'RR', 'thermal', 'rad_evol', 'dynamic_phi', 'mechanism',
    'flux_model', 'r_core', 'r_fixed', 'R_B', 'R_H'])

# species order: H, He, D, O, C, N, S
MU = np.array([mu_H, mu_He, mu_D, mu_O, mu_C, mu_N, mu_S]) # atomic masses [kg/atom]
MOLAR = np.array([M_H, M_He, M_D, M_O, M_C, M_N, M_S]) # molar masses [kg/mol]

# rows of the output array, in isocalc solution order
OUTPUTS = ('Rp', 'Ratm', 'Matm', 'Vpot', 'fatm', 'Mloss', 'phi', 'phic',
           'N_H', 'N_He', 'N_D', 'N_O', 'N_C', 'N_N', 'N_S',
           'N_H_int', 'N_He_int', 'N_D_int', 'N_O_int', 'N_C_int', 'N_N_int', 'N_S_int',
           'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
           'Phi_H', 'Phi_He', 'Phi_D', 'Phi_O', 'Phi_C', 'Phi_N', 'Phi_S')
RP, RATM, MATM, VPOT, FATM, MLOSS, PHI, PHIC = (OUTPUTS.index(name) for name in
                                                 ('Rp', 'Ratm', 'Matm', 'Vpot', 'fatm', 'Mloss', 'phi', 'phic'))
N, N_INT, X, FLUX = (OUTPUTS.index(name) for name in ('N_H', 'N_H_int', 'x1', 'Phi_H')) # first of 7 species rows

# state slots carried between steps
M_ATM, F_ATM, X_DH, AREA, MU_ATM = range(5)


_Fxuv = njit(isofunks.Fxuv)
_R_core = njit(isofunks.R_core)
_R_atm = njit(isofunks.R_atm)
_R_env = njit(isofunks.R_env)
_V_reduction = njit(isofunks.V_reduction)
_phi_kill = njit(isofunks.phi_kill)
_Phi_2 = njit(isofunks.Phi_2)


@njit
def _F_power_law(t, p):
    return _Fxuv(t, p.F0, p.t0, p.t_sat, p.beta, p.step_fn, p.F_final, p.t_pms, p.pms_factor)


@njit
def _phi_E(t, Vpot, F_track, n, p): # port of isofunks.phi_E
    if p.flux_model == FluxModel.POWER_LAW:
        return p.eps*_F_power_law(t, p)/(4*Vpot)
    elif p.flux_model == FluxModel.PHOENIX:
        return p.eps*F_track[n]/(4*Vpot)
    return p.eps*F_track[n]


@njit
def _phi_RR(Rp, t, p): # port of isofunks.phi_RR
    F = _F_power_law(t, p)
    g = G*p.Mp/Rp**2
    T = 1e4
    nu_0 = 4.835e15
    alpha_rec = 2.7e-13*(p.T/1e4)**(-0.9)/1e6
    c_s = np.sqrt(2*kb*T/mu_H)
    R_s = G*p.Mp/(2*c_s**2)
    p1 = 4*np.pi*R_s**2*c_s
    p2 = np.sqrt(F*mu_H**3*g/(h*nu_0*alpha_rec*2*kb*p.T))
    p3 = np.exp((Rp/R_s - 1)*G*p.Mp/Rp/c_s**2)
    return p1*p2*p3/(4*np.pi*Rp**2)


@njit
def _phiE_CP(area, mu, R_env, p): # port of isofunks.phiE_CP
    R_c = _R_core(p.Mp)
    V_pot = G*p.Mp/R_c
    gamma = 7/5
    R_B = (gamma - 1)*G*p.Mp*mu/(gamma*kb*p.T)
    kappa = 0.01
    L = 64*np.pi*sbc*p.T**4*R_B/(3*kappa*p.rho_rcb)
    phi_L = L/(V_pot*area)
    c_s = np.sqrt(kb*p.T/mu_H)
    R_rcb = R_c + R_env
    phi_B = c_s*p.rho_rcb*np.exp(-G*p.Mp/(c_s**2*R_rcb))
    return min(phi_L, phi_B)


@njit
def _Phi_1(phi, b, H1, H2, m1, m2, x1, x2, mu): # port of isofunks.Phi_1(..., output = 1)
    phi_c = b*x1*(m2 - m1)/H1
    phi_d2 = b/H2
    if mu == 0:
        return 0.0, phi_c
    if phi < phi_c:
        return phi/m1, phi_c
    return (x1*phi + x1*x2*(m2 - m1)*phi_d2)/mu, phi_c


@njit
def _Phi_minor(Phi_1, Phi_2, H_1, H_2, H_m, N_1, N_2, N_m, N_tot, b_12, b_1m, b_2m):
    # port of isofunks.Phi_minor_species; with species 1, 2 = H, He it is also Phi_D_Z90 ... Phi_S_Z90
    if N_1 == 0:
        return 0.0
    alpha_2 = b_1m/b_12
    alpha_3 = b_1m/b_2m
    Phi_DL_m = b_1m*(1/H_m - 1/H_1)
    Phi_DL_2 = b_12*(1/H_2 - 1/H_1)
    num = Phi_1 - Phi_DL_m + alpha_2*Phi_DL_2*(N_2/N_tot) + alpha_3*Phi_2
    denom = 1 + alpha_3*(N_2/N_1)
    return max(0.0, (N_m/N_1)*num/denom)


@njit(inline='always')
def fluxes(n, t_a, y, y_int, flux, state, out, p, F_track, b_table):
    '''
    Computes the escape fluxes for step n and records the step in out.
    Returns False, after filling the remaining steps, once the atmosphere is gone.
    '''
    M_atm = state[M_ATM]
    f_atm = state[F_ATM]
    radius_core = p.r_core if n == 0 else p.r_fixed
    N_tot = 0.0
    for i in range(7):
        N_tot += y[i]
    if M_atm <= 0 or N_tot <= 0:
        K = max(_V_reduction(p.Mp, p.Mstar, p.d, radius_core), 0.01)
        for m in range(n, out.shape[1]):
            out[RP, m] = radius_core
            out[VPOT, m] = K*G*p.Mp/radius_core
            for i in range(7):
                out[X + i, m] = out[X + i, n - 1]
        return False

    mu = 0.0
    for i in range(7):
        mu += y[i]*MU[i]
    mu = mu/N_tot

    t = t_a[n]
    if p.rad_evol:
        radius_env = _R_env(p.Mp, f_atm, p.Fp, t, p.thermal)
        radius_atm = _R_atm(p.T, p.Mp, radius_core, radius_env, mu)
        radius_p = min(radius_core + radius_atm + radius_env, p.R_B, p.R_H)
    else:
        radius_env = 0.0
        radius_p = radius_core

    K = max(_V_reduction(p.Mp, p.Mstar, p.d, radius_p), 0.01)
    Vpot = K*G*p.Mp/radius_p
    A = 4*np.pi*radius_p**2

    if p.mechanism == Mechanism.CPML:
        phi = _phiE_CP(A, mu, radius_env, p)
    elif p.mechanism == Mechanism.PHI_KILL:
        phi = _phi_kill(p.Mp*f_atm, radius_p, p.t_total - t)
    else:
        phi = _phi_E(t, Vpot, F_track, n, p)
        if p.RR:
            phi = min(_phi_RR(radius_p, t, p), phi)
        if p.mechanism == Mechanism.XUV_CPML:
            phi = phi + _phiE_CP(A, mu, radius_env, p)

    g = G*p.Mp/radius_p**2

    # binary pair: H/He, or the two most abundant species ordered light, heavy
    if p.dynamic_phi:
        most = 0
        for i in range(1, 7):
            if y[i] > y[most]:
                most = i
        second = 1 if most == 0 else 0
        for i in range(7):
            if i != most and y[i] > y[second]:
                second = i
        if MU[most] <= MU[second]:
            l, hv = most, second
        else:
            l, hv = second, most
    else:
        l, hv = 0, 1
    if y[l] + y[hv] == 0:
        X1 = 0.0
        X2 = 0.0
    else:
        X1 = y[l]/(y[l] + y[hv])
        X2 = y[hv]/(y[l] + y[hv])
    MU_pair = X1*MU[l] + X2*MU[hv]
    b = b_table[l, hv]
    H_l = R_gas*p.T/(MOLAR[l]*g) # scale heights [m]
    H_hv = R_gas*p.T/(MOLAR[hv]*g)
    flux[l], phi_c = _Phi_1(phi, b, H_l, H_hv, MU[l], MU[hv], X1, X2, MU_pair)
    flux[hv] = _Phi_2(phi, b, H_l, H_hv, MU[l], MU[hv], X1, X2, MU_pair)
    for i in range(7):
        if i != l and i != hv:
            flux[i] = _Phi_minor(flux[l], flux[hv], H_l, H_hv, R_gas*p.T/(MOLAR[i]*g), y[l], y[hv], y[i], N_tot,
                                 b, b_table[l, i], b_table[hv, i])

    out[MATM, n] = M_atm
    out[FATM, n] = f_atm
    out[RATM, n] = radius_env
    out[RP, n] = radius_p
    out[VPOT, n] = Vpot
    out[PHI, n] = phi
    out[PHIC, n] = phi_c
    out[MLOSS, n] = phi*A*p.delta_t
    for i in range(7):
        out[N + i, n] = y[i]
        out[N_INT + i, n] = y_int[i]
        out[X + i, n] = y[i]/N_tot
        out[FLUX + i, n] = flux[i]
    state[AREA] = A
    state[MU_ATM] = mu
    return True


@njit(inline='always')
def advance(y, y_int, flux, state, p):
    '''
    Removes the escaped atoms of the step last computed by fluxes.
    '''
    if y[0] + y[2] + y_int[0] + y_int[2] != 0:
        state[X_DH] = (y[2] + y_int[2])/(y[0] + y[2] + y_int[0] + y_int[2])
    lost_mass = 0.0
    for i in range(7):
        loss = flux[i]*state[AREA]*p.delta_t
        lost_mass += loss*MU[i]
        y[i] = max(y[i] - loss, 0.0)
    state[M_ATM] -= lost_mass
    state[F_ATM] = state[M_ATM]/p.Mp


@njit
def run(n0, n1, t_a, y, y_int, flux, state, out, p, F_track, b_table):
    '''
    Runs steps n0 to n1 - 1. Returns the step at which the atmosphere is gone, or -1.
    '''
    for n in range(n0, n1):
        if not fluxes(n, t_a, y, y_int, flux, state, out, p, F_track, b_table):
            return n
        advance(y, y_int, flux, state, p)
    return -1
