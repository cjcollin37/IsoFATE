'''
Cheap analytic stand-in for isofate_coupler.isocalc (model.backend = "toy"), for testing the MCMC plumbing.
Not physics.

Same signature and output keys as isocalc for the given n_steps. f_atm decays as exp(-t/tau), with tau set by
F0, eps and Mp, and the atmosphere is lost below 1e-7. Rp grows with f_atm, He stops escaping when the flux is
subcritical, and the final Atmodeller dict is NaN once the atmosphere is gone, as in isocalc. Rp carries a small
resolution-dependent bias and each call sleeps 1 us per Atmodeller call, so the resolution study has something
to find. Imports neither jax nor numba.
'''

import time as _time

import numpy as np

from isofate.constants import G, Me, Re, avogadro, mu_H, s2yr

# isofate.escape_kernel.OUTPUTS (not imported: escape_kernel imports numba)
OUTPUTS = ('Rp', 'Ratm', 'Matm', 'Vpot', 'fatm', 'Mloss', 'phi', 'phic',
           'N_H', 'N_He', 'N_D', 'N_O', 'N_C', 'N_N', 'N_S',
           'N_H_int', 'N_He_int', 'N_D_int', 'N_O_int', 'N_C_int', 'N_N_int', 'N_S_int',
           'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
           'Phi_H', 'Phi_He', 'Phi_D', 'Phi_O', 'Phi_C', 'Phi_N', 'Phi_S')
SPECIES = ('H', 'He', 'D', 'O', 'C', 'N', 'S')
GASES = ('H2', 'H2O', 'O2', 'CO2', 'CO', 'CH4', 'N2', 'S2', 'H2O4S', 'SO2') # isocalc's n_*_a order
F_LOST = 1e-7 # atmosphere lost below this mass fraction
RETENTION = {'H': 1.0, 'He': 0.5, 'D': 0.8, 'O': 0.3, 'C': 0.3, 'N': 0.3, 'S': 0.3} # N_X ~ (f/f0)**RETENTION
MANTLE = 0.05 # dissolved fraction of the initial inventory, reached as the atmosphere goes


def _molecules(N_H, N_D, N_O, N_C, N_N, N_S):
    '''
    Toy speciation [mol] conserving C, N, S and H+D (lumped, as Atmodeller sees it), and O while O > C.
    '''
    H, O, C, N, S = (N_H + N_D)/avogadro, N_O/avogadro, N_C/avogadro, N_N/avogadro, N_S/avogadro
    m = np.minimum(C, O)
    CO, CO2 = 0.5*m, 0.01*m
    CH4 = C - CO - CO2
    SO2, H2SO4 = 0.05*S, 1e-6*S
    S2 = (S - SO2 - H2SO4)/2
    O_left = np.maximum(O - CO - 2*CO2 - 2*SO2 - 4*H2SO4, 0)
    O2 = 1e-9*O_left
    H2O = O_left - 2*O2
    H2 = np.maximum(H/2 - H2O - 2*CH4 - H2SO4, 0)
    return {'H2': H2, 'H2O': H2O, 'O2': O2, 'CO2': CO2, 'CO': CO, 'CH4': CH4, 'N2': N/2, 'S2': S2,
            'H2O4S': H2SO4, 'SO2': SO2}


def _xuv(t_yr, F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor):
    '''
    Incident XUV flux [W/m2]: saturated at F0 until t_sat, then a power law or F_final; pms_factor*F0 before
    t_pms.
    '''
    F = np.where(t_yr < t_sat, F0, F_final if step_fn else F0*(np.maximum(t_yr, t0)/t_sat)**beta)
    if t_pms > t0:
        F = np.where(t_yr < t_pms, pms_factor*F0, F)
    return F


def isocalc(f_atm, Mp, Mstar, F0, Fp, T, d, time = 5e9, mechanism = 'XUV', rad_evol = True,
            N_H = 0, N_He = 0, N_D = 0, N_O = 0, N_C = 0, N_N = 0, N_S = 0, melt_fraction_override = False,
            mu = 0.00235/avogadro, eps = 0.15, activity = 'medium', flux_model = 'power law', stellar_type = 'M1',
            Rp_override = False, t_sat = 5e8, step_fn = False, F_final = 0, t_pms = 0, pms_factor = 1e2,
            n_steps = int(1e5), t0 = 1e6, rho_rcb = 1.0, RR = True, thermal = True,
            beta = -1.23, n_atmodeller = int(1e2), save_molecules = False, mantle_iron_dict = False,
            dynamic_phi = False):
    '''
    Toy isocalc: same arguments and output keys (time [s], the escape kernel rows, T_surf_analytic,
    T_surf_atmod, atmodeller_final when n_atmodeller > 0, n_*_a when save_molecules).
    '''
    if n_atmodeller < 0 or n_atmodeller != int(n_atmodeller):
        raise ValueError(f'n_atmodeller must be a non-negative whole number of timesteps, got {n_atmodeller}')
    if mechanism not in ('XUV', 'CPML', 'XUV+CPML', 'phi kill'):
        raise ValueError(f"unknown mechanism '{mechanism}'")
    n, n_atm = int(n_steps), int(n_atmodeller)
    if n_atm != 0 and melt_fraction_override is False and not 1 <= Mp/Me <= 21:
        raise ValueError(f'melt fraction grid does not cover Mp = {Mp/Me} Me') # as atmodeller_coupler

    t0_s = t0/s2yr # [s]
    delta_t = (time/s2yr - t0_s)/n # [s]
    t_a = delta_t*np.linspace(1, n + 1, n) + t0_s # [s], as isocalc
    t_yr = t_a*s2yr
    tau = 2e9*(Mp/(5.6*Me))**2*(0.15/eps)*np.sqrt(0.588/max(F0, 1e-30)) # e-folding time [yr]
    g = np.exp(-(t_yr - t0)/tau) # f_atm(t)/f_atm(t0)
    alive = f_atm*g >= F_LOST
    n_end = n if alive.all() else int(np.argmin(alive)) # first step without an atmosphere
    live = slice(0, n_end)
    gl = g[:n_end]

    out = {name: np.zeros(n) for name in OUTPUTS}
    R_core = Re*(Mp/Me)**0.25 # [m]
    R_fixed = float(Rp_override) if not rad_evol and Rp_override is not False else R_core # as isocalc's r_fixed
    fatm = f_atm*gl
    bias = 1 + 0.05*n_atm/n + 1e3/n # resolution-dependent error
    Rp = R_core*(1 + 4*np.sqrt(fatm))*bias if rad_evol else np.full(n_end, R_fixed)
    out['Rp'][:] = R_core if rad_evol else R_fixed
    out['Rp'][live] = Rp
    out['Ratm'][live] = Rp - R_core
    out['fatm'][live] = fatm
    out['Matm'][live] = fatm*Mp
    out['Vpot'][:] = G*Mp/out['Rp']
    out['Vpot'][live] = G*Mp/Rp

    N0 = dict(zip(SPECIES, (N_H, N_He, N_D, N_O, N_C, N_N, N_S)))
    for X in SPECIES:
        out[f'N_{X}'][live] = N0[X]*gl**RETENTION[X]
        if n_atm > 0:
            out[f'N_{X}_int'][live] = MANTLE*N0[X]*(1 - gl)
    N_tot = sum(out[f'N_{X}'][live] for X in SPECIES)
    for i, X in enumerate(SPECIES, 1):
        out[f'x{i}'][live] = out[f'N_{X}'][live]/np.where(N_tot > 0, N_tot, 1)
        if n_end < n:
            out[f'x{i}'][n_end:] = out[f'x{i}'][n_end - 1] if n_end > 0 else 0 # frozen, as isocalc

    F = _xuv(t_yr[live], F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor)
    phi = eps*F/(4*out['Vpot'][live]) # energy-limited mass flux [kg/m2/s]
    phic = np.full(n_end, 5e-11*Mp/(5.6*Me)) # critical flux for He escape [kg/m2/s]
    out['phi'][live], out['phic'][live] = phi, phic
    out['Mloss'][live] = phi*4*np.pi*Rp**2*delta_t
    supercritical = phi > phic
    N_Hl = np.where(out['N_H'][live] > 0, out['N_H'][live], 1)
    Phi_H = phi*out['x1'][live]/mu_H # [atoms/m2/s]
    out['Phi_H'][live] = Phi_H
    out['Phi_He'][live] = np.where(supercritical, Phi_H*out['N_He'][live]/N_Hl*(1 - phic/np.maximum(phi, phic)), 0)
    out['Phi_D'][live] = 0.8*Phi_H*out['N_D'][live]/N_Hl
    for X in ('O', 'C', 'N', 'S'):
        out[f'Phi_{X}'][live] = np.where(supercritical, 0.1*Phi_H*out[f'N_{X}'][live]/N_Hl, 0)

    sol = {'time': t_a}
    sol.update(out)
    T_surf_analytic, T_surf_atmod = np.zeros(n), np.zeros(n)
    molecules = {}
    if n_atm > 0:
        calls = np.arange(n_end)//n_atm*n_atm # values hold from the most recent Atmodeller call
        T_s = T*(1 + 500*np.sqrt(f_atm*g[calls]))
        T_surf_analytic[live], T_surf_atmod[live] = T_s, np.minimum(T_s, 6000.0)
        if save_molecules:
            molecules = _molecules(*(out[f'N_{X}'][calls] for X in ('H', 'D', 'O', 'C', 'N', 'S')))
    sol['T_surf_analytic'], sol['T_surf_atmod'] = T_surf_analytic, T_surf_atmod

    if save_molecules:
        for suffix, scale in (('', 1.0), ('_int', MANTLE)):
            for gas in GASES:
                sol[f'n_{gas}_a{suffix}'] = np.zeros(n)
                if gas in molecules:
                    sol[f'n_{gas}_a{suffix}'][live] = scale*molecules[gas]
        sol['fO2_a'] = np.zeros(n)
        if n_atm > 0:
            sol['fO2_a'][live] = 1e-8

    if n_atm > 0:
        final = {}
        if n_end == n:
            gas = _molecules(*(out[f'N_{X}'][-1] for X in ('H', 'D', 'O', 'C', 'N', 'S')))
            for name in ('H2O', 'H2', 'O2', 'CO', 'CO2', 'CH4', 'N2', 'S2', 'H2O4S'):
                final[f'{name}_atm'] = float(gas[name])
                final[f'{name}_mantle'] = float(MANTLE*gas[name])
            final['SO2_atm'] = float(gas['SO2'])
            final['He_mantle'] = float(out['N_He_int'][-1]/avogadro)
            final['O2_fugacity'] = 1e-8
            final['log10dIW_1_bar'] = -2.0
        else: # exhausted: 21 NaN entries and no log10dIW_1_bar, as isocalc
            for name in ('H2O', 'H2', 'O2', 'CO', 'CO2', 'CH4'):
                final[f'{name}_atm'] = final[f'{name}_mantle'] = np.nan
            final['He_mantle'] = np.nan
            for name in ('N2', 'S2', 'H2O4S'):
                final[f'{name}_atm'] = final[f'{name}_mantle'] = np.nan
            final['SO2_atm'] = final['O2_fugacity'] = np.nan
        sol['atmodeller_final'] = final
        _time.sleep(1e-6*n/n_atm) # cost of the Atmodeller calls
    return sol
