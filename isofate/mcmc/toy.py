'''
Toy backend: a cheap analytic stand-in for isofate_coupler.isocalc (model.backend = "toy"), for testing the MCMC
plumbing. Not physics.

It has the signature and output keys of isocalc for the given n_steps, and imports neither jax nor numba, so the
tests run fast in one process or in a small pool. f_atm decays as exp(-t/tau), with tau set by F0, eps and Mp,
and the atmosphere is lost below f_atm = 1e-7 (F_LOST). Rp grows with f_atm (with radius evolution) and drops to
the bare core when the atmosphere is gone. He stops escaping when the flux is subcritical, and the final
Atmodeller dict is NaN once the atmosphere is gone, as in isocalc. Rp carries a small resolution-dependent bias,
and each call sleeps 1 us per Atmodeller call, so the resolution study has an error and a cost to find.
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
SPECIES = ('H', 'He', 'D', 'O', 'C', 'N', 'S') # escape kernel order
GASES = ('H2', 'H2O', 'O2', 'CO2', 'CO', 'CH4', 'N2', 'S2', 'H2O4S', 'SO2') # isocalc's n_*_a order
F_LOST = 1e-7 # atmosphere lost below this mass fraction
# a toy fractionation: H leaves with the atmosphere, the heavier species are held back (exponents below 1)
RETENTION = {'H': 1.0, 'He': 0.5, 'D': 0.8, 'O': 0.3, 'C': 0.3, 'N': 0.3, 'S': 0.3} # N_X ~ (f/f0)**RETENTION
MANTLE = 0.05 # dissolved fraction of the initial inventory, reached as the atmosphere goes


def _molecules(N_H, N_D, N_O, N_C, N_N, N_S):
    '''
    Toy speciation [mol] conserving C, N, S and H+D (lumped, as Atmodeller sees it), and O while O > C.
    Takes atom numbers [atoms] (scalars or arrays) and returns {gas: amount [mol]} under the names of GASES.
    '''
    H, O, C, N, S = (N_H + N_D)/avogadro, N_O/avogadro, N_C/avogadro, N_N/avogadro, N_S/avogadro # [mol]
    m = np.minimum(C, O) # C that finds O to bind
    CO, CO2 = 0.5*m, 0.01*m
    CH4 = C - CO - CO2 # the rest of the C
    SO2, H2SO4 = 0.05*S, 1e-6*S
    S2 = (S - SO2 - H2SO4)/2 # the rest of the S
    O_left = np.maximum(O - CO - 2*CO2 - 2*SO2 - 4*H2SO4, 0) # O not bound to C or S
    O2 = 1e-9*O_left # a trace
    H2O = O_left - 2*O2
    H2 = np.maximum(H/2 - H2O - 2*CH4 - H2SO4, 0) # the H left over
    return {'H2': H2, 'H2O': H2O, 'O2': O2, 'CO2': CO2, 'CO': CO, 'CH4': CH4, 'N2': N/2, 'S2': S2,
            'H2O4S': H2SO4, 'SO2': SO2}


def _xuv(t_yr, F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor):
    '''
    Incident XUV flux [W/m2], a simplified isofunks.Fxuv (power law adapted from Ribas et al 2005)
    Saturated at F0 until t_sat, then F0*(t/t_sat)**beta, or F_final with step_fn. Before t_pms (when t_pms > t0)
    a constant pms_factor*F0, where isofunks.Fxuv decays as a power law from pms_factor*F0 at t0 to F0 at t_pms.
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
    Runs the toy model, with the arguments and output keys of isofate_coupler.isocalc
    Raises the ValueErrors isocalc raises for an n_atmodeller that is not a non-negative whole number, an unknown
    mechanism, and (with Atmodeller and no melt fraction override) a planet mass outside the melt grid, 1-21
    M_earth.

    Inputs: as isofate_coupler.isocalc (SI units; time, t0, t_sat and t_pms in yr). The toy uses f_atm, Mp, F0,
        T, time, rad_evol, Rp_override, the N_X, eps, the XUV settings (t0, t_sat, beta, step_fn, F_final, t_pms,
        pms_factor), n_steps, n_atmodeller, save_molecules, and melt_fraction_override for the melt-grid check.
        mechanism is only checked; Mstar, Fp, d, mu, activity, flux_model, stellar_type, rho_rcb, RR, thermal,
        mantle_iron_dict and dynamic_phi are ignored.

    Output: dict of arrays (n_steps,)
        - time: t0 + dt (1 ... n_steps + 1) with dt = (time - t0)/n_steps, so it ends one step past the age, as
          isocalc [s]
        - the escape kernel rows of OUTPUTS: Rp, Ratm [m], Matm [kg], Vpot [J/kg], fatm [ndim], Mloss [kg],
          phi, phic [kg/m2/s], N_X, N_X_int [atoms], x1-x7 [ndim], Phi_X [atoms/m2/s]; 0 once the atmosphere is
          lost, except Rp and Vpot (the bare core, or the fixed radius) and x1-x7 (frozen, as isocalc)
        - T_surf_analytic, T_surf_atmod: surface temperature [K], held from the most recent Atmodeller call
          (T_surf_atmod capped at 6000 K); 0 without Atmodeller and once the atmosphere is lost
        - n_<gas>_a, n_<gas>_a_int [mol] and fO2_a [bar]: with save_molecules only
        - atmodeller_final: the final Atmodeller dict, with n_atmodeller > 0 only; 21 NaN entries and no
          log10dIW_1_bar once the atmosphere is gone, as isocalc
    '''
    if n_atmodeller < 0 or n_atmodeller != int(n_atmodeller):
        raise ValueError(f'n_atmodeller must be a non-negative whole number of timesteps, got {n_atmodeller}')
    if mechanism not in ('XUV', 'CPML', 'XUV+CPML', 'phi kill'):
        raise ValueError(f"unknown mechanism '{mechanism}'")
    n, n_atm = int(n_steps), int(n_atmodeller)
    if n_atm != 0 and melt_fraction_override is False and not 1 <= Mp/Me <= 21:
        raise ValueError(f'melt fraction grid does not cover Mp = {Mp/Me} Me') # as atmodeller_coupler

    ###_____Time grid and atmospheric mass_____###

    t0_s = t0/s2yr # [s]
    delta_t = (time/s2yr - t0_s)/n # [s]
    t_a = delta_t*np.linspace(1, n + 1, n) + t0_s # [s], as isocalc
    t_yr = t_a*s2yr
    # 2 Gyr for 5.6 M_earth at eps = 0.15 and F0 = 0.588 W/m2; longer for a heavier planet, a lower heating
    # efficiency or a fainter XUV flux
    tau = 2e9*(Mp/(5.6*Me))**2*(0.15/eps)*np.sqrt(0.588/max(F0, 1e-30)) # e-folding time [yr]
    g = np.exp(-(t_yr - t0)/tau) # f_atm(t)/f_atm(t0)
    alive = f_atm*g >= F_LOST
    n_end = n if alive.all() else int(np.argmin(alive)) # first step without an atmosphere
    live = slice(0, n_end) # the steps with an atmosphere
    gl = g[:n_end]

    ###_____Radius and inventories_____###

    out = {name: np.zeros(n) for name in OUTPUTS}
    R_core = Re*(Mp/Me)**0.25 # [m], the core radius of isofunks.R_core (Lopez & Fortney 2014)
    R_fixed = float(Rp_override) if not rad_evol and Rp_override is not False else R_core # as isocalc's r_fixed
    fatm = f_atm*gl
    bias = 1 + 0.05*n_atm/n + 1e3/n # resolution-dependent error
    Rp = R_core*(1 + 4*np.sqrt(fatm))*bias if rad_evol else np.full(n_end, R_fixed)
    out['Rp'][:] = R_core if rad_evol else R_fixed # what is left after the atmosphere is lost
    out['Rp'][live] = Rp
    out['Ratm'][live] = Rp - R_core
    out['fatm'][live] = fatm
    out['Matm'][live] = fatm*Mp
    out['Vpot'][:] = G*Mp/out['Rp'] # [J/kg]
    out['Vpot'][live] = G*Mp/Rp

    N0 = dict(zip(SPECIES, (N_H, N_He, N_D, N_O, N_C, N_N, N_S))) # initial inventories [atoms]
    for X in SPECIES:
        out[f'N_{X}'][live] = N0[X]*gl**RETENTION[X]
        if n_atm > 0: # a mantle only with Atmodeller
            out[f'N_{X}_int'][live] = MANTLE*N0[X]*(1 - gl)
    N_tot = sum(out[f'N_{X}'][live] for X in SPECIES)
    for i, X in enumerate(SPECIES, 1): # mole fractions x1-x7
        out[f'x{i}'][live] = out[f'N_{X}'][live]/np.where(N_tot > 0, N_tot, 1)
        if n_end < n:
            out[f'x{i}'][n_end:] = out[f'x{i}'][n_end - 1] if n_end > 0 else 0 # frozen, as isocalc

    ###_____Escape_____###

    F = _xuv(t_yr[live], F0, t0, t_sat, beta, step_fn, F_final, t_pms, pms_factor)
    phi = eps*F/(4*out['Vpot'][live]) # energy-limited mass flux [kg/m2/s]
    phic = np.full(n_end, 5e-11*Mp/(5.6*Me)) # critical flux for He escape [kg/m2/s]
    out['phi'][live], out['phic'][live] = phi, phic
    out['Mloss'][live] = phi*4*np.pi*Rp**2*delta_t # [kg] per step
    supercritical = phi > phic # He and the metals escape only above the critical flux
    N_Hl = np.where(out['N_H'][live] > 0, out['N_H'][live], 1) # N_H, safe to divide by
    Phi_H = phi*out['x1'][live]/mu_H # [atoms/m2/s]
    out['Phi_H'][live] = Phi_H
    # He follows H in proportion to N_He/N_H, damped near the critical flux
    out['Phi_He'][live] = np.where(supercritical, Phi_H*out['N_He'][live]/N_Hl*(1 - phic/np.maximum(phi, phic)), 0)
    out['Phi_D'][live] = 0.8*Phi_H*out['N_D'][live]/N_Hl # D: 0.8 times the H rate per atom
    for X in ('O', 'C', 'N', 'S'): # metals: 0.1 times the H rate per atom, above the critical flux only
        out[f'Phi_{X}'][live] = np.where(supercritical, 0.1*Phi_H*out[f'N_{X}'][live]/N_Hl, 0)

    ###_____Surface temperature and molecules_____###

    sol = {'time': t_a}
    sol.update(out)
    T_surf_analytic, T_surf_atmod = np.zeros(n), np.zeros(n)
    molecules = {}
    if n_atm > 0:
        calls = np.arange(n_end)//n_atm*n_atm # values hold from the most recent Atmodeller call
        T_s = T*(1 + 500*np.sqrt(f_atm*g[calls])) # [K], hotter under a thicker atmosphere
        T_surf_analytic[live], T_surf_atmod[live] = T_s, np.minimum(T_s, 6000.0) # capped, as atmodeller_coupler
        if save_molecules:
            molecules = _molecules(*(out[f'N_{X}'][calls] for X in ('H', 'D', 'O', 'C', 'N', 'S')))
    sol['T_surf_analytic'], sol['T_surf_atmod'] = T_surf_analytic, T_surf_atmod

    if save_molecules:
        for suffix, scale in (('', 1.0), ('_int', MANTLE)): # atmosphere, and a dissolved copy
            for gas in GASES:
                sol[f'n_{gas}_a{suffix}'] = np.zeros(n)
                if gas in molecules:
                    sol[f'n_{gas}_a{suffix}'][live] = scale*molecules[gas]
        sol['fO2_a'] = np.zeros(n)
        if n_atm > 0:
            sol['fO2_a'][live] = 1e-8 # [bar]

    ###_____Final Atmodeller dict_____###

    if n_atm > 0:
        final = {}
        if n_end == n:
            gas = _molecules(*(out[f'N_{X}'][-1] for X in ('H', 'D', 'O', 'C', 'N', 'S')))
            for name in ('H2O', 'H2', 'O2', 'CO', 'CO2', 'CH4', 'N2', 'S2', 'H2O4S'):
                final[f'{name}_atm'] = float(gas[name]) # [mol]
                final[f'{name}_mantle'] = float(MANTLE*gas[name])
            final['SO2_atm'] = float(gas['SO2'])
            final['He_mantle'] = float(out['N_He_int'][-1]/avogadro) # [mol]
            final['O2_fugacity'] = 1e-8 # [bar]
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
