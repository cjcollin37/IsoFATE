'''
Collin Cherubim
June 30, 2025
Main IsoFATE script for coulped model.
'''

import numpy as np
from isofate.atmodeller_coupler import *
from isofate.constants import *
from isofate.isofunks import *
from isofate.orbit_params import *
from isofate import escape_kernel
from isofate.escape_kernel import FluxModel, Mechanism


def isocalc(f_atm, Mp, Mstar, F0, Fp, T, d, time = 5e9, mechanism = 'XUV', rad_evol = True,
N_H = 0, N_He = 0, N_D = 0, N_O = 0, N_C = 0, N_N = 0, N_S = 0, melt_fraction_override = False,
mu = mu_solar, eps = 0.15, activity = 'medium', flux_model = 'power law', stellar_type = 'M1',
Rp_override = False, t_sat = 5e8, step_fn = False, F_final = 0, t_pms = 0, pms_factor = 1e2,
n_steps = int(1e5), t0 = 1e6, rho_rcb = 1.0, RR = True, thermal = True, 
beta = -1.23, n_atmodeller = int(1e2), save_molecules = False, mantle_iron_dict = False,
dynamic_phi = False):
    """
    This is a test
    
    Description

    Args:
        a (array): a test array
        b (array): a test array

    Returns:
        array: a test array

    Returns:
        array: a test array
    """

    # '''
    # Computes species abundances in ternary mixture of H, D, and He 
    # via time-integrated numerical simulation of atmospheric escape.
    # Note: species 1 is H, species 2 is He, species 3 is D

    # Inputs:
    #  - f_atm: atmospheric mass fraction(s), must be in the form of an array [ndim]
    #  - Mp: planet mass [kg]
    #  - Mstar: stellar mass [kg]
    #  - F0: initial incident XUV flux [W/m2]
    #  - Fp: incident bolometric flux [W/m2]
    #  - T: planet equilibrium temperature [K]
    #  - d: orbtial distance [m]
    #  - time: total simulation time; scalar [yr]
    #  - mechanism: 'XUV', 'XUV+RR', 'CPML', 'XUV+CPML', 'fix phi subcritical',
    #  'fix phi supercritical', 'phi kill'
    #  - rad_evol: set to False to fix planet radius at core radius [Bool]
    #  - N_x: initial abundance for species x [atoms]
    #  - melt_fraction_override: set a fixed mantle melt fraction. If False, melt fraction is calculated based on Mp and T_surface [False or float]
    #  - mu: average atmospheric particle mass, default to H/He solar comp [kg]
    #  - eps: heat transfer efficiency [ndim]
    #  - activity: for Fxuv_hazmat function (uses semi-empirical data from MUSCLES survey); 
    #  'low' (lower quartile), 'medium' (median), 'high' (upper quartile)
    #  - flux_model: 'power law' for analytic power law, 'phoenix' for Fxuv_hazmat, 'Johnstone' for Fxuv_Johnstone
    #  - stellar_type: 'M1', 'K5', or 'G5' must be specified for flux_model == Fxuv_Johnstone
    #  - Rp_override: enter scalar planet radius value to manually set constant radius, note radius will not evolve [m]
    #  - t_sat: saturation time for Fxuv power law; 5e8 matches semi-empirical MUSCLES data [yr]
    #  - n_steps: number of timesteps. Convergence occurs at 1e6.
    #  - t0: simulation start time [yr]
    #  - rho_rcb: gas density at the RCB in CPML phi equatin [kg/m3]
    #  - RR: toggles radiation-recombination effect (Ly alpha cooling; Murray-Clay et al 2009)  [Bool]
    #  - thermal: toggles planet radius contraction in Lopez/Fortney equations (False removes age term) [Bool]
    #  - beta: exponential in Fxuv function; determines rate of XUV decrease. -1.23 consistent with MUSCLES data
    #  - n_atmodeller: interval of timesteps between each Atmodeller call
    #  - save_molecules: save molecular abundances at each time step [True] or only final abundances [False]
    #  - mantle_iron_dict: allow Fe in mantle to react with O2.['type']="dynamic" to allow only molten mantle Fe to react; 
    #  ['type']="static" to allow all mantle Fe to react; specify ['Fe_mass_fraction']
    #  - dynamic_phi: toggle dynamic phi calculation based on most abundant species [True] or static phi calculation [False]
     
    # Output: Dictionary of 2-D arrays [len(f_atm) x n_steps] with keys,
    #  - 'time': simulation time array [s]
    #  - 'Rp': total planet radius [m]
    #  - 'Ratm': convective atm depth [m]
    #  - 'Matm': atmospheric mass [kg]
    #  - 'Vpot': gravitational potential at outer layer [J/kg]
    #  - 'fatm': total atmospheric mass fraction [ndim]
    #  - 'Mloss': atm mass loss per time step [kg]
    #  - 'phi': atm mass flux [kg/m2/s]
    #  - 'phic': critical mass flux for species 2 escape [kg/m2/s]
    #  - 'N_H': H number [atoms]
    #  - 'N_He': He number [atoms]
    #  - 'N_D': D number [atoms]
    #  - 'x1': H molar concentration [ndim]
    #  - 'x2': He molar concentration [ndim]
    #  - 'Phi_H': H number flux [atoms/s/m2]
    #  - 'Phi_He': He number flux [atoms/s/m2]
    #  - 'Phi_D': D number flux [atoms/s/m2]
    # '''
    
###_____Initialize physical values_____###

    radius_core = R_core(Mp) # [m]
    R_B = R_Bondi(Mp, mu, T) # Bondi radius [m]
    R_H = R_Hill(Mp, Mstar, d) # Hill radius [m]

    # binary diffusion coefficients [molecules/m/s], indexed by species [H, He, D, O, C, N, S]
    species_names = ['H', 'He', 'D', 'O', 'C', 'N', 'S']
    if dynamic_phi == True:
        b_table = np.array([[get_binary_diffusion_coeff(s1, s2, T) for s2 in species_names] for s1 in species_names],
                           dtype=float)
    else:
        b_table = np.zeros((7, 7))
        b_table[0, 1] = b_H_He(T) # from Mason & Marrero 1970 for H in He
        b_table[0, 2:] = [b_H_D(T), b_H_O(T), b_H_C(T), b_H_N(T), b_H_S(T)]
        b_table[1, 2:] = [b_He_D(T), b_He_O(T), b_He_C(T), b_He_N(T), b_He_S(T)]

###_____Initialize timesteps_____###

    n_tot = n_steps # timesteps
    t0 = t0/s2yr # simulation start time [s]
    t = time/s2yr - t0 # total simulation time [s]
    delta_t = t/n_tot # timestep [s]
    t_a = delta_t*np.linspace(1, n_tot + 1, n_tot) + t0 # time array [s]
    if n_atmodeller < 0 or n_atmodeller != int(n_atmodeller):
        raise ValueError(f'n_atmodeller must be a non-negative whole number of timesteps, got {n_atmodeller}')
    n_atmodeller = int(n_atmodeller)

    mechanisms = {'XUV': Mechanism.XUV, 'CPML': Mechanism.CPML, 'phi kill': Mechanism.PHI_KILL,
                  'XUV+CPML': Mechanism.XUV_CPML}
    if mechanism not in mechanisms:
        raise ValueError(f"unknown mechanism '{mechanism}'")
    flux_models = {'power law': FluxModel.POWER_LAW, 'phoenix': FluxModel.PHOENIX, 'Johnstone': FluxModel.JOHNSTONE}
    F_track = np.empty(0) # incident XUV flux per timestep for tabulated flux models [W/m2]
    if mechanism in ('XUV', 'XUV+CPML'):
        if flux_model not in flux_models:
            raise ValueError(f"unknown flux_model '{flux_model}'")
        if flux_model == 'phoenix':
            F_track = np.array([Fxuv_hazmat(t_n, d, activity) for t_n in t_a], dtype=float)
        elif flux_model == 'Johnstone':
            F_track = np.asarray(Fxuv_Johnstone(t_a, d, stellar_type), dtype=float)

    # mechanism and flux_model as plain ints: Numba checks IntEnum arguments slowly on every kernel call
    params = escape_kernel.Params(
        Mp=float(Mp), Mstar=float(Mstar), F0=float(F0), Fp=float(Fp), T=float(T), d=float(d), delta_t=float(delta_t),
        t_total=float(t), t0=float(t0*s2yr), t_sat=float(t_sat), beta=float(beta), step_fn=step_fn == True,
        F_final=float(F_final), t_pms=float(t_pms), pms_factor=float(pms_factor), eps=float(eps), rho_rcb=float(rho_rcb),
        RR=RR == True, thermal=thermal == True, rad_evol=rad_evol != False, dynamic_phi=dynamic_phi == True,
        mechanism=int(mechanisms[mechanism]), flux_model=int(flux_models.get(flux_model, FluxModel.POWER_LAW)),
        r_core=float(radius_core),
        r_fixed=float(Rp_override) if rad_evol == False and Rp_override != False else float(radius_core),
        R_B=float(R_B), R_H=float(R_H))

###_____Set initial values____###

    ### atmodeller interior
    X_DH = N_D/(N_H + N_D) if N_H + N_D != 0 else 0 # ignores D in mantle

    if n_atmodeller != 0:
        atmodeller_solver = AtmodellerSolver()
    atmod_full_output = {} # dictionary to store atmodeller full output

    if mantle_iron_dict:
        mantle_iron_dict['mantle_mass'] = 0.704665308539034*Mp # fraction from atmodeller
        mantle_iron_dict['mass_Fe'] = mantle_iron_dict['mantle_mass']*mantle_iron_dict['Fe_mass_fraction']
        mantle_iron_dict['mass_Fe2'] = mantle_iron_dict['mass_Fe']
        mantle_iron_dict['X_Fe2'] = mantle_iron_dict['mass_Fe2']/mantle_iron_dict['mass_Fe']

    ### atmosphere
    M_atm = Mp*f_atm # initial atmospheric mass [kg]
    y = np.array([N_H, N_He, N_D, N_O, N_C, N_N, N_S], dtype=float) # atmospheric numbers [atoms]
    y_int = np.zeros(7) # mantle numbers [atoms]
    flux = np.zeros(7) # number fluxes [atoms/s/m2]
    state = np.zeros(5)
    state[escape_kernel.M_ATM] = M_atm
    state[escape_kernel.F_ATM] = f_atm
    state[escape_kernel.X_DH] = X_DH

###_____Initialize arrays_____###

    out = np.zeros((len(escape_kernel.OUTPUTS), n_tot)) # per-step outputs, rows named by escape_kernel.OUTPUTS

    H2_a = np.zeros(n_tot) # atmospheric H2 number array [molecules]
    H2O_a = np.zeros(n_tot) # atmospheric H2O number array [molecules]
    O2_a = np.zeros(n_tot) # atmospheric O2 number array [molecules]
    CO2_a = np.zeros(n_tot) # atmospheric CO2 number array [molecules]
    CO_a = np.zeros(n_tot) # atmospheric CO number array [molecules]
    CH4_a = np.zeros(n_tot) # atmospheric CH4 number array [molecules]
    N2_a = np.zeros(n_tot) # atmospheric N2 number array [molecules]
    S2_a = np.zeros(n_tot) # atmospheric S2 number array [molecules]
    H2O4S_a = np.zeros(n_tot) # atmospheric H2O4S gas number array [molecules]
    SO2_a = np.zeros(n_tot) # atmospheric SO2 number array [molecules]
    fO2_a = np.zeros(n_tot) # fugacity array [bar]
    H2_a_int = np.zeros(n_tot) # mantle H2 number array [molecules]
    H2O_a_int = np.zeros(n_tot) # mantle H2O number array [molecules]
    O2_a_int = np.zeros(n_tot) # mantle O2 number array [molecules]
    CO2_a_int = np.zeros(n_tot) # mantle CO2 number array [molecules]
    CO_a_int = np.zeros(n_tot) # mantle CO number array [molecules]
    CH4_a_int = np.zeros(n_tot) # mantle CH4 number array [molecules]
    N2_a_int = np.zeros(n_tot) # mantle N2 number array [molecules]
    S2_a_int = np.zeros(n_tot) # mantle S2 number array [molecules]
    H2O4S_a_int = np.zeros(n_tot) # mantle H2O4S gas number array [molecules]
    SO2_a_int = np.zeros(n_tot) # mantle SO2 number array [molecules]
    T_surf_analytic_a = np.zeros(n_tot) # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot) # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Loop through timesteps_____###

    n_end = n_tot # first timestep without an atmosphere
    n = 0
    while n < n_tot:
        # k: next step that calls atmodeller (a multiple of n_atmodeller, or the final step)
        k = n_tot if n_atmodeller == 0 else min(-(-n//n_atmodeller)*n_atmodeller, n_tot - 1)
        lost = escape_kernel.run(n, k, t_a, y, y_int, flux, state, out, params, F_track, b_table)
        if lost >= 0:
            n_end = lost
            break
        if k == n_tot:
            break
        if not escape_kernel.fluxes(k, t_a, y, y_int, flux, state, out, params, F_track, b_table):
            n_end = k
            break
        n = k
        radius_p = out[escape_kernel.RP, n]
        mu = state[escape_kernel.MU_ATM]
        y1, y2, y3, y4, y5, y6, y7 = y
        N_H_int, N_He_int, N_D_int, N_O_int, N_C_int, N_N_int, N_S_int = y_int
        X_DH = state[escape_kernel.X_DH]

        ##### run atmodeller ######
        if n == n_steps - 1:
            # atmod_full_output = {}
            atmod_sol = AtmodellerCoupler(T, Mp, radius_p, mu, melt_fraction_override, mantle_iron_dict,
                                              y1+y3, y2, y4, y5, y6, y7, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, atmodeller_solver)[1]
            atmod_full_output['H2O_atm'] = atmod_sol['H2O_g']['atmosphere_moles'][0]
            atmod_full_output['H2O_mantle'] = atmod_sol['H2O_g']['dissolved_moles'][0]
            atmod_full_output['H2_atm'] = atmod_sol['H2_g']['atmosphere_moles'][0]
            atmod_full_output['H2_mantle'] = atmod_sol['H2_g']['dissolved_moles'][0]
            atmod_full_output['O2_atm'] = atmod_sol['O2_g']['atmosphere_moles'][0]
            atmod_full_output['O2_mantle'] = atmod_sol['O2_g']['dissolved_moles'][0]
            atmod_full_output['CO_atm'] = atmod_sol['CO_g']['atmosphere_moles'][0]
            atmod_full_output['CO_mantle'] = atmod_sol['CO_g']['dissolved_moles'][0]
            atmod_full_output['CO2_atm'] = atmod_sol['CO2_g']['atmosphere_moles'][0]
            atmod_full_output['CO2_mantle'] = atmod_sol['CO2_g']['dissolved_moles'][0]
            atmod_full_output['CH4_atm'] = atmod_sol['CH4_g']['atmosphere_moles'][0]
            atmod_full_output['CH4_mantle'] = atmod_sol['CH4_g']['dissolved_moles'][0]
            atmod_full_output['N2_atm'] = atmod_sol['N2_g']['atmosphere_moles'][0]
            atmod_full_output['N2_mantle'] = atmod_sol['N2_g']['dissolved_moles'][0]
            atmod_full_output['S2_atm'] = atmod_sol['S2_g']['atmosphere_moles'][0]
            atmod_full_output['S2_mantle'] = atmod_sol['S2_g']['dissolved_moles'][0]
            atmod_full_output['H2O4S_atm'] = atmod_sol['H2O4S_g']['atmosphere_moles'][0]
            atmod_full_output['H2O4S_mantle'] = atmod_sol['H2O4S_g']['dissolved_moles'][0]
            atmod_full_output['SO2_atm'] = atmod_sol['O2S_g']['atmosphere_moles'][0]
            atmod_full_output['He_mantle'] = atmod_sol['He_g']['dissolved_moles'][0]
            atmod_full_output['O2_fugacity'] = atmod_sol['O2_g']['fugacity'][0]
            atmod_full_output['log10dIW_1_bar'] = atmod_sol['O2_g']['log10dIW_1_bar'][0]
        if n%n_atmodeller == 0: # run atmodeller every n_atmodeller steps.
            atmod_results, atmod_full, mantle_iron_dict = AtmodellerCoupler(T, Mp, radius_p, mu, melt_fraction_override, mantle_iron_dict,
                                              y1+y3, y2, y4, y5, y6, y7, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, atmodeller_solver)
            N_H_int = atmod_results['N_H_int']*(1 - X_DH)
            N_D_int = atmod_results['N_H_int']*X_DH
            N_He_int = atmod_results['N_He_int']
            N_O_int = atmod_results['N_O_int']
            N_C_int = atmod_results['N_C_int']
            N_N_int = atmod_results['N_N_int']
            N_S_int = atmod_results['N_S_int']
            if atmod_results['N_H_atm'] == 0:
                y1 = 0
                y3 = 0
            else:
                Y3 = X_DH*atmod_results['N_H_atm']
                Y1 = (1 - X_DH)*atmod_results['N_H_atm']
                y1 = Y1
                y3 = Y3
            y2 = atmod_results['N_He_atm']
            y4 = atmod_results['N_O_atm']
            y5 = atmod_results['N_C_atm']
            y6 = atmod_results['N_N_atm']
            y7 = atmod_results['N_S_atm']
            M_atm = atmod_results['M_atm']
            T_surf_analytic = atmod_results['T_surface']
            T_surf_atmod = atmod_results['T_surface_atmod']
            if save_molecules == True:
                H2_a[n] = atmod_full['H2_g']['atmosphere_moles'][0]
                H2O_a[n] = atmod_full['H2O_g']['atmosphere_moles'][0]
                O2_a[n] = atmod_full['O2_g']['atmosphere_moles'][0]
                CO2_a[n] = atmod_full['CO2_g']['atmosphere_moles'][0]
                CO_a[n] = atmod_full['CO_g']['atmosphere_moles'][0]
                CH4_a[n] = atmod_full['CH4_g']['atmosphere_moles'][0]
                N2_a[n] = atmod_full['N2_g']['atmosphere_moles'][0]
                S2_a[n] = atmod_full['S2_g']['atmosphere_moles'][0]
                H2O4S_a[n] = atmod_full['H2O4S_g']['atmosphere_moles'][0]
                SO2_a[n] = atmod_full['O2S_g']['atmosphere_moles'][0]
                H2_a_int[n] = atmod_full['H2_g']['dissolved_moles'][0]
                H2O_a_int[n] = atmod_full['H2O_g']['dissolved_moles'][0]
                O2_a_int[n] = atmod_full['O2_g']['dissolved_moles'][0]
                CO2_a_int[n] = atmod_full['CO2_g']['dissolved_moles'][0]
                CO_a_int[n] = atmod_full['CO_g']['dissolved_moles'][0]
                CH4_a_int[n] = atmod_full['CH4_g']['dissolved_moles'][0]
                N2_a_int[n] = atmod_full['N2_g']['dissolved_moles'][0]
                S2_a_int[n] = atmod_full['S2_g']['dissolved_moles'][0]
                H2O4S_a_int[n] = atmod_full['H2O4S_g']['dissolved_moles'][0]
                SO2_a_int[n] = atmod_full['O2S_g']['dissolved_moles'][0]
                fO2_a[n] = atmod_full['O2_g']['fugacity'][0]
            y[:] = [y1, y2, y3, y4, y5, y6, y7]
            y_int[:] = [N_H_int, N_He_int, N_D_int, N_O_int, N_C_int, N_N_int, N_S_int]
            state[escape_kernel.M_ATM] = M_atm
            T_surf_analytic_a[n] = T_surf_analytic
            T_surf_atmod_a[n] = T_surf_atmod

        # advance to next step
        escape_kernel.advance(y, y_int, flux, state, params)
        n += 1

    # values between atmodeller calls hold from the most recent call
    if n_atmodeller != 0:
        last_call = np.arange(n_end)//n_atmodeller*n_atmodeller
        for a in (H2_a, H2O_a, O2_a, CO2_a, CO_a, CH4_a, N2_a, S2_a, H2O4S_a, SO2_a, fO2_a, H2_a_int, H2O_a_int,
                  O2_a_int, CO2_a_int, CO_a_int, CH4_a_int, N2_a_int, S2_a_int, H2O4S_a_int, SO2_a_int,
                  T_surf_analytic_a, T_surf_atmod_a):
            a[:n_end] = a[last_call]

    ### Entire atmosphere lost: atmodeller full ouput for monte carlo runs
    if n_end < n_tot and n_atmodeller != 0:
        atmod_full_output['H2O_atm'] = np.nan
        atmod_full_output['H2O_mantle'] = np.nan
        atmod_full_output['H2_atm'] = np.nan
        atmod_full_output['H2_mantle'] = np.nan
        atmod_full_output['O2_atm'] = np.nan
        atmod_full_output['O2_mantle'] = np.nan
        atmod_full_output['CO_atm'] = np.nan
        atmod_full_output['CO_mantle'] = np.nan
        atmod_full_output['CO2_atm'] = np.nan
        atmod_full_output['CO2_mantle'] = np.nan
        atmod_full_output['CH4_atm'] = np.nan
        atmod_full_output['CH4_mantle'] = np.nan
        atmod_full_output['He_mantle'] = np.nan
        atmod_full_output['N2_atm'] = np.nan
        atmod_full_output['N2_mantle'] = np.nan
        atmod_full_output['S2_atm'] = np.nan
        atmod_full_output['S2_mantle'] = np.nan
        atmod_full_output['H2O4S_atm'] = np.nan
        atmod_full_output['H2O4S_mantle'] = np.nan
        atmod_full_output['SO2_atm'] = np.nan
        atmod_full_output['O2_fugacity'] = np.nan

    # save results
    solutions = {'time': t_a}
    solutions.update(zip(escape_kernel.OUTPUTS, out))
    solutions['T_surf_analytic'] = T_surf_analytic_a
    solutions['T_surf_atmod'] = T_surf_atmod_a
    if save_molecules == True:
        solutions['n_H2_a'] = H2_a
        solutions['n_H2O_a'] = H2O_a
        solutions['n_O2_a'] = O2_a
        solutions['n_CO2_a'] = CO2_a
        solutions['n_CO_a'] = CO_a
        solutions['n_CH4_a'] = CH4_a
        solutions['n_N2_a'] = N2_a
        solutions['n_S2_a'] = S2_a
        solutions['n_H2O4S_a'] = H2O4S_a
        solutions['n_SO2_a'] = SO2_a
        solutions['n_H2_a_int'] = H2_a_int
        solutions['n_H2O_a_int'] = H2O_a_int
        solutions['n_O2_a_int'] = O2_a_int
        solutions['n_CO2_a_int'] = CO2_a_int
        solutions['n_CO_a_int'] = CO_a_int
        solutions['n_CH4_a_int'] = CH4_a_int
        solutions['n_N2_a_int'] = N2_a_int
        solutions['n_S2_a_int'] = S2_a_int
        solutions['n_H2O4S_a_int'] = H2O4S_a_int
        solutions['n_SO2_a_int'] = SO2_a_int
        solutions['fO2_a'] = fO2_a
    if n_atmodeller != 0:
        solutions['atmodeller_final'] = atmod_full_output

    return solutions
