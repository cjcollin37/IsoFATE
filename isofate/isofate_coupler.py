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
from atmodeller import InteriorAtmosphere
from atmodeller import Species
from atmodeller import SpeciesCollection
from atmodeller.solubility import get_solubility_models
solubility_models = get_solubility_models()


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

    b = 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
    radius_core = R_core(Mp) # [m]
    R_B = R_Bondi(Mp, mu, T) # Bondi radius [m]
    R_H = R_Hill(Mp, Mstar, d) # Hill radius [m]

###_____Initialize timesteps_____###

    n_tot = n_steps # timesteps
    t0 = t0/s2yr # simulation start time [s]
    t = time/s2yr - t0 # total simulation time [s]
    delta_t = t/n_tot # timestep [s]

###_____Set initial values____###

    atomic_masses = [mu_H, mu_He, mu_D, mu_O, mu_C, mu_N, mu_S]
    species_names = ['H', 'He', 'D', 'O', 'C', 'N', 'S']

    ### atmodeller interior
    N_H_int = 0
    N_He_int = 0
    N_D_int = 0
    N_O_int = 0
    N_C_int = 0
    N_N_int = 0
    N_S_int = 0
    if n_atmodeller == 0:
        T_surf_analytic = 0
        T_surf_atmod = 0
    if N_H != 0 and N_D != 0: # needed to allow D and H to outgas from mantle
        X_DH = N_D/(N_H + N_D) # ignores D in mantle

    H2O_g = Species.create_gas("H2O", solubility=solubility_models["H2O_basalt_dixon95"])
    H2_g = Species.create_gas("H2", solubility=solubility_models["H2_basalt_hirschmann12"])
    O2_g = Species.create_gas("O2")
    CO_g = Species.create_gas("CO", solubility=solubility_models["CO_basalt_yoshioka19"])
    CO2_g = Species.create_gas("CO2", solubility=solubility_models["CO2_basalt_dixon95"])
    CH4_g = Species.create_gas("CH4", solubility=solubility_models["CH4_basalt_ardia13"])
    He_g = Species.create_gas("He", solubility=solubility_models["He_basalt_jambon86"])
    N2_g = Species.create_gas("N2", solubility=solubility_models["N2_basalt_libourel03"])
    S2_g = Species.create_gas("S2", solubility=solubility_models["S2_sulfide_basalt_boulliung23"])
    H2O4S_g = Species.create_gas("H2O4S")
    SO2_g = Species.create_gas("SO2")

    species = SpeciesCollection((H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g, He_g, N2_g, S2_g, H2O4S_g, SO2_g))
    interior_atmosphere = InteriorAtmosphere(species)
    atmod_full_output = {} # dictionary to store atmodeller full output

    if mantle_iron_dict:
        mantle_iron_dict['mantle_mass'] = 0.704665308539034*Mp # fraction from atmodeller
        mantle_iron_dict['mass_Fe'] = mantle_iron_dict['mantle_mass']*mantle_iron_dict['Fe_mass_fraction']
        mantle_iron_dict['mass_Fe2'] = mantle_iron_dict['mass_Fe']
        mantle_iron_dict['X_Fe2'] = mantle_iron_dict['mass_Fe2']/mantle_iron_dict['mass_Fe']

    ### atmosphere
    M_atm0 = Mp*f_atm # initial atmospheric mass [kg]
    M_atm = M_atm0
    y1 = 1*N_H # H number [atoms]
    y2 = 1*N_He # He number [atoms]
    y3 = 1*N_D # D number [atoms]
    y4 = 1*N_O # O number [atoms]
    y5 = 1*N_C # O number [atoms]
    y6 = 1*N_N # N number [atoms]
    y7 = 1*N_S # S number [atoms]
###_____Initialize arrays_____###

    t_a = delta_t*np.linspace(1, n_tot + 1, n_tot) + t0 # time array [s]
    phi_a = np.zeros(n_tot) # mass flux array [kg/s/m2]
    phic_a = np.zeros(n_tot) # critical mass flux array [kg/s/m2] 
    Rp_a = np.zeros(n_tot) # total radius, diagnostic [m]
    Renv_a = np.zeros(n_tot) # envelope radius [m]
    Matm_a = np.zeros(n_tot) # atmospheric mass [kg]
    fatm_a = np.zeros(n_tot) # atm mass fraction [ndim]
    Vpot_a = np.zeros(n_tot) # grav potential, diagnostic [J/kg]
    Mloss_a = np.zeros(n_tot) # mass lost per timestep [kg]

    y1_a = np.zeros(n_tot) # H number array [atoms]
    y2_a = np.zeros(n_tot) # He number array [atoms]
    y3_a = np.zeros(n_tot) # D number array [atoms]
    y4_a = np.zeros(n_tot) # O number array [atoms]
    y5_a = np.zeros(n_tot) # C number array [atoms]
    y6_a = np.zeros(n_tot) # N number array [atoms]
    y7_a = np.zeros(n_tot) # S number array [atoms]
    y1_a_int = np.zeros(n_tot) # mantle H number array [atoms]
    y2_a_int = np.zeros(n_tot) # mantle He number array [atoms]
    y3_a_int = np.zeros(n_tot) # mantle D number array [atoms]
    y4_a_int = np.zeros(n_tot) # mantle O number array [atoms]
    y5_a_int = np.zeros(n_tot) # mantle C number array [atoms]
    y6_a_int = np.zeros(n_tot) # mantle N number array [atoms]
    y7_a_int = np.zeros(n_tot) # mantle S number array [atoms]
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
    x1_a = np.zeros(n_tot) # H molar concentration array [ndim]
    x2_a = np.zeros(n_tot) # He molar concentration array [ndim]
    x3_a = np.zeros(n_tot) # D molar concentration array [ndim]
    x4_a = np.zeros(n_tot) # O molar concentration array [ndim]
    x5_a = np.zeros(n_tot) # C molar concentration array [ndim]
    x6_a = np.zeros(n_tot) # N molar concentration array [ndim]
    x7_a = np.zeros(n_tot) # S molar concentration array [ndim]
    Phi_H_a = np.zeros(n_tot) # H number flux array [atoms/s/m2]
    Phi_He_a = np.zeros(n_tot) # He number flux array [atoms/s/m2]
    Phi_D_a = np.zeros(n_tot) # D number flux array [atoms/s/m2]
    Phi_O_a = np.zeros(n_tot) # O number flux array [atoms/s/m2]
    Phi_C_a = np.zeros(n_tot) # C number flux array [atoms/s/m2]
    Phi_N_a = np.zeros(n_tot) # N number flux array [atoms/s/m2]
    Phi_S_a = np.zeros(n_tot) # S number flux array [atoms/s/m2]
    T_surf_analytic_a = np.zeros(n_tot) # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot) # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Loop through timesteps_____###

    for n in range(n_tot):

    ### Stop simulation when entire atmosphere is lost
        if M_atm <= 0 or y1 + y2 + y3 + y4 + y5 + y6 + y7 <= 0:
            Matm_a[n:] = 0 #M_atm #Matm_a[n-1]
            fatm_a[n:] = 0 #f_atm #fatm_a[n-1]
            Renv_a[n:] = 0 #radius_env #Renv_a[n-1]
            Rp_a[n:] = radius_core
            K = np.max([V_reduction(Mp, Mstar, d, radius_core), 0.01]) # grav potential reduction factor due to stellar tidal forces
            Vpot_a[n:] = K*G*Mp/radius_core
            phi_a[n:] = 0
            Mloss_a[n:] = 0

            y1_a[n:] = 0 #y1 #y1_a[n-1]
            y2_a[n:] = 0 #y2 #y2_a[n-1]
            y3_a[n:] = 0 #y3 #y3_a[n-1]
            y4_a[n:] = 0 #y4 #y4_a[n-1]
            y5_a[n:] = 0 #y5 #y5_a[n-1]
            y6_a[n:] = 0 #y6 #y6_a[n-1]
            y7_a[n:] = 0 #y7 #y7_a[n-1]
            y1_a_int[n:] = 0 #y1_int #y1_a_int[n-1]
            y2_a_int[n:] = 0 #y2_int #y2_a_int[n-1]
            y3_a_int[n:] = 0 #y3_int #y3_a_int[n-1]
            y4_a_int[n:] = 0 #y4_int #y4_a_int[n-1]
            y5_a_int[n:] = 0 #y5_int #y5_a_int[n-1]
            y6_a_int[n:] = 0 #y6_int #y6_a_int[n-1]
            y7_a_int[n:] = 0 #y7_int #y7_a_int[n-1]
            H2_a[n:] = 0 #H2_a[n-1]
            H2O_a[n:] = 0 #H2O_a[n-1]
            O2_a[n:] = 0 #O2_a[n-1]
            CO2_a[n:] = 0 #CO2_a[n-1]
            CO_a[n:] = 0 #CO_a[n-1]
            CH4_a[n:] = 0 #CH4_a[n-1]
            N2_a[n:] = 0 #N2_a[n-1]
            S2_a[n:] = 0 #S2_a[n-1]
            H2_a_int[n:] = 0 #H2_a_int[n-1]
            H2O_a_int[n:] = 0 #H2O_a_int[n-1]
            O2_a_int[n:] = 0 #O2_a_int[n-1]
            CO2_a_int[n:] = 0 #CO2_a_int[n-1]
            CO_a_int[n:] = 0 #CO_a_int[n-1]
            CH4_a_int[n:] = 0 #CH4_a_int[n-1]
            N2_a_int[n:] = 0 #N2_a_int[n-1]
            S2_a_int[n:] = 0 #S2_a_int[n-1]
            H2O4S_a_int[n:] = 0 #H2O4S_a_int[n-1]
            SO2_a_int[n:] = 0 #SO2_a_int[n-1]
            x1_a[n:] = x1_a[n-1] # x1_a[max(np.nonzero(x1_a)[0])] # get last non-zero value in array
            x2_a[n:] = x2_a[n-1] # x2_a[max(np.nonzero(x2_a)[0])]
            x3_a[n:] = x3_a[n-1] # x3_a[max(np.nonzero(x3_a)[0])]
            x4_a[n:] = x4_a[n-1] # x4_a[max(np.nonzero(x4_a)[0])]
            x5_a[n:] = x5_a[n-1] # x5_a[max(np.nonzero(x5_a)[0])]
            x6_a[n:] = x6_a[n-1] # x6_a[max(np.nonzero(x6_a)[0])]
            x7_a[n:] = x7_a[n-1] # x7_a[max(np.nonzero(x7_a)[0])]
            Phi_H_a[n:] = 0
            Phi_He_a[n:] = 0    
            Phi_D_a[n:] = 0
            Phi_O_a[n:] = 0
            Phi_C_a[n:] = 0
            Phi_N_a[n:] = 0
            Phi_S_a[n:] = 0

            # atmodeller full ouput for monte carlo runs
            if n_atmodeller != 0:
                    # atmod_full_output = {}
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
                if save_molecules == True:
                    H2_a[n:] = 0
                    H2O_a[n:] = 0
                    O2_a[n:] = 0
                    CO2_a[n:] = 0
                    CO_a[n:] = 0
                    CH4_a[n:] = 0
                    N2_a[n:] = 0
                    S2_a[n:] = 0
                    H2O4S_a[n:] = 0
                    SO2_a[n:] = 0
                    H2_a_int[n:] = 0
                    H2O_a_int[n:] = 0
                    O2_a_int[n:] = 0
                    CO2_a_int[n:] = 0
                    CO_a_int[n:] = 0
                    CH4_a_int[n:] = 0
                    N2_a_int[n:] = 0
                    S2_a_int[n:] = 0
                    H2O4S_a_int[n:] = 0
                    SO2_a_int[n:] = 0
                    fO2_a[n:] = 0

            break

        # time-variable average atomic mass
        N_tot = y1 + y2 + y3 + y4 + y5 + y6 + y7
        mu = (y1*mu_H + y2*mu_He + y3*mu_D + y4*mu_O + y5*mu_C + y6*mu_N + y7*mu_S)/N_tot

        if rad_evol == False:
            radius_env = 0
            radius_atm = 0
            radius_p = radius_core
            if Rp_override != False:
                radius_core = Rp_override
                radius_env = 0
                radius_atm = 0
        else:
            radius_env = R_env(Mp, f_atm, Fp, t_a[n], thermal)
            radius_atm = R_atm(T, Mp, radius_core, radius_env, mu)
            radius_p = radius_core + radius_atm + radius_env
            radius_p = np.min([R_B, R_H, radius_p]) # limits Rp to the min of Bondi/Hill/Lopez+Fortney radius

        K = np.max([V_reduction(Mp, Mstar, d, radius_p), 0.01]) # grav potential reduction factor due to stellar tidal forces
        Vpot = K*G*Mp/radius_p
        A = 4*np.pi*radius_p**2

    # sets mass flux [kg/m2/s]
        if mechanism == 'XUV':
            if RR == True:
                phi = np.min([phi_RR(radius_p, Mp, T, t_a[n], F0, t0*s2yr, t_sat, beta, step_fn, F_final, t_pms, pms_factor), 
                              phi_E(t_a[n], eps, Vpot, d, F0, t0*s2yr, t_sat, beta, activity, flux_model, stellar_type, step_fn, F_final, t_pms, pms_factor)])
            else:
                phi = phi_E(t_a[n], eps, Vpot, d, F0, t0*s2yr, t_sat, beta, activity, flux_model, stellar_type, step_fn, F_final, t_pms, pms_factor)
        elif mechanism == 'CPML':
            phi = phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
        elif mechanism == 'phi kill':
            phi = phi_kill(Mp*f_atm, radius_p, t - t_a[n])
        elif mechanism == 'XUV+CPML':
            if RR == True:
                phi_XUV = np.min([phi_RR(radius_p, Mp, T, t_a[n], F0, t0*s2yr, t_sat, beta, step_fn, F_final, t_pms, pms_factor), 
                                  phi_E(t_a[n], eps, Vpot, d, F0, t0*s2yr, t_sat, beta, activity, flux_model, stellar_type, step_fn, F_final, t_pms, pms_factor)])
            else:
                phi_XUV = phi_E(t_a[n], eps, Vpot, d, F0, t0*s2yr, t_sat, beta, activity, flux_model, stellar_type, step_fn, F_final, t_pms, pms_factor)
            phi = phi_XUV + phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
        
        mass_loss = phi*A*delta_t
        g = G*Mp/radius_p**2
        H_H = R_gas*T/(M_H*g) # H scale height [m]
        H_He = R_gas*T/(M_He*g) # He scale height [m]
        H_D = R_gas*T/(M_D*g) # D scale height [m]
        H_O = R_gas*T/(M_O*g) # O scale height [m]
        H_C = R_gas*T/(M_C*g) # C scale height [m]
        H_N = R_gas*T/(M_N*g) # N scale height [m]
        H_S = R_gas*T/(M_S*g) # S scale height [m]

        x1 = y1/N_tot
        x2 = y2/N_tot
        x3 = y3/N_tot
        x4 = y4/N_tot
        x5 = y5/N_tot
        x6 = y6/N_tot
        x7 = y7/N_tot

        if dynamic_phi == False:
            if y1 + y2 == 0:
                X1 = 0
                X2 = 0
            else:
                X1 = y1/(y1+y2)
                X2 = y2/(y1+y2)
            MU = X1*mu_H + X2*mu_He
            Phi_H, phi_c = Phi_1(phi, b, H_H, H_He, mu_H, mu_He, X1, X2, MU, output = 1) # H number flux [atoms/s/m2]
            Phi_He = Phi_2(phi, b, H_H, H_He, mu_H, mu_He, X1, X2, MU) # He number flux [atoms/s/m2]
            Phi_D = Phi_D_Z90(Phi_H, Phi_He, H_H, H_D, H_He, y1, y2, y3, y4, y5, y6, y7, T) # D number flux [atoms/s/m2]
            Phi_O = Phi_O_Z90(Phi_H, Phi_He, H_H, H_O, H_He, y1, y2, y3, y4, y5, y6, y7, T) # O number flux [atoms/s/m2]
            Phi_C = Phi_C_Z90(Phi_H, Phi_He, H_H, H_C, H_He, y1, y2, y3, y4, y5, y6, y7, T) # C number flux [atoms/s/m2]
            Phi_N = Phi_N_Z90(Phi_H, Phi_He, H_H, H_N, H_He, y1, y2, y3, y4, y5, y6, y7, T) # N number flux [atoms/s/m2]
            Phi_S = Phi_S_Z90(Phi_H, Phi_He, H_H, H_S, H_He, y1, y2, y3, y4, y5, y6, y7, T) # S number flux [atoms/s/m2]

        elif dynamic_phi == True:
            N_values = [y1, y2, y3, y4, y5, y6, y7]  # [H, He, D, O, C, N, S]
            abundances_with_idx = [(i, N_values[i]) for i in range(7)]
            abundances_with_idx.sort(key=lambda x: x[1], reverse=True)
            
            most_abundant_idx = abundances_with_idx[0][0]
            second_most_abundant_idx = abundances_with_idx[1][0]
            
            # Get masses and scale heights for the two most abundant species
            scale_heights = [H_H, H_He, H_D, H_O, H_C, H_N, H_S]
            
            mass_most = atomic_masses[most_abundant_idx]
            mass_second = atomic_masses[second_most_abundant_idx] 
            H_most = scale_heights[most_abundant_idx]
            H_second = scale_heights[second_most_abundant_idx]

            # Determine which is lighter (species 1) and heavier (species 2) by MASS
            if mass_most <= mass_second:
                light_dominant_idx = most_abundant_idx      # species 1 (lighter)
                heavy_dominant_idx = second_most_abundant_idx # species 2 (heavier)
                mass_1 = mass_most
                mass_2 = mass_second
                H_1 = H_most
                H_2 = H_second
            else:
                light_dominant_idx = second_most_abundant_idx # species 1 (lighter)
                heavy_dominant_idx = most_abundant_idx        # species 2 (heavier)
                mass_1 = mass_second
                mass_2 = mass_most
                H_1 = H_second
                H_2 = H_most
            
            # Calculate molar fractions for the two dominant species
            N1 = N_values[light_dominant_idx]   # lightest dominant (species 1)
            N2 = N_values[heavy_dominant_idx]   # heaviest dominant (species 2)
            N_tot_binary = N1 + N2
            
            X1 = N1 / N_tot_binary  # molar fraction of species 1 in binary mixture
            X2 = N2 / N_tot_binary  # molar fraction of species 2 in binary mixture
            MU = X1 * mass_1 + X2 * mass_2
            
            # Calculate binary diffusion coefficient between the two dominant species
            light_name = species_names[light_dominant_idx]  # species 1
            heavy_name = species_names[heavy_dominant_idx]  # species 2
            
            b = get_binary_diffusion_coeff(light_name, heavy_name, T)
            
            # Calculate escape fluxes for the two dominant species
            Phi_1_calc, phi_c = Phi_1(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU, output=1)
            Phi_2_calc = Phi_2(phi, b, H_1, H_2, mass_1, mass_2, X1, X2, MU)
            
            # Assign fluxes to correct species based on light/heavy dominant indices
            if light_dominant_idx == 0:      # H is species 1 (lighter dominant)
                Phi_H = Phi_1_calc
            elif light_dominant_idx == 1:    # He is species 1
                Phi_He = Phi_1_calc
            elif light_dominant_idx == 2:    # D is species 1
                Phi_D = Phi_1_calc
            elif light_dominant_idx == 3:    # O is species 1
                Phi_O = Phi_1_calc
            elif light_dominant_idx == 4:    # C is species 1
                Phi_C = Phi_1_calc
            elif light_dominant_idx == 5:    # N is species 1
                Phi_N = Phi_1_calc
            elif light_dominant_idx == 6:    # S is species 1
                Phi_S = Phi_1_calc
                
            if heavy_dominant_idx == 0:      # H is species 2 (heavier dominant)
                Phi_H = Phi_2_calc
            elif heavy_dominant_idx == 1:    # He is species 2
                Phi_He = Phi_2_calc
            elif heavy_dominant_idx == 2:    # D is species 2
                Phi_D = Phi_2_calc
            elif heavy_dominant_idx == 3:    # O is species 2
                Phi_O = Phi_2_calc
            elif heavy_dominant_idx == 4:    # C is species 2
                Phi_C = Phi_2_calc
            elif heavy_dominant_idx == 5:    # N is species 2
                Phi_N = Phi_2_calc
            elif heavy_dominant_idx == 6:    # S is species 2
                Phi_S = Phi_2_calc
            
            # Calculate fluxes for remaining species using corrected generalized function
            for i in range(7):
                if i != light_dominant_idx and i != heavy_dominant_idx:
                    flux = Phi_minor_species(
                        Phi_1_calc, Phi_2_calc, H_1, H_2, scale_heights[i], 
                        N_values, T, i, light_dominant_idx, heavy_dominant_idx
                    )
                    # Assign to correct species
                    if i == 0:      # H
                        Phi_H = flux
                    elif i == 1:    # He
                        Phi_He = flux
                    elif i == 2:    # D
                        Phi_D = flux
                    elif i == 3:    # O
                        Phi_O = flux
                    elif i == 4:    # C
                        Phi_C = flux
                    elif i == 5:    # N
                        Phi_N = flux
                    elif i == 6:    # S
                        Phi_S = flux

        # record values
        Matm_a[n] = M_atm
        fatm_a[n] = f_atm
        Renv_a[n] = radius_env # this will still change even with Rp limited to min(R_Bondi, R_Hill)
        Rp_a[n] = radius_p
        Vpot_a[n] = Vpot
        phi_a[n] = phi
        phic_a[n] = phi_c
        Mloss_a[n] = mass_loss
        
        y1_a[n] = y1
        y2_a[n] = y2
        y3_a[n] = y3
        y4_a[n] = y4
        y5_a[n] = y5
        y6_a[n] = y6
        y7_a[n] = y7
        y1_a_int[n] = N_H_int
        y2_a_int[n] = N_He_int
        y3_a_int[n] = N_D_int
        y4_a_int[n] = N_O_int
        y5_a_int[n] = N_C_int
        y6_a_int[n] = N_N_int
        y7_a_int[n] = N_S_int
        x1_a[n] = x1
        x2_a[n] = x2
        x3_a[n] = x3
        x4_a[n] = x4
        x5_a[n] = x5
        x6_a[n] = x6
        x7_a[n] = x7
        Phi_H_a[n] = Phi_H
        Phi_He_a[n] = Phi_He
        Phi_D_a[n] = Phi_D
        Phi_O_a[n] = Phi_O
        Phi_C_a[n] = Phi_C
        Phi_N_a[n] = Phi_N
        Phi_S_a[n] = Phi_S

        ##### run atmodeller ######
        if n_atmodeller != 0: # save final molecular abundances on last time step
            if n == n_steps - 1:
                # atmod_full_output = {}
                atmod_sol = AtmodellerCoupler(T, Mp, radius_p, mu, melt_fraction_override, mantle_iron_dict,
                                                  y1+y3, y2, y4, y5, y6, y7, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, interior_atmosphere)[1]
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
                                                  y1+y3, y2, y4, y5, y6, y7, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, interior_atmosphere)
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
            else:
                H2_a[n] = H2_a[n-1]
                H2O_a[n] = H2O_a[n-1]
                O2_a[n] = O2_a[n-1]
                CO2_a[n] = CO2_a[n-1]
                CO_a[n] = CO_a[n-1]
                CH4_a[n] = CH4_a[n-1]
                N2_a[n] = N2_a[n-1]
                S2_a[n] = S2_a[n-1]
                H2O4S_a[n] = H2O4S_a[n-1]
                SO2_a[n] = SO2_a[n-1]
                H2_a_int[n] = H2_a_int[n-1]
                H2O_a_int[n] = H2O_a_int[n-1]
                O2_a_int[n] = O2_a_int[n-1]
                CO2_a_int[n] = CO2_a_int[n-1]
                CO_a_int[n] = CO_a_int[n-1]
                CH4_a_int[n] = CH4_a_int[n-1]
                N2_a_int[n] = N2_a_int[n-1]
                S2_a_int[n] = S2_a_int[n-1]
                H2O4S_a_int[n] = H2O4S_a_int[n-1]
                SO2_a_int[n] = SO2_a_int[n-1]
                fO2_a[n] = fO2_a[n-1]
        T_surf_analytic_a[n] = T_surf_analytic
        T_surf_atmod_a[n] = T_surf_atmod

        if y1 + y3 + N_H_int + N_D_int != 0: # needed to allow D and H to outgas from mantle
            X_DH = (y3 + N_D_int)/(y1 + y3 + N_H_int + N_D_int) # assumes D/H is in equilibrium between interior and atmosphere

        # advance to next step
        y1_loss = Phi_H*A*delta_t
        y2_loss = Phi_He*A*delta_t
        y3_loss = Phi_D*A*delta_t
        y4_loss = Phi_O*A*delta_t
        y5_loss = Phi_C*A*delta_t
        y6_loss = Phi_N*A*delta_t
        y7_loss = Phi_S*A*delta_t
        # M_atm -= mass_loss # comes from phi*A*delta_t
        M_atm -= (y1_loss*mu_H + y2_loss*mu_He + y3_loss*mu_D + y4_loss*mu_O + y5_loss*mu_C + y6_loss*mu_N + y7_loss*mu_S)
        f_atm = M_atm/Mp
        y1 -= y1_loss
        y2 -= y2_loss
        y3 -= y3_loss
        y4 -= y4_loss
        y5 -= y5_loss
        y6 -= y6_loss
        y7 -= y7_loss
        y1 = max(y1, 0)
        y2 = max(y2, 0)
        y3 = max(y3, 0)
        y4 = max(y4, 0)
        y5 = max(y5, 0)
        y6 = max(y6, 0)
        y7 = max(y7, 0)

    # save results
    solutions = {
    'time': t_a,
    'Rp': Rp_a,
    'Ratm': Renv_a,
    'Matm': Matm_a,
    'Vpot': Vpot_a,
    'fatm': fatm_a,
    'Mloss': Mloss_a,
    'phi': phi_a,
    'phic': phic_a,
    'N_H': y1_a,
    'N_He': y2_a,
    'N_D': y3_a,
    'N_O': y4_a,
    'N_C': y5_a,
    'N_N': y6_a,
    'N_S': y7_a,
    'N_H_int': y1_a_int,
    'N_He_int': y2_a_int,
    'N_D_int': y3_a_int,
    'N_O_int': y4_a_int,
    'N_C_int': y5_a_int,
    'N_N_int': y6_a_int,
    'N_S_int': y7_a_int,
    'x1': x1_a,
    'x2': x2_a,
    'x3': x3_a,
    'x4': x4_a,
    'x5': x5_a,
    'x6': x6_a,
    'x7': x7_a,
    'Phi_H': Phi_H_a,
    'Phi_He': Phi_He_a,
    'Phi_D': Phi_D_a,
    'Phi_O': Phi_O_a,
    'Phi_C': Phi_C_a,
    'Phi_N': Phi_N_a,
    'Phi_S': Phi_S_a,
    'T_surf_analytic': T_surf_analytic_a,
    'T_surf_atmod': T_surf_atmod_a
    }
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
