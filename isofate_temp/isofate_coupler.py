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
from atmodeller.atmodeller import InteriorAtmosphere
from atmodeller.atmodeller import Species
from atmodeller.atmodeller import SpeciesCollection
# from atmodeller.atmodeller.solubility import get_solubility_models
solubility_models = get_solubility_models()

def isocalc(f_atm, Mp, Mstar, F0, Fp, T, d, time = 5e9, mechanism = 'XUV', rad_evol = True,
N_H = 0, N_He = 0, N_D = 0, N_O = 0, N_C = 0, melt_fraction_override = False,
mu = mu_solar, eps = 0.15, activity = 'medium', flux_model = 'power law', stellar_type = 'M1',
Rp_override = False, t_sat = 5e8, step_fn = False, F_final = 0, t_pms = 0, pms_factor = 1e2,
n_steps = int(1e5), t0 = 1e6, rho_rcb = 1.0, RR = True, thermal = True, 
beta = -1.23, n_atmodeller = int(1e2), save_molecules = False, mantle_iron_dict = False):
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

    ### atmodeller interior
    N_H_int = 0
    N_He_int = 0
    N_D_int = 0
    N_O_int = 0
    N_C_int = 0
    if n_atmodeller == 0:
        T_surf_analytic = 0
        T_surf_atmod = 0
    if N_H != 0 and N_D != 0: # needed to allow D and H to outgas from mantle
        X_DH = N_D/(N_H + N_D) # ignores D in mantle

    H2O_g = Species.create_gas("H2O_g", solubility=solubility_models["H2O_basalt_dixon95"])
    H2_g = Species.create_gas("H2_g", solubility=solubility_models["H2_basalt_hirschmann12"])
    O2_g = Species.create_gas("O2_g")
    CO_g = Species.create_gas("CO_g", solubility=solubility_models["CO_basalt_yoshioka19"])
    CO2_g = Species.create_gas("CO2_g", solubility=solubility_models["CO2_basalt_dixon95"])
    CH4_g = Species.create_gas("CH4_g", solubility=solubility_models["CH4_basalt_ardia13"])
    He_g = Species.create_gas("He_g", solubility=solubility_models["He_basalt_jambon86"])

    species = SpeciesCollection((H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g, He_g))

    interior_atmosphere = InteriorAtmosphere(species)

    atmod_full_output = {}
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
    y1_a_int = np.zeros(n_tot) # mantle H number array [atoms]
    y2_a_int = np.zeros(n_tot) # mantle He number array [atoms]
    y3_a_int = np.zeros(n_tot) # mantle D number array [atoms]
    y4_a_int = np.zeros(n_tot) # mantle O number array [atoms]
    y5_a_int = np.zeros(n_tot) # mantle C number array [atoms]
    H2_a = np.zeros(n_tot) # atmospheric H2 number array [molecules]
    H2O_a = np.zeros(n_tot) # atmospheric H2O number array [molecules]
    O2_a = np.zeros(n_tot) # atmospheric O2 number array [molecules]
    CO2_a = np.zeros(n_tot) # atmospheric CO2 number array [molecules]
    CO_a = np.zeros(n_tot) # atmospheric CO number array [molecules]
    CH4_a = np.zeros(n_tot) # atmospheric CH4 number array [molecules]
    fO2_a = np.zeros(n_tot) # fugacity array [bar]
    x1_a = np.zeros(n_tot) # H molar concentration array [ndim]
    x2_a = np.zeros(n_tot) # He molar concentration array [ndim]
    x3_a = np.zeros(n_tot) # D molar concentration array [ndim]
    x4_a = np.zeros(n_tot) # O molar concentration array [ndim]
    x5_a = np.zeros(n_tot) # C molar concentration array [ndim]
    Phi1_a = np.zeros(n_tot) # H number flux array [atoms/s/m2]
    Phi2_a = np.zeros(n_tot) # He number flux array [atoms/s/m2]
    Phi3_a = np.zeros(n_tot) # D number flux array [atoms/s/m2]
    Phi4_a = np.zeros(n_tot) # O number flux array [atoms/s/m2]
    Phi5_a = np.zeros(n_tot) # C number flux array [atoms/s/m2]
    T_surf_analytic_a = np.zeros(n_tot) # surface temperature from analytic calculation array [K]
    T_surf_atmod_a = np.zeros(n_tot) # atmodeller surface temperature array (capped at 6000 K) [K]

    ###_____Loop through timesteps_____###

    for n in range(n_tot):

    ### Stop simulation when entire atmosphere is lost
        if M_atm <= 0 or y1 + y2 + y3 + y4 + y5 <= 0:
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
            x1_a[n:] = x1_a[n-1] # x1_a[max(np.nonzero(x1_a)[0])] # get last non-zero value in array
            x2_a[n:] = x2_a[n-1] # x2_a[max(np.nonzero(x2_a)[0])]
            x3_a[n:] = x3_a[n-1] # x3_a[max(np.nonzero(x3_a)[0])]
            x4_a[n:] = x4_a[n-1] # x4_a[max(np.nonzero(x4_a)[0])]
            x5_a[n:] = x5_a[n-1] # x5_a[max(np.nonzero(x5_a)[0])]
            Phi1_a[n:] = 0
            Phi2_a[n:] = 0    
            Phi3_a[n:] = 0
            Phi4_a[n:] = 0
            Phi5_a[n:] = 0

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
                atmod_full_output['O2_fugacity'] = np.nan
                if save_molecules == True:
                    H2_a[n:] = 0
                    H2O_a[n:] = 0
                    O2_a[n:] = 0
                    CO2_a[n:] = 0
                    CO_a[n:] = 0
                    CH4_a[n:] = 0
                    fO2_a[n:] = 0

            break

        # time-variable average atomic mass
        N_tot = y1 + y2 + y3 + y4 + y5
        mu = (y1*mu_H + y2*mu_He + y3*mu_D + y4*mu_O + y5*mu_C)/N_tot

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
        H_C = R_gas*T/(M_C*g) # O scale height [m]

        x1 = y1/N_tot
        x2 = y2/N_tot
        x3 = y3/N_tot
        x4 = y4/N_tot
        x5 = y5/N_tot

        if y1 + y2 == 0:
            X1 = 0
            X2 = 0
        else:
            X1 = y1/(y1+y2)
            X2 = y2/(y1+y2)
        MU = X1*mu_H + X2*mu_He

        Phi1, phi_c = Phi_1(phi, b, H_H, H_He, mu_H, mu_He, X1, X2, MU, output = 1) # H number flux [atoms/s/m2]
        Phi2 = Phi_2(phi, b, H_H, H_He, mu_H, mu_He, X1, X2, MU) # He number flux [atoms/s/m2]
        Phi3 = Phi_D_Z90(Phi1, Phi2, H_H, H_D, H_He, y1, y2, y3, y4, y5, T) # D number flux [atoms/s/m2]
        Phi4 = Phi_O_Z90(Phi1, Phi2, H_H, H_O, H_He, y1, y2, y3, y4, y5, T) # O number flux [atoms/s/m2]
        Phi5 = Phi_C_Z90(Phi1, Phi2, H_H, H_C, H_He, y1, y2, y3, y4, y5, T) # C number flux [atoms/s/m2]

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
        y1_a_int[n] = N_H_int
        y2_a_int[n] = N_He_int
        y3_a_int[n] = N_D_int
        y4_a_int[n] = N_O_int
        y5_a_int[n] = N_C_int
        x1_a[n] = x1
        x2_a[n] = x2
        x3_a[n] = x3
        x4_a[n] = x4
        x5_a[n] = x5
        Phi1_a[n] = Phi1
        Phi2_a[n] = Phi2
        Phi3_a[n] = Phi3
        Phi4_a[n] = Phi4
        Phi5_a[n] = Phi5

        ##### run atmodeller ######
        if n_atmodeller != 0: # save final molecular abundances on last time step
            if n == n_steps - 1:
                # atmod_full_output = {}
                atmod_sol = AtmodellerCoupler(T, Mp, radius_p, mu, melt_fraction_override, mantle_iron_dict,
                                                  y1+y3, y2, y4, y5, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, interior_atmosphere)[1]
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
                atmod_full_output['He_mantle'] = atmod_sol['He_g']['dissolved_moles'][0]
                atmod_full_output['O2_fugacity'] = atmod_sol['O2_g']['fugacity'][0]
                atmod_full_output['log10dIW_1_bar'] = atmod_sol['O2_g']['log10dIW_1_bar'][0]
            if n%n_atmodeller == 0: # run atmodeller every n_atmodeller steps.
                atmod_results, atmod_full, mantle_iron_dict = AtmodellerCoupler(T, Mp, radius_p, mu, melt_fraction_override, mantle_iron_dict,
                                                  y1+y3, y2, y4, y5, N_H_int+N_D_int, N_He_int, N_O_int, N_C_int, interior_atmosphere)
                N_H_int = atmod_results['N_H_int']*(1 - X_DH)
                N_D_int = atmod_results['N_H_int']*X_DH
                N_He_int = atmod_results['N_He_int']
                N_O_int = atmod_results['N_O_int']
                N_C_int = atmod_results['N_C_int']
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
                    fO2_a[n] = atmod_full['O2_g']['fugacity'][0]
            else:
                H2_a[n] = H2_a[n-1]
                H2O_a[n] = H2O_a[n-1]
                O2_a[n] = O2_a[n-1]
                CO2_a[n] = CO2_a[n-1]
                CO_a[n] = CO_a[n-1]
                CH4_a[n] = CH4_a[n-1]
                fO2_a[n] = fO2_a[n-1]
        T_surf_analytic_a[n] = T_surf_analytic
        T_surf_atmod_a[n] = T_surf_atmod

        if y1 + y3 + N_H_int + N_D_int != 0: # needed to allow D and H to outgas from mantle
            X_DH = (y3 + N_D_int)/(y1 + y3 + N_H_int + N_D_int) # assumes D/H is in equilibrium between interior and atmosphere

        # advance to next step
        y1_loss = Phi1*A*delta_t
        y2_loss = Phi2*A*delta_t
        y3_loss = Phi3*A*delta_t
        y4_loss = Phi4*A*delta_t
        y5_loss = Phi5*A*delta_t
        # M_atm -= mass_loss # comes from phi*A*delta_t
        M_atm -= (y1_loss*mu_H + y2_loss*mu_He + y3_loss*mu_D + y4_loss*mu_O + y5_loss*mu_C)
        f_atm = M_atm/Mp
        y1 -= y1_loss
        y2 -= y2_loss
        y3 -= y3_loss
        y4 -= y4_loss
        y5 -= y5_loss

        y1 = max(y1, 0)
        y2 = max(y2, 0)
        y3 = max(y3, 0)
        y4 = max(y4, 0)
        y5 = max(y5, 0)
    

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
    'N_H_int': y1_a_int,
    'N_He_int': y2_a_int,
    'N_D_int': y3_a_int,
    'N_O_int': y4_a_int,
    'N_C_int': y5_a_int,
    'x1': x1_a,
    'x2': x2_a,
    'x3': x3_a,
    'x4': x4_a,
    'x5': x5_a,
    'Phi_H': Phi1_a,
    'Phi_He': Phi2_a,
    'Phi_D': Phi3_a,
    'Phi_O': Phi4_a,
    'Phi_C': Phi5_a,
    'T_surf_analytic': T_surf_analytic_a,
    'T_surf_atmod': T_surf_atmod_a
    }
    if save_molecules == True:
        solutions['n_H2_a'] = H2_a,
        solutions['n_H2O_a'] = H2O_a,
        solutions['n_O2_a'] = O2_a,
        solutions['n_CO2_a'] = CO2_a,
        solutions['n_CO_a'] = CO_a,
        solutions['n_CH4_a'] = CH4_a,
        solutions['fO2_a'] = fO2_a
    if n_atmodeller != 0:
        solutions['atmodeller_final'] = atmod_full_output

    return solutions
