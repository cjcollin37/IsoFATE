'''
Collin Cherubim
October 3, 2022
Main code for running IsoFATE calculation for binary mixture
'''

from isofunks import *
from constants import *
from orbit_params import *
import numpy as np

def isocalc(f_atm, Mp, Mstar, F0, Fp, T, d, time = 5e9, mechanism = 'XUV', rad_evol = True,
mu = mu_solar, eps = 0.15, activity = 'medium', flux_model = 'power law', stellar_type = 'M1',
Rp_override = False, t_sat = 5e8, f_atm_final = 'null', n_TO_final = 'null', 
n_steps = int(1e6), t0 = 1e6, rho_rcb = 1.0, Johnson = False, RR = True, f_pred = False, thermal = True, 
beta = -1.23):
    '''
    Computes species abundances in ternary mixture of H, D, and He 
    via time-integrated numerical simulation of atmospheric escape.
    Note: species 1 is H, species 2 is He, and species 3 is D.

    Inputs:
     - f_atm: atmospheric mass fraction(s), must be in the form of an array [ndim]
     - Mp: planet mass [kg]
     - Mstar: stellar mass [kg]
     - F0: initial incident XUV flux [W/m2]
     - Fp: incident bolometric flux [W/m2]
     - T: planet equilibrium temperature [K]
     - d: orbtial distance [m]
     - time: total simulation time; scalar [yr]
     - mechanism: 'XUV', 'XUV+RR', 'CPML', 'XUV+CPML', 'fix phi subcritical',
     'fix phi supercritical', 'phi kill'
     - rad_evol: set to False to fix planet radius at core radius
     - mu: average atmospheric particle mass, default to H/He solar comp [kg]
     - eps: heat transfer efficiency [ndim]
     - activity: for Fxuv_hazmat function (uses semi-empirical data from MUSCLES survey); 
     'low' (lower quartile), 'medium' (median), 'high' (upper quartile)
     - flux_model: 'power law' for analytic power law, 'phoenix' for Fxuv_hazmat, 'Johnstone' for Fxuv_Johnstone
     - stellar_type: 'M1', 'K5', or 'G5' must be specified for flux_model == Fxuv_Johnstone
     - Rp_override: enter scalar planet radius value to manually set constant radius, note radius will not evolve [m]
     - t_sat: saturation time for Fxuv power law; 5e8 matches semi-empirical MUSCLES data [yr]
     - f_atm_final: final desired atmospheric mass fraction that sets lower bound on N_H to end simulation
     - n_TO_final: final desired terrestrial oceans worth of oxygen that sets lower bound on N_H to end simulation
     - adaptive_steps: sets error tolerance for Euler method w/ adaptive timesteps. Set to False for no adaptive steps
     - n_steps: number of timesteps. Convergence occurs at 1e6.
     - t0: simulation start time [yr]
     - rho_rcb: gas density at the RCB in CPML phi equatin [kg/m3]
     - Johnson: toggles Johnson et al 2013 phi efficiency reduction [True/False]
     - RR: toggles radiation-recombination effect (Ly alpha cooling; Murray-Clay et al 2009)  [True/False]
     - f_pred: toggles f_atm prediction based on planet mass; eq 6 Gupta & Schlichting 2022 [True/False]
     - thermal: toggles planet radius contraction in Lopez/Fortney equations (False removes age term) [True/Flase]
     - beta: exponential in Fxuv function; determines rate of XUV decrease. -1.23 consistent with MUSCLES data

    Output: Dictionary of 2-D arrays [len(f_atm) x n_steps] with keys,
     - 'time': simulation time array [s]
     - 'rp': total planet radius [m]
     - 'renv': convective atm depth [m]
     - 'mp': planet mass [kg]
     - 'vpot': gravitational potential at outer layer [J/kg]
     - 'fenv': total atmospheric mass fraction [ndim]
     - 'mloss': atm mass loss per time step [kg]
     - 'X3': D/H mole ratio [ndim]
     - 'X3_final': final D/H mole ratio for each f_atm [ndim] (1D array)
     - 'phi': atm mass flux [kg/m2/s]
     - 'phic': critical mass flux for species 2 escape [kg/m2/s]
     - 'N1': H number [atoms]
     - 'N2': He number [atoms]
     - 'N3': D number [atoms]
     - 'x1': H molar concentration [ndim]
     - 'x2': He molar concentration [ndim]
     - 'Phi1': H number flux [atoms/s/m2]
     - 'Phi2': He number flux [atoms/s/m2]
     - 'Phi3': D number flux [atoms/s/m2]
     - 'N1_final': final H number set by f_atm_final or n_TO_final [atoms]
    '''
###_____Ensure that f_atm is an array_____###
    if f_pred == 'f_pred2':
        f_atm = f_atm_pred2(Mp, T)
    elif f_pred == 'f_pred3':
        f_atm = f_atm_pred3(Mp, T)
    if type(f_atm) == int or type(f_atm) == float or type(f_atm) == np.float64:
        f_atm = np.array([f_atm])


###_____Initialize physical values_____###

    b = 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
    radius_core = R_core(Mp) # [m]
    R_B = R_Bondi(Mp, mu, T) # Bondi radius [m]
    R_H = R_Hill(Mp, Mstar, d) # Hill radius [m]

###_____Initialize timesteps_____###

    n_tot = n_steps # timesteps
    t0 = t0/s2yr # simulation start time at 1 Myr [s]
    t = time/s2yr - t0 # total simulation time [s]
    delta_t = t/n_tot # timestep [s]

###_____Initialize arrays and conditions_____###

    rp_a = np.zeros((len(f_atm), n_tot)) # total planet radius timeseries for each f_atm, previously 'radius_a'
    renv_a = np.zeros((len(f_atm), n_tot)) # envelope radius timeseries for each f_atm
    mass_a = np.zeros((len(f_atm), n_tot)) # atmospheric mass timeseries for each f_atm
    vpot_a = np.zeros((len(f_atm), n_tot)) # grav potential timeseries for each f_atm
    fenv_a = np.zeros((len(f_atm), n_tot)) # atm mass fraction timeseries for each f_atm
    mloss_a = np.zeros((len(f_atm), n_tot)) # atm mass lost timeseries for each f_atm
    phi_a = np.zeros((len(f_atm), n_tot)) # mass flux timeseries for each f_atm
    phic_a = np.zeros((len(f_atm), n_tot)) # critical mass flux timeseries for each f_atm
    x1_aa = np.zeros((len(f_atm), n_tot)) # mole fraction of light species for each f_atm
    x2_aa = np.zeros((len(f_atm), n_tot)) # mole fraction of heavy species for each f_atm
    phi1_a = np.zeros((len(f_atm), n_tot)) # light species number flux for each f_atm
    phi2_a = np.zeros((len(f_atm), n_tot)) # heavy species number flux for each f_atm
    phi3_a = np.zeros((len(f_atm), n_tot)) # heavy species number flux for each f_atm for ternary mix

    X3_a = np.zeros((len(f_atm), n_tot)) # D/H mole ratio timeseries for each f_atm
    N1_a = np.zeros((len(f_atm), n_tot)) # H number timeseries for each f_atm
    N2_a = np.zeros((len(f_atm), n_tot)) # He number timeseries for each f_atm
    N3_a = np.zeros((len(f_atm), n_tot)) # D number timeseries for each f_atm
    X3_final = np.zeros(len(f_atm)) # final D/H ratio for each f_atm

###_____Loop through f_atm values____###

    for i, f_atm in enumerate(f_atm):

    # initial values
        M_atm = Mp*f_atm # initial atmospheric mass [kg]
        mu_avg = mu_HHe # should be avg of H/D/He, need to check; small error expected
        He0 = (1/13.6)*M_atm/mu_avg # initial He number [atoms]
        H0 = (12.6/13.6)*(1 - DtoH_solar)*M_atm/mu_avg # initial H number [atoms]
        D0 =(12.6/13.6)*DtoH_solar*M_atm/mu_avg # initial D number [atoms]

        if rad_evol == True:
            if Rp_override == False:
                radius_env = R_env(Mp, f_atm, Fp, t0, thermal) # convective atm depth [m]
                radius_atm = R_atm(T, Mp, radius_core, radius_env, mu) # radiative atm depth [m]
                radius_p = np.min([R_B, R_H, radius_core + radius_atm + radius_env])
            else:
                raise ValueError('Cannot specify Rp_override value if rad_evol == True.')
        elif rad_evol == False:
            if Rp_override == False:
                radius_p = radius_core
                radius_env = 0
                radius_atm = 0
            else:
                radius_p = Rp_override
                radius_env = 0
                radius_atm = 0
        
        K = np.max([V_reduction(Mp, Mstar, d, radius_p), 0.01]) # grav potential reduction factor due to stellar tidal forces
        Vpot = K*G*Mp/radius_p # initial grav potential [J/kg]
        
        A = 4*np.pi*radius_p**2
        g = G*Mp/radius_p**2

        H_H = R_gas*T/(M_H*g) # H scale height [m]
        H_D = R_gas*T/(M_D*g) # D scale height [m]
        H_He = R_gas*T/(M_He*g) # He scale height [m]

    ###_____Set species values_____###

        mu1 = mu_H
        mu2 = mu_He
        M1 = M_H
        M2 = M_He
        H1 = H_H
        H2 = H_He
        N1 = H0
        N2 = He0
        N3 = D0

    # sets initial mass flux phi [kg/m2/s]
        if mechanism == 'XUV':
            if RR == True:
                phi = np.min([phi_RR(radius_p, Mp, T, t0, F0, t_sat), phi_E(t0, eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)])
            else:
                phi = phi_E(t0, eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)
            if Johnson == True:
                phi = Johnson_reduction(eps, F0, radius_p, Vpot)*phi
        elif mechanism == 'fix phi subcritical':
            phi_c = b*(N1/(N1+N2))*(mu2 - mu1)/H1
            phi = phi_c/2
        elif mechanism == 'fix phi critical':
            phi_c = b*(N1/(N1+N2))*(mu2 - mu1)/H1
            phi = phi_c
        elif mechanism == 'fix phi supercritical':
            phi_c = b*(N1/(N1+N2))*(mu2 - mu1)/H1
            phi = phi_c*10
        elif mechanism == 'CPML':
            phi = phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
        elif mechanism == 'phi kill':
            phi = phi_kill(Mp*f_atm, radius_p, t)
        elif mechanism == 'XUV+CPML':
            if RR == True:
                phi_XUV = np.min([phi_RR(radius_p, Mp, T, t0, F0, t_sat), phi_E(t0, eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)])
            else:
                phi_XUV = phi_E(t0, eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type) 
            phi = phi_XUV + phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
        else:
            raise ValueError('Loss mechanism unspecified. \n Please enter: mechanism = "XUV", "CPML", "XUV+CPML", "fix phi subcritical", or "fix phi supercritical".')
        
        mass_loss = phi*A*delta_t # mass lost in first time step [kg]

        y1 = 1*N1 # H number [atoms]
        y2 = 1*N2 # He number [atoms]
        y3 = 1*N3 # D number [atoms]
        x1 = N1/(N1 + N2 + N3) # molar concentration H [ndim]
        x2 = N2/(N1 + N2 + N3) # molar concentration He [ndim]

        Phi1 = Phi_1(phi, b, H1, H2, mu1, mu2, x1, x2) # H number flux [atoms/s/m2]
        Phi2 = Phi_2(phi, b, H1, H2, mu1, mu2, x1, x2) # He number flux [atoms/s/m2]
        Phi3 = Phi_D_Z90(Phi1, Phi2, H_D, H_He, y1, y2, y3, T) # D number flux [atoms/s/m2]
        phi_c = b*x1*(mu2 - mu1)/H1 # critical mass flux for He escape [kg/s/m2]

    ###_____Initialize arrays_____###

        t_a = delta_t*np.linspace(1, n_tot + 1, n_tot) + t0 # time array [s]
        Phi_a = np.zeros(n_tot) # mass flux array [kg/s/m2]
        Phic_a = np.zeros(n_tot) # critical mass flux array [kg/s/m2] 
        Rp_a = np.zeros(n_tot) # total radius, diagnostic [m]
        Renv_a = np.zeros(n_tot) # envelope radius [m]
        Matm_a = np.zeros(n_tot) # atmospheric mass [kg]
        fatm_a = np.zeros(n_tot) # atm mass fraction [ndim]
        Vpot_a = np.zeros(n_tot) # grav potential, diagnostic [J/kg]
        Mloss_a = np.zeros(n_tot) # mass lost per timestep [kg]

        y1_a = np.zeros(n_tot) # H number array [atoms]
        y2_a = np.zeros(n_tot) # He number array [atoms]
        y3_a = np.zeros(n_tot) # D number array [atoms]
        X3_aa = np.zeros(n_tot) # D/H mole ratio array
        x1_a = np.zeros(n_tot) # H molar concentration array [ndim]
        x2_a = np.zeros(n_tot) # He molar concentration array [ndim]
        Phi1_a = np.zeros(n_tot) # H number flux array [atoms/s/m2]
        Phi2_a = np.zeros(n_tot) # He number flux array [atoms/s/m2]
        Phi3_a = np.zeros(n_tot) # D number flux array [atoms/s/m2]

    ###_____Loop through timesteps_____###

        for n in range(n_tot):

    # record values
            Matm_a[n] = M_atm
            fatm_a[n] = f_atm
            Renv_a[n] = radius_env # this will still change even with Rp limited to min(R_Bondi, R_Hill)
            Rp_a[n] = radius_p
            Vpot_a[n] = Vpot
            Phi_a[n] = phi
            Phic_a[n] = phi_c
            Mloss_a[n] = mass_loss

            y1_a[n] = y1
            y2_a[n] = y2
            x1_a[n] = x1
            x2_a[n] = x2
            Phi1_a[n] = Phi1
            Phi2_a[n] = Phi2

            y3_a[n] = y3
            X3_aa[n] = y3/y1
            Phi3_a[n] = Phi3

            # advance to next step
            M_atm -= mass_loss

        ### stop simulation when lower bound on N_H is reached
            if n_TO_final != 'null':
                N1_final = TO(Mp, f_atm_final, n_TO_final)[0]
                if y1 <= N1_final:
                    Matm_a[n:] = Matm_a[n-1]
                    fatm_a[n:] = fatm_a[n-1]
                    Renv_a[n:] = Renv_a[n-1]
                    Rp_a[n:] = Rp_a[n-1]
                    Vpot_a[n:] = Vpot_a[n-1]
                    Phi_a[n:] = 0
                    Mloss_a[n:] = 0

                    y1_a[n:] = y1_a[n-1]
                    y2_a[n:] = y2_a[n-1]
                    y3_a[n:] = y3_a[n-1]
                    X3_aa[n:] = y3_a[n-1]/y1_a[n-1]
                    Phi3_a[n:] = 0
                    x1_a[n:] = x1_a[n-1]
                    x2_a[n:] = x2_a[n-1]
                    Phi1_a[n:] = 0
                    Phi2_a[n:] = 0
                    break

            elif f_atm_final != 'null':
                N1_final = TO(Mp, f_atm_final, n_TO_final)[0]
                if f_atm <= f_atm_final:
                    Matm_a[n:] = Matm_a[n-1]
                    fatm_a[n:] = fatm_a[n-1]
                    Renv_a[n:] = Renv_a[n-1]
                    if Rp_override != False:
                        Rp_a[n:] = Rp_override
                    else:
                        Rp_a[n:] = Rp_a[n-1]
                    Vpot_a[n:] = Vpot_a[n-1]
                    Phi_a[n:] = 0
                    Mloss_a[n:] = 0

                    y1_a[n:] = y1_a[n-1]
                    y2_a[n:] = y2_a[n-1]
                    y3_a[n:] = y3_a[n-1]
                    X3_aa[n:] = y3_a[n-1]/y1_a[n-1]
                    Phi3_a[n:] = 0
                    x1_a[n:] = x1_a[n-1]
                    x2_a[n:] = x2_a[n-1]
                    Phi1_a[n:] = 0
                    Phi2_a[n:] = 0
                    break

        ### Stop simulation when entire atmosphere is lost (if f_atm_final = 'null' and n_TO_final = 'null')
            if M_atm < 0 and f_atm_final == 'null' and n_TO_final == 'null':
                Matm_a[n:] = 0
                fatm_a[n:] = 0
                Renv_a[n:] = 0
                Rp_a[n:] = radius_core
                Vpot_a[n:] = K*G*Mp/radius_core
                Phi_a[n:] = 0
                Mloss_a[n+1:] = 0

                y1_a[n:] = y1
                y2_a[n:] = y2
                x1_a[n:] = x1
                x2_a[n:] = x2
                Phi1_a[n:] = 0
                Phi2_a[n:] = 0    
                y3_a[n:] = y3
                X3_aa[n:] = y3/y1
                Phi3_a[n:] = 0
                break

            f_atm = M_atm/Mp

            if rad_evol == False:
                radius_env = 0
                radius_atm = 0
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
                    phi = np.min([phi_RR(radius_p, Mp, T, t_a[n], F0, t_sat), phi_E(t_a[n], eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)])
                else:
                    phi = phi_E(t_a[n], eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)
                if Johnson == True:
                    phi = Johnson_reduction(eps, F0, radius_p, Vpot)*phi
            elif mechanism == 'CPML':
                phi = phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
                if Johnson == True:
                    Fp = Fxuv(t_a[n], F0, t_sat)
                    phi = Johnson_reduction(eps, Fp, radius_p, Vpot)*phi
            elif mechanism == 'phi kill':
                phi = phi_kill(Mp*f_atm, radius_p, t - t_a[n])
            elif mechanism == 'XUV+CPML':
                if RR == True:
                    phi_XUV = np.min([phi_RR(radius_p, Mp, T, t_a[n], F0, t_sat), phi_E(t_a[n], eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)])
                else:
                    phi_XUV = phi_E(t_a[n], eps, Vpot, d, F0, t_sat, beta, activity, flux_model, stellar_type)
                phi = phi_XUV + phiE_CP(T, Mp, rho_rcb, eps, Vpot, A, mu, radius_env)
            mass_loss = phi*A*delta_t
            g = G*Mp/radius_p**2
            H1 = R_gas*T/(M1*g) # H scale height [m]
            H2 = R_gas*T/(M2*g) # He scale height [m]
            H_D = R_gas*T/(M_D*g) # D scale height [m]
            
            y1 -= Phi1*A*delta_t
            y2 -= Phi2*A*delta_t
            y3 -= Phi3*A*delta_t

            x1 = y1/(y1 + y2 + y3)
            x2 = y2/(y1 + y2 + y3)

            Phi1 = Phi_1(phi, b, H1, H2, mu1, mu2, x1, x2)
            Phi2 = Phi_2(phi, b, H1, H2, mu1, mu2, x1, x2)
            Phi3 = Phi_D_Z90(Phi1, Phi2, H_D, H_He, y1, y2, y3, T)
            phi_c = b*x1*(mu2 - mu1)/H1

    # 2 dimensional arrays; f_atm x n_steps
        rp_a[i, :] = Rp_a[:] # total planet radius [m]
        renv_a[i, :] = Renv_a[:] # convective atm depth [m]
        mass_a[i, :] = Matm_a[:] # atmospheric mass [kg]
        vpot_a[i, :] = Vpot_a[:] # gravitational potential [J/kg]
        fenv_a[i, :] = fatm_a[:] # atmospheric mass fraction [ndim]
        mloss_a[i, :] = Mloss_a[:] # mass loss per timestep [kg]
        X3_a[i, :] = X3_aa[:] # D/H mole ratio [ndim]
        X3_final[i] = X3_aa[-1] # final D/H mole ratio for each f_atm (1D array)
        phi_a[i, :] = Phi_a[:] # mass flux [kg/2/m2]
        phic_a[i, :] = Phic_a[:] # critical mass flux [kg/s/m2]
        N1_a[i, :] = y1_a[:] # H number [atoms]
        N2_a[i, :] = y2_a[:] # He number [atoms]
        N3_a[i, :] = y3_a[:] # D number [atoms]
        x1_aa[i, :] = x1_a[:] # H molar concentration [ndim]
        x2_aa[i, :] = x2_a[:] # He molar concentration [ndim]
        phi1_a[i, :] = Phi1_a[:] # H number flux [atoms/s/m2]
        phi2_a[i, :] = Phi2_a[:] # He number flux [atoms/s/m2]
        phi3_a[i, :] = Phi3_a[:] # D number flux [atoms/s/m2]

    if f_atm_final != 'null' or n_TO_final != 'null':
        solutions = {
        'time': t_a,
        'rp': rp_a,
        'renv': renv_a,
        'menv': mass_a,
        'vpot': vpot_a,
        'fenv': fenv_a,
        'mloss': mloss_a,
        'X3': X3_a,
        'X3_final': X3_final,
        'phi': phi_a,
        'phic': phic_a,
        'N1': N1_a,
        'N2': N2_a,
        'N3': N3_a,
        'x1': x1_aa,
        'x2': x2_aa,
        'Phi1': phi1_a,
        'Phi2': phi2_a,
        'Phi3': phi3_a,
        'N1_final': N1_final
    }
    else:
        solutions = {
        'time': t_a,
        'rp': rp_a,
        'renv': renv_a,
        'menv': mass_a,
        'vpot': vpot_a,
        'fenv': fenv_a,
        'mloss': mloss_a,
        'X3': X3_a,
        'X3_final': X3_final,
        'phi': phi_a,
        'phic': phic_a,
        'N1': N1_a,
        'N2': N2_a,
        'N3': N3_a,
        'x1': x1_aa,
        'x2': x2_aa,
        'Phi1': phi1_a,
        'Phi2': phi2_a,
        'Phi3': phi3_a
    }

    return solutions
