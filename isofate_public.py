'''
Collin Cherubim
October 3, 2022
Main code for running IsoFATE calculation for binary mixture
'''

from isofunks import *
from constants import *
from orbit_params import *
import numpy as np

def isocalc(f_atm, Mp, Mstar, F0, Fp, T, d, time = 5e9, mechanism = 'XUV', species = 'H/D', rad_evol = True,
mu = mu_solar, eps = 0.15, activity = 'medium', flux_model = 'power law', stellar_type = 'M1',
Rp_override = False, t_sat = 5e8, f_atm_final = 'null', n_TO_final = 'null', 
n_steps = int(1e6), t0 = 1e6, rho_rcb = 1.0, Johnson = False, RR = True, f_pred = False, thermal = True, 
beta = -1.23):
    '''
    Computes species abundances in binary mixture of H/H2 and D/HD or H/H2 and He via 
    time-integrated numerical simulation of atmospheric escape.

    Inputs:
     - f_atm: atmospheric mass fraction(s) [atm_mass/planet_mass]; scalar or array
     - Mp: planet mass [kg]
     - Mstar: stellar mass [kg]
     - F0: initial incident XUV flux [W/m2]
     - Fp: incident bolometric flux [W/m2]
     - T: planet equilibrium temperature [K]
     - d: orbtial distance [m]
     - time: total simulation time; scalar [yr]
     - mechanism: 'XUV', 'XUV+RR', 'CPML', 'XUV+CPML', 'fix phi subcritical',
     'fix phi supercritical', 'phi kill'
     - species: Specify the species to be fractionated - 'H/D', 'H2/HD', 'H/He', or 'H2/He'
     - rad_evol: set to False to fix planet radius at core radius (default) or Rp_override value
     - mu: average atmospheric particle mass, default to H/He solar comp [kg]
     - eps: heat transfer efficiency [ndim]
     - activity: for Fxuv_hazmat function (uses semi-empirical data from MUSCLES survey); 
     'low' (lower quartile), 'medium' (median), 'high' (upper quartile)
     - flux_model: 'power law' for analytic power law, 'phoenix' for Fxuv_hazmat, 'Johnstone' for Fxuv_Johnstone
     - stellar_type: 'M1', 'K5', or 'G5' must be specified for flux_model == Fxuv_Johnstone
     - Rp_override: enter scalar planet radius value to manually set fixed radius. rad_evol must be False. [m]
     - t_sat: saturation time for flux_model = 'power law'; recommended: 5e8 for M, 2e8 for K, and 1e8 for G dwarfs [yr]
     - f_atm_final: final atmospheric mass fraction that sets lower bound on N_1 to end simulation
     - n_TO_final: final terrestrial oceans worth of oxygen that sets lower bound on N_1 to end simulation
     - n_steps: number of timesteps. Convergence occurs at 1e6.
     - t0: simulation start time [yr]
     - rho_rcb: gas density at the RCB in CPML phi equation [kg/m3]
     - Johnson: toggles Johnson et al 2013 phi efficiency reduction [True/False]
     - RR: toggles radiation-recombination effect (Ly alpha cooling; Murray-Clay et al 2009)  [True/False]
     - f_pred: toggles f_atm prediction based on planet mass; eq 6 Gupta & Schlichting 2022
        'f_pred2' for Ginzburg et al 2016 eq 18, 'f_pred3' for Ginzburg et al 2016 eq 24
     - thermal: toggles planet radius contraction due to cooling (False removes age term in Lop+Fort '14) [True/Flase]
     - beta: exponential in Fxuv function; determines rate of XUV decrease. -1.23 consistent with MUSCLES data +

    Output: Dictionary of 2-D arrays [len(f_atm) x n_steps] with keys,
     - 'time': simulation time array [s]
     - 'rp': total planet radius [m]
     - 'renv': convective atm depth [m]
     - 'mp': planet mass [kg]
     - 'vpot': gravitational potential at outer layer [J/kg]
     - 'fenv': total atmospheric mass fraction [ndim]
     - 'mloss': atm mass loss per time step [kg]
     - 'X2': N2/N1 mole ratio [ndim]
     - 'X2_final': final N2/N1 mole ratio for each f_atm [ndim] (1D array)
     - 'phi': atm mass flux [kg/m2/s]
     - 'phic': critical mass flux for species 2 escape [kg/m2/s]
     - 'N1': light species number [particles]
     - 'N2': heavy species number [particles]
     - 'x1': light species molar concentration [ndim]
     - 'x2': heavy species molar concentration [ndim]
     - 'Phi1': light species number flux [particles/s/m2]
     - 'Phi2': heavy species number flux [particles/s/m2]
     - 'N1_final': final species 1 number set by f_atm_final or n_TO_final [atoms]
    '''

###_____Ensure that f_atm is an array_____###

    if f_pred == 'f_pred2':
        f_atm = f_atm_pred2(Mp, T)
    elif f_pred == 'f_pred3':
        f_atm = f_atm_pred3(Mp, T)
    if type(f_atm) == int or type(f_atm) == float or type(f_atm) == np.float64:
        f_atm = np.array([f_atm])

###_____Initialize physical values_____###

    if species == 'H/D' or species == 'H2/HD':
        b = 4.48e19*T**0.75 # [molecules/m/s] from Genda 2008 for H2 in HD
    elif species == 'H/He' or species == 'H2/He':
        b = 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
    radius_core = R_core(Mp) # planet core (rocky component) radius [m]
    R_B = R_Bondi(Mp, mu, T) # Bondi radius [m]
    R_H = R_Hill(Mp, Mstar, d) # Hill radius [m]

###_____Initialize timesteps_____###

    n_tot = n_steps # timesteps
    t0 = t0/s2yr # simulation start time at 1 Myr [s]
    t = time/s2yr - t0 # total simulation time [s]
    delta_t = t/n_tot # timestep [s]

###_____Initialize arrays and conditions_____###

    rp_a = np.zeros((len(f_atm), n_tot)) # total planet radius timeseries for each f_atm
    renv_a = np.zeros((len(f_atm), n_tot)) # convective atm depth timeseries for each f_atm
    mass_a = np.zeros((len(f_atm), n_tot)) # atmospheric mass timeseries for each f_atm
    vpot_a = np.zeros((len(f_atm), n_tot)) # grav potential timeseries for each f_atm
    fenv_a = np.zeros((len(f_atm), n_tot)) # atm mass fraction timeseries for each f_atm
    mloss_a = np.zeros((len(f_atm), n_tot)) # atm mass lost per time step timeseries for each f_atm
    phi_a = np.zeros((len(f_atm), n_tot)) # mass flux timeseries for each f_atm
    phic_a = np.zeros((len(f_atm), n_tot)) # critical mass flux timeseries for each f_atm
    x1_aa = np.zeros((len(f_atm), n_tot)) # mole fraction of light species for each f_atm
    x2_aa = np.zeros((len(f_atm), n_tot)) # mole fraction of heavy species for each f_atm
    phi1_a = np.zeros((len(f_atm), n_tot)) # light species number flux for each f_atm
    phi2_a = np.zeros((len(f_atm), n_tot)) # heavy species number flux for each f_atm

    X2_a = np.zeros((len(f_atm), n_tot)) # N2/N1 mole ratio timeseries for each f_atm
    N2_a = np.zeros((len(f_atm), n_tot)) # species 2 particle count timeseries for each f_atm
    N1_a = np.zeros((len(f_atm), n_tot)) # species 1 particle count number timeseries for each f_atm
    X2_final = np.zeros(len(f_atm)) # final N2/N1 mole ratio for each f_atm

###_____Loop through f_atm values____###

    for i, f_atm in enumerate(f_atm):

        # initial values
        M_atm = Mp*f_atm # initial atmospheric mass [kg]
        if species == 'H/D' or species == 'H2/HD':
            mu_avg = (1 - DtoH_solar)*mu_H + DtoH_solar*mu_D # avg atomic mass [kg/particle]
            H0 = (1 - DtoH_solar)*M_atm/mu_avg # initial H number [atoms]
            D0 = DtoH_solar*M_atm/mu_avg # initial D number [atoms]
            H2_0 = (1/2)*(H0 - D0) # initial H2 number [molecules]
            HD_0 = D0 # initial HD number [molecules]
        elif species == 'H/He':
            mu_avg = mu_HHe # avg atomic mass [kg/particle]
            H0 = (12.6/13.6)*M_atm/mu_avg # initial H number [atoms]
            He0 = (1/13.6)*M_atm/mu_avg # initial He number [atoms]
        elif species == 'H2/He':
            mu_avg = mu_HHe # avg atomic mass [kg/particle]
            H0 = (12.6/13.6)*M_atm/mu_avg # initial H number [atoms]
            He0 = (1/13.6)*M_atm/mu_avg # initial He number [atoms]
            H2_0 = (1/2)*H0 # initial H2 number [molecules]

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

        H_H2 = R_gas*T/(M_H2*g) # H2 scale height [m]
        H_HD = R_gas*T/(M_HD*g) # HD scale height [m]
        H_H = R_gas*T/(M_H*g) # H scale height [m]
        H_D = R_gas*T/(M_D*g) # D scale height [m]
        H_He = R_gas*T/(M_He*g) # He scale height [m]

    ###_____Set species values_____###

        if species == 'H/D':
            mu1 = mu_H
            mu2 = mu_D
            M1 = M_H
            M2 = M_D
            H1 = H_H
            H2 = H_D
            N1 = H0
            N2 = D0
        elif species == 'H2/HD':
            mu1 = mu_H2
            mu2 = mu_HD
            M1 = M_H2
            M2 = M_HD
            H1 = H_H2
            H2 = H_HD
            N1 = H2_0
            N2 = HD_0
        elif species == 'H/He':
            mu1 = mu_H
            mu2 = mu_He
            M1 = M_H
            M2 = M_He
            H1 = H_H
            H2 = H_He
            N1 = H0
            N2 = He0
        elif species == 'H2/He':
            mu1 = mu_H2
            mu2 = mu_He
            M1 = M_H2
            M2 = M_He
            H1 = H_H2
            H2 = H_He
            N1 = H2_0
            N2 = He0

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
            phi = phi_c*8
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

        x1 = N1/(N1 + N2) # molar concentration light species [ndim]
        x2 = N2/(N1 + N2) # molar concentration heavy species [ndim]
        y1 = 1*N1 # light species number [atoms or molecules]
        y2 = 1*N2 # heavy species number [atoms or molecules]

        Phi1 = Phi_1(phi, b, H1, H2, mu1, mu2, x1, x2) # light species number flux [particles/s/m2]
        Phi2 = Phi_2(phi, b, H1, H2, mu1, mu2, x1, x2) # heavy species number flux
        phi_c = b*x1*(mu2 - mu1)/H1 # critical mass flux for heavy species escape [kg/s/m2]

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

        y1_a = np.zeros(n_tot) # light species number array [particles]
        y2_a = np.zeros(n_tot) # heavy species number array [particles]
        X2_aa = np.zeros(n_tot) # N2/N1 mole ratio [ndim]
        x1_a = np.zeros(n_tot) # light species molar concentration array [ndim]
        x2_a = np.zeros(n_tot) # heavy species molar concentration array [ndim]
        Phi1_a = np.zeros(n_tot) # light species number flux array [particles/s/m2]
        Phi2_a = np.zeros(n_tot) # heavy species number flux array [particles/s/m2]

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
            if species == 'H/D' or species == 'H/He':
                X2_aa[n] = y2/y1
            elif species == 'H2/HD':
                X2_aa[n] = y2/(2*y1 + y2) # because N_H = 2*N_H2, checked with D0/H0 = HD_0/(2*H2_0 + HD_0)
            elif species == 'H2/He':
                X2_aa[n] = y2/(2*y1)
            x1_a[n] = x1
            x2_a[n] = x2
            Phi1_a[n] = Phi1
            Phi2_a[n] = Phi2

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
                    if species == 'H/D' or species == 'H/He':
                        X2_aa[n:] = y2_a[n-1]/y1_a[n-1]
                    elif species == 'H2/HD' or species == 'H2/He':
                        X2_aa[n:] = y2_a[n-1]/(2*y1_a[n-1]) 
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
                    if rad_evol == False:
                        Rp_a[n:] = radius_core
                    if Rp_override != False:
                        Rp_a[n:] = Rp_override
                    else:
                        Rp_a[n:] = Rp_a[n-1]
                    Vpot_a[n:] = Vpot_a[n-1]
                    Phi_a[n:] = 0
                    Mloss_a[n:] = 0

                    y1_a[n:] = y1_a[n-1]
                    y2_a[n:] = y2_a[n-1]
                    if species == 'H/D' or species == 'H/He':
                        X2_aa[n:] = y2_a[n-1]/y1_a[n-1]
                    elif species == 'H2/HD' or species == 'H2/He':
                        X2_aa[n:] = y2_a[n-1]/(2*y1_a[n-1]) 
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
                if species == 'H/D' or species == 'H/He':
                    X2_aa[n:] = y2/y1
                elif species == 'H2/HD' or species == 'H2/He':
                    X2_aa[n:] = y2/(2*y1)
                x1_a[n:] = x1
                x2_a[n:] = x2
                Phi1_a[n:] = 0
                Phi2_a[n:] = 0
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
            H1 = R_gas*T/(M1*g) # light species scale height [m]
            H2 = R_gas*T/(M2*g) # heavy species scale height [m]
            
            y1 -= Phi1*A*delta_t
            y2 -= Phi2*A*delta_t
            x1 = y1/(y1 + y2)
            x2 = y2/(y1 + y2)
            Phi1 = Phi_1(phi, b, H1, H2, mu1, mu2, x1, x2)
            Phi2 = Phi_2(phi, b, H1, H2, mu1, mu2, x1, x2)
            phi_c = b*x1*(mu2 - mu1)/H1

    # 2 dimensional arrays; f_atm x n_steps
        rp_a[i, :] = Rp_a[:] # total planet radius [m]
        renv_a[i, :] = Renv_a[:] # convective atm depth [m]
        mass_a[i, :] = Matm_a[:] # atmospheric mass [kg]
        vpot_a[i, :] = Vpot_a[:] # gravitational potential [J/kg]
        fenv_a[i, :] = fatm_a[:] # atmospheric mass fraction [ndim]
        mloss_a[i, :] = Mloss_a[:] # mass loss per timestep [kg]
        X2_a[i, :] = X2_aa[:] # N2/N1 mole ratio [ndim]
        X2_final[i] = X2_aa[-1] # final N2/N1 mole ratio for each f_atm (1D array)
        phi_a[i, :] = Phi_a[:] # mass flux [kg/2/m2]
        phic_a[i, :] = Phic_a[:] # critical mass flux [kg/s/m2]
        N1_a[i, :] = y1_a[:] # light species number [particles]
        N2_a[i, :] = y2_a[:] # heavy species number [particles]
        x1_aa[i, :] = x1_a[:] # light species molar concentration [ndim]
        x2_aa[i, :] = x2_a[:] # heavy species molar concentration [ndim]
        phi1_a[i, :] = Phi1_a[:] # light species number flux [particles/s/m2]
        phi2_a[i, :] = Phi2_a[:] # heavy species number flux [particles/s/m2]

    if f_atm_final != 'null' or n_TO_final != 'null':
        solutions = {
        'time': t_a,
        'rp': rp_a,
        'renv': renv_a,
        'menv': mass_a,
        'vpot': vpot_a,
        'fenv': fenv_a,
        'mloss': mloss_a,
        'X2': X2_a,
        'X2_final': X2_final,
        'phi': phi_a,
        'phic': phic_a,
        'N1': N1_a,
        'N2': N2_a,
        'x1': x1_aa,
        'x2': x2_aa,
        'Phi1': phi1_a,
        'Phi2': phi2_a,
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
        'X2': X2_a,
        'X2_final': X2_final,
        'phi': phi_a,
        'phic': phic_a,
        'N1': N1_a,
        'N2': N2_a,
        'x1': x1_aa,
        'x2': x2_aa,
        'Phi1': phi1_a,
        'Phi2': phi2_a
    }

    return solutions
