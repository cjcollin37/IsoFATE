'''
Collin Cherubim
October 3, 2022
IsoFATE functions
'''

import imports
from constants import *
from orbit_params import *
import matplotlib.patches as patches

# incident XUV flux

def Fxuv(t, F0, t_sat = 5e8, beta = -1.23):
    '''
    Calculates incident XUV flux
    Adapted from Ribas et al 2005
    Consistent with empirical data from MUSCLES spectra for early M dwarfs

    Inputs:
        - t: time/age [s]
        - F0: initial incident XUV flux [W/m2]
        - t_sat: saturation time [yr]; change this for different stellar types (M1:500Myr, G:50Myr)
        - beta: exponential term [ndim]

    Output: incident XUV flux [W/m2]
    '''
    if t*s2yr < t_sat:
        return F0
    else:
        return F0*(t*s2yr/t_sat)**beta

def Fxuv_a(t, F0, t_sat = 5e8, beta = -1.23):
    '''
    Calculates incident XUV flux
    Adapted from Ribas et al 2005
    Consistent with empirical data from MUSCLES spectra for early M dwarfs

    Inputs:
        - t: time/age [s; array]
        - F0: initial incident XUV flux [W/m2]
        - t_sat: saturation time [yr]; change this for different stellar types (M1:500Myr, G:50Myr)
        - beta: exponential term [ndim]
    Output: incident XUV flux [W/m2]
    '''
    output = np.zeros(len(t))

    for i in range(len(t)):    
        if t[i]*s2yr < t_sat:
            output[i] = F0
        else:
            output[i] =  F0*(t[i]*s2yr/t_sat)**beta
    
    return output

def Fxuv_Johnstone(t, d, stellar_type):
    '''
    Calculates incident XUV flux
    Adapted from Johnstone et al 2021 semi-empirical XUV tracks

    Inputs:
       - t: time/age [s]
       - d: orbital distance [m]
       - stellar_type: 'M1', 'K5', 'G5' [str]
    Output: incident XUV flux [W/m2]
    '''

    if stellar_type == 'M1':
        path = '/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/0p5Msun_50percentile_basic.dat'
    elif stellar_type == 'K5':
        path = '/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/0p7Msun_50percentile_basic.dat'
    elif stellar_type == 'G5':
        path = '/Users/collin/Documents/Harvard/Research/atm_escape/RotationXUVTracks/TrackGrid_MstarPercentile/1p0Msun_50percentile_basic.dat'
    else:
        print('Stellar type not supported. Please input "M1", "K5", or "G5"')

    data = np.loadtxt(path, unpack = True)
    age = data[0]*1e6/s2yr # [s]
    L_EUV = (data[4] + data[5] + data[6])*erg2joule # [W]
    F_EUV = L_EUV/(4*np.pi*d**2) # [W/m2]

    return np.interp(t, age, F_EUV)

def Fxuv_SF(t):
    '''
    Calculates incident XUV flux
    Adapted from Sanz-Forcada et al 2011 semi-empirical study for M to F stars

    Inputs:
       - t: time/age [s]
       - d: orbital distance [m]
    Output: incident XUV flux [W/m2]
    '''
    L_EUV = 10**(22.12 - 1.24*np.log10(t*s2yr/1e9))
    return L_EUV

def Fxuv_Ribas(t):
    '''
    Calculates XUV luminosity
    Adapted from Ribas et al 2005 Sun in time program (eq 1)

    Inputs:
       - t: time/age [s]
    Output: incident XUV flux [W/m2]
    '''
    tau = t*s2yr/1e9
    F = 29.7*tau**-1.23
    F = F*cgs2si_flux # [W/m2]
    L = F*4*np.pi*au2m**2
    return L # [W]


# semi-empirical incident XUV flux based on MUSCLES spectrum

def Fxuv_hazmat(t, d, activity):
    '''
    Semi-empirical XUV flux estimates based on HAZMAT and MUSCLES
    programs. Consistent with Fxuv power law approximation.

    Inputs: t (time, s); d (orbital distance, m); activity ('high', 'medium', 'low')
    Outputs: incident planetary XUV flux [W/m2]
    '''
    flux_10myr_uq = 45400 # [W/m2]
    flux_10myr_med = 41681
    flux_10myr_lq = 30355
    flux_45myr_uq = 36661
    flux_45myr_med = 3661
    flux_45myr_lq = 30288
    flux_120myr_uq = 47629
    flux_120myr_med = 21601
    flux_120myr_lq = 21601
    flux_650myr_uq = 48673
    flux_650myr_med = 20213
    flux_650myr_lq = 4144
    flux_5000myr_uq = 3964
    flux_5000myr_med = 1146
    flux_5000myr_lq = 1100

    if activity == 'high':
        if t*s2yr/1e6 < 10:
            return flux_10myr_uq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 45:
            return flux_45myr_uq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 120:
            return flux_120myr_uq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 650:
            return flux_650myr_uq*(0.515*Rs)**2/(d**2)
        else:
            return flux_5000myr_uq*(0.515*Rs)**2/(d**2)
    elif activity == 'medium':
        if t*s2yr/1e6 < 10:
            return flux_10myr_med*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 45:
            return flux_45myr_med*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 120:
            return flux_120myr_med*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 650:
            return flux_650myr_med*(0.515*Rs)**2/(d**2)
        else:
            return flux_5000myr_med*(0.515*Rs)**2/(d**2)
    elif activity == 'low':
        if t*s2yr/1e6 < 10:
            return flux_10myr_lq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 45:
            return flux_45myr_lq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 120:
            return flux_120myr_lq*(0.515*Rs)**2/(d**2)
        elif t*s2yr/1e6 < 650:
            return flux_650myr_lq*(0.515*Rs)**2/(d**2)
        else:
            return flux_5000myr_lq*(0.515*Rs)**2/(d**2)

# energy-limited total mass flux

def phi_E(t, eps, Vpot, d, F0, t_sat = 5e8, beta = -1.23, activity = 'medium', 
          flux_model = 'power law', stellar_type = 'M1'):
    '''
    Calculates energy-limited mass flux for XUV-friven hydrodynamic escape
    Adapted from Wordsworth et al. 2018

    Inputs:
        - t: system age [s]
        - eps: efficiency factor [ndim]
        - Vpot: planetary grav. potential [J/kg]
        - d: orbital distance [m]
        - F0: initial incident XUV flux [W/m2]
        - t_sat: Fxuv saturation time [yr]
        - beta: Fxuv exponential term (determines how quickly Fxuv decays)
        - activity: 'low', 'medium', or 'high'
        - flux model: 'power law' or 'phoenix'

    Output: mass flux [kg/m2/s]
    '''
    if flux_model == 'power law':
        return eps*Fxuv(t, F0, t_sat, beta)/(4*Vpot)
    elif flux_model == 'phoenix':
        return eps*Fxuv_hazmat(t, d, activity = 'medium')/(4*Vpot)
    elif flux_model == 'Johnstone':
        return eps*Fxuv_Johnstone(t, d, stellar_type)

# core-powered mass loss mass flux

def phiE_CP(Teq, Mp, rho_rcb, eps, Vpot, area, mu, R_env):
    '''
    Atmospheric mass flux for core-powered mass loss scenario
    Adapted from Gupta & Schlicting 2020

    Inputs:
        - Teq: planetary equilibrium temperature [K]
        - Mp: planetary mass [kg]
        - rho_rcb: density at the RCB [kg/m3]; ref value =1 kg/m3 from eq 7 Gupta & Schlichting 2020
        - eps: heat transfer efficiency factor [ndim]
        - Vpot: planetary gravitational potential [J/kg]
        - area: planetary surface area [m2]
        - mu: average particle mass [kg]
    Output: mass flux [kg/m2/s]
    '''
    R_c = R_core(Mp)
    V_pot = G*Mp/R_c
    gamma = 7/5 # adiabatic index for H2
    R_B = (gamma - 1)*G*Mp*mu/(gamma*kb*Teq) # Bondi radius
    kappa = 0.01 # opacity at RCB [m2/kg]; Ginzburg et al. 2016 and eq 7 Gupta & Schlichting 2020 (Freedman et al 2008)
    L = 64*np.pi*sbc*Teq**4*R_B/(3*kappa*rho_rcb) # planetary core luminosity
    phi_L = L/(V_pot*area) # mass flux

    c_s = np.sqrt(kb*Teq/mu_H) # sound speed [m/s]
    R_rcb = R_c + R_env # rcb radius [m]
    phi_B = c_s*rho_rcb*np.exp(-G*Mp/(c_s**2*R_rcb)) # Bondi-limited escape (eq 10 Gupta & Schlichting 2020)

    return min(phi_L, phi_B)

# number flux of light species

def Phi_1(phi, b, H1, H2, m1, m2, x1, x2):
    '''
    Calculates number flux for light species in binary gas mixture
    undergoing atm escape.
    Adapted from Wordsworth et al. 2018

    Inputs:
        - phi: mass flux [kg/m2/s]
        - b: binary diffusion coefficient [particles/m/s]
        - H1/H2: scale heights of light/heavy species [m]
        - m1/m2: molecular mass of light/heavy species [kg/particle]
        - x1/x2: molar concentration of light/heavy species (x1=mol_1/mol_tot) [ndim]
    Output: number flux of light species [particles/m2/s]
    '''
    phi_c = b*x1*(m2 - m1)/H1 # critical mass flux [kg/s/m2]
    phi_d2 = b/H2
    mbar = m1*x1 + m2*x2 # mean particle mass

    if phi < phi_c:
        return phi/m1
    else:
        return (x1*phi + x1*x2*(m2 - m1)*phi_d2)/mbar


# number flux of heavy species

def Phi_2(phi, b, H1, H2, m1, m2, x1, x2):
    '''
    Calculates number flux for heavy species in binary gas mixture
    undergoing atm escape.
    Adapted from Wordsworth et al. 2018

    Inputs:
        - (all phi_E)
        - b: binary diffusion coefficient [particles/m/s]
        - H1/H2: scale heights of light/heavy species [m]
        - m1/m2: molecular mass of light/heavy species [kg/particle]
        - x1/x2: molar concentration of light/heavy species (x1=mol_1/mol_tot) [ndim]
    Output: number flux of heavy species [particles/m2/s]
    '''
    phi_c = b*x1*(m2 - m1)/H1 # critical mass flux
    phi_d1 = b/H1 # equivalent diffusion fluxes
    mbar = m1*x1 + m2*x2 # mean particle mass

    if phi < phi_c:
        return 0
    else:
        return (x2*phi + x1*x2*(m1 - m2)*phi_d1)/mbar
    
# number flux deuterium Gu & Chen 2023

def Phi_D_GC23(Phi_H, Phi_He, H_D, H_He, N_H, N_He, N_D, T):
    '''
    Calculates number flux of deuterium for simultaneous calculation of H/He/D escape
    From Gu & Chen 2023

    Inputs:
        - Phi_i: number flux [particles/m2/s]
        - H_i: scale height [m]
        - N_i: particles of species i
        - T: eq temp [K]
    '''
    b_H_D = 7.183e19*T**0.728 # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_He = 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_He_D = 5.087e19*T**0.728 # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription
    alpha_2 = b_H_D/b_H_He
    alpha_3 = b_H_D/b_He_D
    Phi_DL_D = b_H_D/H_D
    Phi_DL_He = b_H_He/H_He
    X_He = N_He/(N_H + N_He + N_D)
    X_H = N_H/(N_H + N_He + N_D)
    X_D = N_D/(N_H + N_He + N_D)
    num = Phi_H - Phi_DL_D + alpha_2*Phi_DL_He*X_He + alpha_3*Phi_He
    denom = X_H + alpha_3*X_He
    return X_D*num/denom

# number flux deuterium derived from Zahnle et al 1990

def Phi_D_Z90(Phi_H, Phi_He, H_D, H_He, N_H, N_He, N_D, T):
    '''
    Calculates number flux of deuterium for simultaneous calculation of H/He/D escape
    Derived from Zahnle et al 1990 starting w/ their Eq (17)

    Inputs:
        - Phi_i: number flux [particles/m2/s]
        - H_i: scale height [m]
        - N_i: particles of species i
        - T: eq temp [K]
    '''
    b_H_D = 7.183e19*T**0.728 # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    b_H_He = 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
    b_He_D = 5.087e19*T**0.728 # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription
    alpha_2 = b_H_D/b_H_He
    alpha_3 = b_H_D/b_He_D
    Phi_DL_D = b_H_D/H_D
    Phi_DL_He = b_H_He/H_He
    x_He = N_He/(N_H + N_He + N_D)
    f_He = N_He/N_H
    f_D = N_D/N_H
    num = Phi_H - Phi_DL_D + alpha_2*Phi_DL_He*x_He + alpha_3*Phi_He
    denom = 1 + alpha_3*f_He
    return max(0, f_D*num/denom)


def Phi_D_Z90_mod(Phi_H, H_D, N_D, N_H, T):
    '''
    Phi_D_Z90 solution with He set to zero
    '''
    b_H_D = 7.183e19*T**0.728 # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
    Phi_DL_D = b_H_D/H_D
    f_D = N_D/N_H
    # return f_D*(Phi_H - Phi_DL_D)
    return f_D*Phi_H - (N_D/(N_D+N_H))*Phi_DL_D


#####_____ Lopez & Fortney 2014 thermal evolution equations _____#####

# planetary core radius

def R_core(Mp):
    '''
    Calculates planetary core radius (rocky component)
    Adapted from Lopez & Fortney 2014

    Input: planetary mass [kg]
    Output: planetary core radius [m]
    '''
    return Re*(Mp/Me)**0.25 # Re not in paper, typo


# planetary atmosphere radius

def R_atm(Teq, Mp, R_core, R_env, mu):
    '''
    Calculates radius of radiative atmosphere above RCB (stratosphere)
    Adapted from Lopez & Fortney 2014

    Inputs:
        - Teq: planet equilibrium temperature [K]
        - Mp: planet mass [kg]
        - Rc: core radius [m]
        - Renv: envelope radius [m]
        - mu: mean molecular mass [kg/particle]

    Outputs: radiative atmosphere radius [m]
    '''
    g = G*Mp/((R_core + R_env)**2) # field strength at base of atm
    H = kb*Teq/(g*mu) # scale height
    return 9*H

# planetary envelope radius

def R_env(Mp, f_env, Fp, age, thermal = True):
    '''
    Calculates radius of lower convective envelope (troposphere)
    Adapted from Lopez & Fortney 2014
    R_env = R_p - R_core - R_atm

    Inputs:
      - Mp: planet mass [kg]
      - f_env: envelope mass fraction [ndim]
      - Fp: incident bolometric flux [W/m2]
      - age: age [s]
      - thermal: toggles radius dependence on thermal evolution [True/False]

    Outputs: R_env: radius of the H/He envelope [m]
    '''
    c1 = Mp/Me # Me = Earth mass [kg]
    c2 = f_env/0.05
    c3 = Fp/Fe # Fe = Earth incident bolometric flux [W/m2]
    if thermal == True:
        c4 = (age*s2yr/5e9)
    elif thermal == False:
        c4 = 1
    R_env = 2.06*Re*c1**(-0.21)*c2**(0.59)*c3**(0.044)*c4**(-0.18)
    return R_env


# atmospheric mass fraction

def f_env(R_core, R_env, Rp, Mp, Teq, mu, Fp, t, thermal = True):
    '''
    Calculates planetary atmospheric mass fraction
    by rearrangement of R_core, R_atm, and R_env equations
    Adapted from Lopez & Fortney 2014

    Inputs:
      - R_core: planet core radius [m]
      - R_env: planet envelope radius (convective part) [m]
      - Rp: total planet radius [m]
      - Mp: planet mass [kg]
      - Teq: planet equilibrium temperature [K]
      - mu: mean molecular mass [kg/particle]
      - Fp: incident bolometric flux [W/m2]
      - t: age [s]
      - thermal: toggles radius dependence on thermal evolution [True/False]

    Outputs: f_env: mass fraction of the H/He envelope relative to total mass [ndim]
    '''
    t = t*s2yr/1e9
    if thermal == True:
        return np.real(0.05*((Rp - Re*(Mp/Me)**0.25 - R_atm(Teq, Mp, R_core, R_env, mu))*(1/(2.06*Re))*(Mp/Me)**(0.21)*(Fp/Fe)**(-0.044)*(t/5)**(0.18))**(1/0.59))
    elif thermal == False:
        return np.real(0.05*((Rp - Re*(Mp/Me)**0.25 - R_atm(Teq, Mp, R_core, R_env, mu))*(1/(2.06*Re))*(Mp/Me)**(0.21)*(Fp/Fe)**(-0.044))**(1/0.59))


# For outlining colormap elements for completely stripped planets

def Patch(nan_array, periods, Mps, ax, color = 'darkviolet', lw = 1.2):
    '''
    For plotting. Outlines cells in red for which all atmosphere is lost.
    Input: nan_array must contain 2D array of fractionation values and == 1 for total loss.
    '''
    x_step = periods[1]*s2day - periods[0]*s2day
    y_step = Mps[1]/Me - Mps[0]/Me
    width = x_step
    height = y_step

    for j in range(len(nan_array)):
        for i in range(len(nan_array[j])):
            if nan_array[j, i] == 1:
                x = periods[i]*s2day - x_step/2
                y = Mps[j]/Me - y_step/2
                patch = patches.Rectangle((x,y), width, height, facecolor = 'none', edgecolor = color, 
                                          linewidth = lw, zorder = 20)
                ax.add_patch(patch)



# Computes number of terrestrial oceans based on planet mass for setting lower bound on N_H to end simulation

def TO(Mp, f_atm = 'null', n_TO = 'null'):
    '''
    Calculates the hydrogen atom number from final desired f_atm or number of terrestrial oceans.
    If f_atm is specificed, fn computes H remaining assuming all is bonded to oxygen in envelope
    assuming solar abundance. If n_TO is specified, fn computes the same for the given value of n_TO.
    All values assume solar abundance values from Lodders 2003
    *** Specify only one: f_atm OR n_TO!
    Inputs:
        - Mp: planet mass [kg]
        - f_atm: final envelope mass fraction [ndim]
        - n_TO: number of terrestrial oceans remaining on planet
    Output: 
        - array[0] = number of hydrogen atoms remaining on planet
        - array[1] = number of terrestrial oceans remaining on planet
    '''
    
    n_OperTO = 7.83e22 # mols O per TO

    if n_TO == 'null' and f_atm == 'null':
        print('Error: Must specify value for either f_atm or n_TO')
        return None

    elif n_TO == 'null':
        n_H = 0.7491*Mp*f_atm/M_H # mols of H in envelope
        n_O = 4.899e-4*n_H # mols of O in envelope (Lodders solar abundance)
        n_TO = n_O/n_OperTO # number of terrestrial oceans worth of oxygen in system
        N_H = 2*n_O*avogadro # atoms of H in envelope

    elif f_atm == 'null':
        n_O = n_TO*n_OperTO # mols of O in envelope
        N_H = 2*n_O*avogadro # atoms of H in envelope

    return np.array([N_H, n_TO])

def R_Bondi(Mp, mu, Teq, gamma = 7/5):
    '''
    Bondi radius calculation

    Inputs:
        - gamma: adiabatic index (heat capacity ratio) [ndim]
        - Teq: planetary equilibrium temperature [K]
        - Mp: planetary mass [kg]
        - mu: average particle mass [kg]
    Output: Bondi radius [m]
    '''
    R_B = (gamma - 1)*G*Mp*mu/(gamma*kb*Teq) # Bondi radius
    return R_B

def R_Hill(Mp, Mstar, a):
    '''
    Hill radius calculation

    Inputs:
        - Mp: planetary mass [kg]
        - Mstar: stellar mass [kg]
        - a: orbital distance [m]
    Output: Hill radius [m]
    '''
    R_H = a*(Mp/3/Mstar)**(1/3)
    return R_H

def phi_kill(M_atm, Rp, age):
    '''
    Calculates mass escape flux to ensure removal of entire atmosphere
    Use with caution, work in progress
    Inputs:
        - M_atm: atmospheric mass fraction; Mp*f_atm [ndim]
        - Rp: planet radius [m]
        - age: system age/simulation time [s]
    Output: mass escape flux [kg/m2/s]
    '''
    A = 4*np.pi*Rp**2
    return M_atm/A/age*10

def F0_kill(Mp, Rp, M_atm, age, eps = 0.15):
    C = 0.893818 # integral of power law portion of F_XUV function
    Vpot = G*Mp/Rp
    phi = phi_kill(M_atm, Rp, age)
    return 40*Vpot*phi/eps - 2*C

def b_H2_HD(T):
    '''
    Input: T, temperature [K]
    Output: Binary diffusion coefficient for H2 in HD from Genda 2008 [molecules/m/s]
    '''
    return 4.48e19*T**0.75

def b_H_He(T):
    '''
    Input: T, temperature [K]
    Output: Binary diffusion coefficient for H in He [molecules/m/s]
    from Mason & Marrero 1970 (and Hu, Seager, Yung 2015)
    '''
    return 1.04e20*T**0.732

def V_reduction(Mp, Ms, a, Rp):
    '''
    Planetary gravitational reduction factor due to stellar tidal forces
    Erkaev et al 2007
    Inputs:
     - Mp: planet mass [kg]
     - Ms: stellar mass [kg]
     - a: orbital distance [m]
     - Rp: planet radius [m]
    Output: Grav potential reduction factor
    '''

    delta = Mp/Ms
    lam = a/Rp
    zeta = lam*(delta/3)**(1/3)
    K = 1 - 3/2/zeta + 1/2/zeta**3
    return K

def phi_RR(Rp, Mp, Teq, t, F0, t_sat):
    '''
    Radiation recombination-limited escape rate used in Lopez & Rice 2018 and others
    Prescription from Murray-Clay et al 2009, and used in Wordsworth et al 2018
    Inputs:
     - Rp: radius of the XUV photosphere [m]
     - Mp: planet mass [kg]
     - Teq: planetary equilibrium temperature [K]
     - t: system age (time) [s]
     - F0: initial planetary incident XUV flux [W/m2]
     - t_sat: XUV saturation time [yr]
    Output: mass flux [kg/m2/s]
    '''
    F = Fxuv(t, F0, t_sat)
    g = G*Mp/Rp**2 # grav field strength at base of flow [m/s2]
    T = 1e4 # temp is thermostatted at 1e4 K by radiation [K]
    nu_0 = 4.835e15 # EUV ionizing radiation frequency (~60 nm/ 20 eV) [Hz]
    alpha_rec = 2.7e-13*(Teq/1e4)**(-0.9)/1e6 # case B recombination coeff for H (Murray-Clay et al 2009 pg 4) [m3/atom/s]
    # H_base = kb*Teq/mu_solar/g # scale height at base of flow [m]
    # n_wind = np.sqrt(Fxuv/h/nu_0/H_base/alpha_rec) # number density at flow base [particles/m3]
    c_s = np.sqrt(2*kb*T/mu_H) # sounds speed at sonic point [m/s] Murray-Clay et al 2009
    R_s = G*Mp/(2*c_s**2) # sonic point [m] Murray-Clay et al 2009

    ### Lopez & Rice 2018 formulation
    # p1 = c_s*n_wind*mu_solar # divided both sides by 4 pi R_s^2 to change units to per area and removed negative sign
    # p2 = np.sqrt(Fxuv*G*Mp/(h*nu_0*alpha_rec*c_s**2*R_base**2))
    # p3 = np.exp((R_base/R_s - 1)*G*Mp/R_base/c_s**2)
    # return p1*p2*p3

    ### Murray-Clay et al 2009 formulation
    p1 = 4*np.pi*R_s**2*c_s
    p2 = np.sqrt(F*mu_H**3*g/(h*nu_0*alpha_rec*2*kb*Teq))
    p3 = np.exp((Rp/R_s - 1)*G*Mp/Rp/c_s**2)

    return p1*p2*p3/(4*np.pi*Rp**2)

def phi_RMC(t, F0, t_sat, Rp):
    F = Fxuv(t, F0, t_sat)
    phi = 4e9*np.sqrt(F/(5e5*cgs2si_flux))/(4*np.pi*Rp**2)
    return phi

def Rp_prim(Mp, f_atm, Fp, t0, T, mu, M_star, d):
    '''
    Calculates primordial planet radius
    Takes minimum of Lopez/Fortney 2014 calculation, Hill radius, Bondi radius
    Inputs:
     - Mp: planet mass [kg]
     - f_atm: atm mass fraction [ndim]
     - Fp: bolometric incident planetary flux [W/m2]
     - t0: simulation start time [s]
     - T: planet eq temp [K]
     - M_star: stellar mass [kg]
     - d: orbital distance [m]
    Output: planet radius [m]
    '''
    r_core = R_core(Mp)
    r_env = R_env(Mp, f_atm, Fp, t0)
    r_atm = R_atm(T, Mp, r_core, r_env, mu)
    R_LF = r_core + r_env + r_atm
    R_B = R_Bondi(Mp, mu, T)
    R_H = R_Hill(Mp, M_star, d)
    if type(R_LF) != 'int':
        Rp = np.zeros(len(R_LF))
        for i in range(len(R_LF)):
            Rp[i] = np.min([R_LF[i], R_B[i], R_H[i]])
    else:
        Rp = np.min([R_LF, R_B, R_H])
    return Rp

def epsilon(Mp, Rp):
    v_esc = np.sqrt(2*G*Mp/Rp)
    eps = 0.1*(v_esc/15e3)**(-2)
    return eps

def radius_valley(P, Rp, upper, lower):
    '''
    Checks if planet falls in the "fractionation valley," near the radius valley
    Inputs:
     - P: orbital period [days]
     - Rp: planetary radius [Earth radii]
     - upper: tuple or array with upper[0] = slopes and upper[1] = intercept for upper limit of valley
     - lower: tuple or array with lower[0] = slopes and lower[1] = intercept for lower limit of valley
    '''
    if Rp < 10**(upper[0]*np.log10(P) + upper[1]) and Rp > 10**(lower[0]*np.log10(P) + lower[1]):
        return True
    else:
        return False
    
def Johnson_reduction(eps, F_xuv, Rp, Vpot):
    '''
    Energy-limited escape rate reduction factor due to thermal and translational energy loss (Johnson et al 2013)
    Used in Hu et al 2015 and Malsky & Rogers 2020
    Inputs:
     - eps: heating efficiency factor [ndim]
     - F_xuv: Incident planetary XUV flux [W/m2]
     - Rp: planetary radius [m]
     - Vpot: planetary gravitational potential [J/kg]
    Output: escape flux reduction factor, f_r
    '''
    # Q_net = eps*L_xuv*Rp**2/4/a**2
    Q_net = eps*F_xuv*np.pi*Rp**2
    U = Vpot*mu_HHe # grav potential energy
    gamma = 5/3 # [ndim] adiabatic index; Malsky & Rogers 2020 uses 5/3, Gupta & Schlichting use 7/5 in CPML
    sigma = 5e-20 # [m2] collisional cross section from Malsky & Rogers 2020
    Kn = 1 # [ndim] Knudsen number from Malsky & Rogers 2020
    Q_c = (4*np.pi*Rp*gamma*U/sigma/Kn)*np.sqrt(2*U/mu_HHe)
    f_r = Q_c/Q_net
    if f_r < 1:
        # return f_r, Q_net, Q_c # use for johnson_reduction.py script
        return f_r
    elif f_r >=1:
        # return 1, Q_net, Q_c
        return 1
    
def f_atm_pred(Mc):
    '''
    Predicts f_atm from planet core mass based on
    models of gas accretion, boil off and disk dispersal.
    Used in Ginzburg et al 2016, Gupta+S 2019, Gupta et al 2022
    Input: planet core mass [kg]
    Output: atmopsheric mass fraction [ndim]
    '''
    return 0.05*np.sqrt(Mc/Me)

def f_atm_pred2(Mc, Teq):
    '''
    Predicts f_atm at time of disk dispersal from planet core mass and Teq based on
    models of gas accretion, boil off and disk dispersal from Ginzburg et al 2016 (eq 18)
    Input: planet core mass [kg], equilibrium temperature [K]
    Output: atmopsheric mass fraction [ndim]
    '''
    return 0.02*(Mc/Me)**0.8*(Teq/1e3)**(-0.25)

def f_atm_pred3(Mc, Teq):
    '''
    Predicts f_atm after disk dispersal\outer layer blow-off from planet core mass and Teq based on
    models of gas accretion, boil off and disk dispersal from Ginzburg et al 2016 (eq 24)
    Input: planet core mass [kg], equilibrium temperature [K]
    Output: atmopsheric mass fraction [ndim]
    '''
    return 0.01*(Mc/Me)**0.44*(Teq/1e3)**(0.25)

def NtoM(N1, N2, MM1, MM2):
    '''
    Converts molar concentration to mass concentration:
    N2/(N1 + N2) --> m2/(m1 + m2)
    Inputs:
        - N1: moles of species 1 [mol or particles]
        - N2: moles of species 2 [mol or particles]
        - MM1: molar mass of species 1 [kg/mol]
        - MM2: molar mass of species 2 [kg/mol]
    Output: Mass concentration
    '''

    return N2*MM2/(N1*MM1 + N2*MM2)

    


#####_____ANALYTIC SOLUTIONS_____#####

# for phi < phi_c

def x2_subcrit(x2_0, tau, t):
    return x2_0/(1 - t/tau)

# for phi > phi_c

def x2_supercrit(x2_0, Mp, Rp, T, phi_1, tau, t):
    # phi_1 must be in particles/m2/s
    m1 = M_H2/avogadro # kg/particle
    m2 = M_HD/avogadro # kg/particle
    b = 4.48e19*T**0.75 # [molecules/m/s] from Genda 2008 for H2 in HD
    g = G*Mp/Rp**2 # N/kg
    gamma = (m2 - m1)*b*g/(kb*T*phi_1) # ndim
    print(gamma)
    return x2_0/(1 - (t/tau))**gamma
