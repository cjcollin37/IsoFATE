'''
Collin Cherubim
October 2, 2022
Frequently used physical constants
'''

inv_cm2m = 100 # convert inverse cm to inverse m
avogadro = 6.022e23 # Avogadro's number [particles/mole]
s2day = 1/(3600*24) # convert s to day
s2yr = 1/(3600*24*365) # convert s to year
au2m = 1.496e11 # convert au to m
cgs2si_flux = 1/1000 # convert flux from erg/cm2/s to W/m2
erg2joule = 1e-7 # convert ergs to Joules
gcm2kgm = 1000 # convert g/cm3 to kg/m3 (SI)
J2E_mass = 1.898e27/5.972e24 # convert Jupiter mass to Earth mass [kg]
J2E_rad = 7.1492e7/6.3781e6 # convert Jupiter radius to Earth radius [m]
R_gas = 8.314462 # gas constant [J/mol/K]
kb = 1.38e-23 # Boltzmann constant [m2 kg/s2 K]
sbc = 5.67e-8 # Stefan Boltzmann constant W/m2/K4
g_e = 9.8 # grav field strength [m/s/s]
G = 6.6743e-11 # [m3/kg/s2]
h = 6.62607015e-34 # [J s]
c = 2.99792458e8 # [m/s]
Re = 6.378e6 # Earth radius [m]
Me = 5.9722e24 # Earth mass [kg]
Fe = 1366 # Earth bolometric flux [W/m2]
Ms = 1.98847e30 # Solar mass [kg]
Rs = 6.957e8 # Solar radius [m]
Ls = 3.828e26 # Solar luminosity [W]
Mjup = 1.898e27 # Jupiter mass [kg]
Rjup = 7.1492e7 # Jupiter radius [m]
M_N2 = 0.028014 # molar mass N2 [kg/mol]
M_H2 = 0.002016 # molar mass H2 [kg/mol]
M_HD = 0.003024 # molar mass HD [kg/mol]
M_D = 0.002014 # molar mass of D [kg/mol]
M_H  = 0.001008 # molar mass of H [kg/mol]
M_He = 0.0040026 # molar mass of He [kg/mol]
M_O = 0.015999 # molar mass of O [kg/mol]
M_C = 0.012011 # molar mass of C [kg/mol]
M_N = 0.014007 # molar mass of N [kg/mol]
M_S = 0.032065 # molar mass of S [kg/mol]
M_Fe = 0.055845 # molar mass of Fe [kg/mol]
M_H2O = 0.018015 # molar mass of H2O [kg/mol]
mu_H2 = M_H2/avogadro # molecular mass of H2 [kg/molecule]
mu_HD = M_HD/avogadro # molecular mass of HD [kg/molecule]
mu_H = M_H/avogadro # atomic mass of H [kg/atom]
mu_D = M_D/avogadro # atomic mass of D [kg/atom]
mu_He = M_He/avogadro # atomic mass of He [kg/atom]
mu_O = M_O/avogadro # atomic mass of O [kg/atom]
mu_C = M_C/avogadro # atomic mass of C [kg/atom]
mu_N = M_N/avogadro # atomic mass of N [kg/atom]
mu_S = M_S/avogadro # atomic mass of S [kg/atom]
mu_Fe = M_Fe/avogadro # atomic mass of Fe [kg/atom]
vsmow = 1.5574e-4 # Vienna Standard Mean Ocean Water (D/H for Earth's oceans)
DtoH_solar = 0.0000194 # D/H solar mole ratio from Lodders 2003
DtoH_solar_mass = DtoH_solar*(mu_D/mu_H) # D/H solar mass ratio from Lodders 2003
HetoH_solar = 0.07991 # He/H solar mole ratio from Lodders 2003 (0.2377/0.7491*mu_H/mu_He)
HetoH_protosolar = 0.09709 # He/H proto-solar mole ratio Lodders 2003 (0.2741/0.711*mu_H/mu_He)
HetoH_protosolar_mass = 0.38551 # He/H proto-solar mass ratio Lodders 2003 (0.2741/0.711)
OtoH_protosolar = 0.00058 # O/H proto-solar mole ratio from Lodders 2003 Table 2 (1.413e7/2.431e10)
OtoH_protosolar_mass = OtoH_protosolar*(mu_O/mu_H) # O/H proto-solar mass ratio from Lodders 2003
CtoH_protosolar = 0.00029 # C/H proto-solar mole ratio from Lodders 2003 Table 2
CtoH_protosolar_mass = CtoH_protosolar*(mu_C/mu_H) # O/H proto-solar mass ratio from Lodders 2003
NtoH_protosolar = 0.000080 # N/H proto-solar mole ratio from Lodders 2003 Table 2 (1.950e6/2.431e10)
NtoH_protosolar_mass = NtoH_protosolar*(mu_N/mu_H) # O/H proto-solar mass ratio
StoH_protosolar = 0.000018 # S/H proto-solar mole ratio from Lodders 2003 Table 2 (4.449e5/2.431e10)
StoH_protosolar_mass = StoH_protosolar*(mu_S/mu_H) # O/H proto-solar mass ratio 
mu_HHe = 0.00122/avogadro # H/He with solar abundances
mu_H2He = 0.00227/avogadro # H2/He with solar abundances
mu_solar = 0.00235/avogadro # average particle mass for solar metallicity
n_OperTO = 7.83e22 # mols O per TO
Venus_TO = 1.32e-5 # number of terrestrial oceans in Venus' atmosphere (needs checking)
def b_H_D(T):
    return 7.183e19*T**0.728 # [molecules/m/s] from Genda & Ikoma 2008 for D in H (not measured directly)
def b_H_He(T):
    return 1.04e20*T**0.732 # [molecules/m/s] from Mason & Marrero 1970 for H in He
def b_He_D(T):
    return 5.087e19*T**0.728 # [molecules/m/s] approximated from b_H_D using Genda/Ikoma 2008 prescription (Appendix C)
def b_H_O(T):
    return 4.8e19*T**0.75 # [molecules/m/s] from Wordsworth et al 2018
def b_He_O(T):
    return 2.61e19*T**0.75 # [molecules/m/s] approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_H_C(T):
    return 4.85e19*T**0.75 # [molecules/m/s] approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_He_C(T):
    return 2.64e19*T**0.75 # [molecules/m/s] approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_H_N(T):
    return 4.85e19*T**0.75 # [molecules/m/s] approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_He_N(T):  
    return 2.65e19*T**0.75 # [molecules/m/s] approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_H_S(T):
    return 4.73e19*T**0.75 # [molecules/m/s] approximated from b_H_O using Genda/Ikoma 2008 prescription (Appendix C)
def b_He_S(T):
    return 2.48e19*T**0.75 # [molecules/m/s] approximated from b_He_O using Genda/Ikoma 2008 prescription (Appendix C)