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
mu_H2 = M_H2/avogadro # molecular mass of H2 [kg/molecule]
mu_HD = M_HD/avogadro # molecular mass of HD [kg/molecule]
mu_H = M_H/avogadro # atomic mass of H [kg/atom]
mu_D = M_D/avogadro # atomic mass of D [kg/atom]
mu_He = M_He/avogadro # atomic mass of He [kg/atom]
vsmow = 1.5574e-4 # Vienna Standard Mean Ocean Water (D/H for Earth's oceans)
DtoH_solar = 0.0000194 # D/H solar ratio from Lodders 2003
HetoH_solar = 1/13.6 # He/H solar number ratio Lodders 2003
mu_HHe = 0.00122/avogadro # H/He with solar abundances
mu_H2He = 0.00227/avogadro # H2/He with solar abundances
mu_solar = 0.00235/avogadro # average particle mass for solar metallicity
Venus_TO = 1.32e-5 # number of terrestrial oceans in Venus' atmosphere (needs checking)