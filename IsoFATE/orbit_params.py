'''
Collin Cherubim
October 3, 2022
Miscellaneous functions for planetary orbital parameters
'''

import numpy as np
from isofate.constants import *

def Luminosity(R, T):
    '''
    Input: R, stellar radius [m]; T, stellar temp [K]
    Output: luminosity [W]
    '''
    L = 4*np.pi*R**2*sbc*T**4
    return L

def SemiMajor(M, P):
    '''
    Input: M, stellar mass [kg]; P, orbital period [s]
    Output: semi-major axis [m]
    '''
    a = ((G*M/4/np.pi**2)*P**2)**(1/3)
    return a

def Period(a, M):
    '''
    Input: M, stellar mass [kg]; a, semi-major axis [m]
    Output: orbital period [s]
    '''
    P = np.sqrt(a**3*4*np.pi**2/G/M)
    return P

def Insolation(L, a):
    '''
    Input: L, stellar luminosity [W]; a, semi-major axis [m]
    Output: incident stellar flux [W/m2]
    '''
    I = L/(4*np.pi*a**2)
    return I

def EqTemp(I, A = 0):
    '''
    Input: planetary incident bolometric flux [W/m2]
    Output: eq temp [K]
    '''
    T = (I*(1 - A)/4/sbc)**(1/4)
    return T

def HabZone(L, Teff, HZ = 'runaway greenhouse'):
    '''
    Calculates HZ distance based on Kopparapu et al 2013
    Inputs:
        - L: stellar luminosity [W]
        - Teff: stellar effective temperature [K]
        - HZ: desired habitable zone
            - 'recent venus'
            - 'runaway greenhouse' (default)
            - 'moist greenhouse'
            - 'max greenhouse'
            - 'early mars'
    Output: orbital distance or corresponding HZ limit [m]
    '''
    if HZ == 'recent venus':
        Seff_solar = 1.7763
        a = 1.4335e-4
        b = 3.3954e-9
        c = -7.6364e-12
        d = -1.1950e-15
    elif HZ == 'runaway greenhouse':
        Seff_solar = 1.0385
        a = 1.2456e-4
        b = 1.4612e-8
        c = -7.6345e-12
        d = -1.7511e-15
    elif HZ == 'moist greenhouse':
        Seff_solar = 1.0146
        a = 8.1884e-5 
        b = 1.9394e-9
        c = -4.3618e-12 
        d = -6.8260e-16
    elif HZ == 'max greenhouse':
        Seff_solar = 0.3507
        a = 5.9578e-5 
        b = 1.6707e-9
        c = -3.0058e-12 
        d = -5.1925e-16
    elif HZ == 'early mars':
        Seff_solar = 0.3207
        a = 5.4471e-5 
        b = 1.5275e-9
        c = -2.1709e-12 
        d = -3.8282e-16

    T_star = Teff - 5780 #[K]
    Seff = Seff_solar + a*T_star + b*T_star**2 + c*T_star**3 + d*T_star**4 # HZ stellar flux
    d = (L/Ls/Seff)**0.5*au2m # orbtial distance of corresponding HZ
    return d
