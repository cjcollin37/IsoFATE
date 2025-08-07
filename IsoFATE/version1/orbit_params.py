'''
Collin Cherubim
October 3, 2022
Miscellaneous functions for planetary orbital parameters
'''

import numpy as np
from constants import *

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
