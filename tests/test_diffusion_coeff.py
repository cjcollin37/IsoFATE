'''
Tests of isofunks.get_binary_diffusion_coeff, the binary diffusion coefficients b [molecules/m/s] that set how
strongly the escaping flow drags each minor species along (the mass fractionation of Zahnle et al 1990).
Checks that b is symmetric in the two species, that the tabulated pairs of constants.py come back as they are,
and that the pairs without their own coefficient follow the Genda & Ikoma 2008 reduced-mass scaling.
'''

import numpy as np
import pytest

from isofate import isofunks as f
from isofate.constants import *

SPECIES = ['H', 'He', 'D', 'O', 'C', 'N', 'S']
MOLAR = {'H': M_H, 'D': M_D, 'He': M_He, 'O': M_O, 'C': M_C, 'N': M_N, 'S': M_S} # molar masses [kg/mol]
# the pairs with their own coefficient in constants.py
TABULATED = {('H', 'D'): b_H_D, ('H', 'He'): b_H_He, ('H', 'O'): b_H_O, ('H', 'C'): b_H_C, ('H', 'N'): b_H_N,
             ('H', 'S'): b_H_S, ('He', 'D'): b_He_D, ('He', 'O'): b_He_O, ('He', 'C'): b_He_C, ('He', 'N'): b_He_N,
             ('He', 'S'): b_He_S}


def reduced(s1, s2):
    '''
    Reduced molar mass of the pair s1, s2 [kg/mol]
    '''
    return MOLAR[s1]*MOLAR[s2]/(MOLAR[s1] + MOLAR[s2])


@pytest.mark.parametrize('T', [226.0, 1000.0])
def test_symmetric_and_tabulated(T):
    '''
    Checks that b(s1, s2) = b(s2, s1) for every pair of different species, and that each tabulated pair returns
    its own coefficient in either order. T [K]: 226 K is about the equilibrium temperature of LHS 1140 b.
    '''
    for s1 in SPECIES:
        for s2 in SPECIES:
            if s1 != s2:
                assert f.get_binary_diffusion_coeff(s1, s2, T) == f.get_binary_diffusion_coeff(s2, s1, T)
    for (s1, s2), func in TABULATED.items():
        assert f.get_binary_diffusion_coeff(s1, s2, T) == func(T)
        assert f.get_binary_diffusion_coeff(s2, s1, T) == func(T)


def test_He_D_is_not_H_He():
    '''
    Checks that the He-D pair gets b_He_D, which is b_H_He scaled by the reduced mass (0.775 b_H_He), not b_H_He
    itself: a look-up that sorts the pair to ('D', 'He') misses the table and silently returns b_H_He instead.
    '''
    b = f.get_binary_diffusion_coeff('He', 'D', 226.0)
    assert b == b_He_D(226.0) and b != b_H_He(226.0)
    np.testing.assert_allclose(b/b_H_He(226.0), np.sqrt(reduced('H', 'He')/reduced('He', 'D')), rtol = 1e-3)


def test_untabulated_pairs_follow_reduced_mass_scaling():
    '''
    Checks that the pairs without their own coefficient (D-O, D-C, D-N, D-S and the metal-metal pairs) get
    b_H_O*sqrt(mu_HO/mu_pair), the Genda & Ikoma 2008 prescription (Appendix C): b scales as 1/sqrt(reduced mass).
    The same scaling must also reproduce the approximated coefficients of constants.py.
    '''
    T = 226.0 # [K]
    for s1, s2 in [('D', 'O'), ('D', 'C'), ('D', 'N'), ('D', 'S'), ('C', 'O'), ('N', 'O'), ('C', 'N'),
                   ('O', 'S'), ('C', 'S'), ('N', 'S')]:
        expected = b_H_O(T)*np.sqrt(reduced('H', 'O')/reduced(s1, s2))
        np.testing.assert_allclose(f.get_binary_diffusion_coeff(s1, s2, T), expected, rtol = 1e-12)
    # the same scaling reproduces the approximated coefficients of constants.py to within 2.5%. Skipped: the
    # literature values (b_H_D from Genda & Ikoma 2008, b_H_He from Mason & Marrero 1970, b_H_O from Wordsworth
    # et al 2018) and b_He_D, which is scaled from b_H_He, not from b_H_O
    for (s1, s2), func in TABULATED.items():
        if func in (b_H_D, b_H_He, b_H_O, b_He_D):
            continue
        np.testing.assert_allclose(b_H_O(T)*np.sqrt(reduced('H', 'O')/reduced(s1, s2)), func(T), rtol = 0.025)
    # and b_He_D is b_H_He scaled from H to D by the reduced mass (the same He-H interaction potential), to 0.1%,
    # at every temperature (the same T^0.732)
    for T in (100.0, 226.0, 1000.0, 3000.0): # [K]
        np.testing.assert_allclose(b_H_He(T)*np.sqrt(reduced('H', 'He')/reduced('He', 'D')), b_He_D(T), rtol = 1e-3)


def test_identical_and_unknown_species():
    '''
    Checks that identical species give NaN (a species does not diffuse through itself), and that an unknown
    species raises a ValueError that names it.
    '''
    assert np.isnan(f.get_binary_diffusion_coeff('He', 'He', 226.0))
    with pytest.raises(ValueError, match = 'Fe'):
        f.get_binary_diffusion_coeff('H', 'Fe', 226.0)
