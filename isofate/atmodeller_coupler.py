'''
Collin Cherubim
June 30, 2025
Test script for atmodeller function for isocalc to call for isofate-atmodeller coupling
'''

import logging
import optimistix as optx
from atmodeller import (
    Planet,
    SolverParameters
)
from atmodeller.solubility import get_solubility_models
solubility_models = get_solubility_models()

#isofate imports
from IsoFATE.isofate..constants import *
from IsoFATE.isofate..orbit_params import *
from IsoFATE.isofate..isofunks import *

#other imports
import numpy as np

# logger = debug_logger()
# logger.setLevel(logging.INFO)

def AtmodellerCoupler(Teq, Mp, Rp, mu, melt_fraction, mantle_iron_dict,
                      N_H_atm, N_He_atm, N_O_atm, N_C_atm,
                      N_H_int, N_He_int, N_O_int, N_C_int, interior_atmosphere):

    results = {}
    gamma = 7/5
    T_surface, P_surface = make_atmosphere_descent(Teq, mu, Rp, Mp, gamma, 2)
    surface_temperature: float = np.min([6000, T_surface]) # K
    if melt_fraction != False:
        mantle_melt_fraction: float = melt_fraction
    elif melt_fraction == False:
        mantle_melt_fraction: float = MeltFraction(Mp, np.clip(T_surface, 10, 16000))
    planet_mass: float = Mp
    surface_radius: float = R_core(Mp)

    planet = Planet(surface_temperature=surface_temperature, mantle_melt_fraction=mantle_melt_fraction, planet_mass = planet_mass, 
                    surface_radius = surface_radius)

    # element masses
    mass_H: float = (N_H_atm + N_H_int)*mu_H
    mass_O: float = (N_O_atm + N_O_int)*mu_O
    mass_C: float = (N_C_atm + N_C_int)*mu_C
    mass_He: float = (N_He_atm + N_He_int)*mu_He

    mass_constraints = {
        "H": mass_H,
        "O": mass_O,
        "C": mass_C,
        "He": mass_He,
    }

    solver = optx.Newton
    solver_parameters = SolverParameters(solver=solver, throw=True)
    interior_atmosphere.solve(
    planet=planet,
    mass_constraints=mass_constraints,
    solver_parameters=solver_parameters
)

    output = interior_atmosphere.output
    sol = output.asdict()
    results['N_H_atm'] = sol['element_H']['atmosphere_number'][0]
    results['N_H_int'] = sol['element_H']['dissolved_number'][0]
    results['N_He_atm'] = sol['element_He']['atmosphere_number'][0]
    results['N_He_int'] = sol['element_He']['dissolved_number'][0]

    if mantle_iron_dict:
        if mantle_iron_dict['type'] == 'dynamic':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['element_O']['atmosphere_number'][0]
                results['N_O_int'] = sol['element_O']['dissolved_number'][0]
            else:
                mantle_iron_dict['mass_Fe'] = mantle_melt_fraction*Mp*mantle_iron_dict['Fe_mass_fraction']
                if mantle_iron_dict['mass_Fe'] > 0:
                    mantle_iron_dict['mass_Fe2'] = mantle_iron_dict['X_Fe2']*mantle_iron_dict['mass_Fe']
                    n_Fe2 = mantle_iron_dict['mass_Fe2']/M_Fe - 4*sol['O2_g']['atmosphere_moles'][0] # oxidize Fe2+ to Fe3+
                    n_O2_atm = sol['O2_g']['atmosphere_moles'][0] - 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2)
                    delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2)
                    sol['O2_g']['atmosphere_moles'][0] = np.max([n_O2_atm,0])
                    results['N_O_atm'] = sol['element_O']['atmosphere_number'] - 2*delta_n_O2_atm*avogadro
                    sol['element_O']['atmosphere_moles'][0] = results['N_O_atm']/avogadro
                    results['N_O_int'] = sol['element_O']['dissolved_number'][0] + 2*delta_n_O2_atm*avogadro
                    sol['element_O']['dissolved_number'][0] = results['N_O_int']/avogadro
                    mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*M_Fe, 0]) # update remaining Fe2 mass
                    mantle_iron_dict['X_Fe2'] = mantle_iron_dict['mass_Fe2']/mantle_iron_dict['mass_Fe']
                else:
                    results['N_O_atm'] = sol['element_O']['atmosphere_number'][0]
                    results['N_O_int'] = sol['element_O']['dissolved_number'][0]
        elif mantle_iron_dict['type'] == 'static':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['element_O']['atmosphere_number'][0]
                results['N_O_int'] = sol['element_O']['dissolved_number'][0]
            else:    
                n_Fe2 = mantle_iron_dict['mass_Fe2']/M_Fe - 4*sol['O2_g']['atmosphere_moles'][0] # oxidize Fe2+ to Fe3+; this goes neg. when n_O2 > n_Fe2/4
                n_O2_atm = sol['O2_g']['atmosphere_moles'][0] - 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2) # goes to zero when when n_O2 >= n_Fe2/4 (won't go neg.)
                delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2) # calculates exactly the n_O2 reacted away
                sol['O2_g']['atmosphere_moles'][0] = np.max([n_O2_atm,0]) # goes to zero even when n_Fe2 goes neg., which should put O2 back in atm. Confirmed: doesn't matter b/c of line results['N_O_atm'] = ...
                results['N_O_atm'] = sol['element_O']['atmosphere_number'] - 2*delta_n_O2_atm*avogadro # takes away only O2 reacted from total O_atm inventory
                sol['element_O']['atmosphere_moles'][0] = results['N_O_atm']/avogadro
                results['N_O_int'] = sol['element_O']['dissolved_number'][0] + 2*delta_n_O2_atm*avogadro
                sol['element_O']['dissolved_moles'][0] = results['N_O_int']/avogadro
                mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*M_Fe, 0]) # update remaining Fe2 mass
    else:
        results['N_O_atm'] = sol['element_O']['atmosphere_number'][0]
        results['N_O_int'] = sol['element_O']['dissolved_number'][0]
    results['N_C_atm'] = sol['element_C']['atmosphere_number'][0]
    results['N_C_int'] = sol['element_C']['dissolved_number'][0]
    results['M_atm'] = sol['atmosphere']['mass'][0]
    results['T_surface'] = T_surface
    results['T_surface_atmod'] = surface_temperature

    return results, sol, mantle_iron_dict
