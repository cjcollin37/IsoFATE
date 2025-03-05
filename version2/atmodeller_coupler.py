'''
Collin Cherubim
July 23, 2024
Atmodeller function for to call for isofate-atmodeller coupling
'''

# atmodeller imports
from atmodeller import debug_logger
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet
from atmodeller.core import GasSpecies, Species
from atmodeller.constraints import FugacityConstraint, BufferedFugacityConstraint, SystemConstraints, MassConstraint, ElementMassConstraint
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.utilities import earth_oceans_to_kg
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import H2O_peridotite_sossi, H2O_basalt_dixon
from atmodeller.solubility.hydrogen_species import H2_basalt_hirschmann
from atmodeller.solubility.other_species import He_basalt
from atmodeller.solubility.carbon_species import CO_basalt_yoshioka
from atmodeller.solubility.carbon_species import CH4_basalt_ardia

from atmodeller import debug_logger
import logging
#isofate imports
from constants import *
from orbit_params import *
from isofunks_public import *
#other imports
import numpy as np

# logger = debug_logger()
# logger.setLevel(logging.INFO)

def AtmodellerCoupler(Teq, Mp, Rp, mu, melt_fraction, mantle_iron_dict,
                      N_H_atm, N_He_atm, N_O_atm, N_C_atm,
                      N_H_int, N_He_int, N_O_int, N_C_int, planet):

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
                    surface_radius = surface_radius, melt_composition='Basalt')

    # logger.info(planet)

    ### Note to user: species solubilities are automatically set when melt_composition is specified in the Planet object
    ### Remove melt_composition to manually set species compositions below if desired.
    H2O_g = GasSpecies("H2O", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) #, solubility=H2O_basalt_dixon())
    H2_g = GasSpecies("H2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) #, solubility = H2_basalt_hirschmann())
    O2_g = GasSpecies("O2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) # no solubility
    CO_g = GasSpecies("CO", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) #, solubility = CO_basalt_yoshioka())
    CO2_g = GasSpecies("CO2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) #, solubility=CO2_basalt_dixon())
    CH4_g = GasSpecies("CH4", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()) #, solubility = CH4_basalt_ardia())
    He_g = GasSpecies("He") #, solubility = He_basalt())

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, He_g])

    # atmospheric masses
    mass_H: float = (N_H_atm + N_H_int)*mu_H
    mass_O: float = (N_O_atm + N_O_int)*mu_O
    mass_C: float = (N_C_atm + N_C_int)*mu_C
    mass_He: float = (N_He_atm + N_He_int)*mu_He

    constraints: SystemConstraints = SystemConstraints([
        ElementMassConstraint('O', mass_O),
        ElementMassConstraint('C', mass_C), 
        ElementMassConstraint('H', mass_H),
        ElementMassConstraint('He', mass_He)
    ])

    # run atmodeller
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)
    interior_atmosphere.solve(constraints, errors="raise")
    interior_atmosphere.solution_dict()
    # interior_atmosphere.output(to_excel=True)
    # sys.exit(1)

    # save output
    # atm_solutions[str(time_slices[i]*s2yr)] = interior_atmosphere.solution_dict()
    sol = interior_atmosphere.output()
    results['N_H_atm'] = sol['H_total'][0]['atmosphere_moles']*avogadro
    results['N_H_int'] = sol['H_total'][0]['melt_moles']*avogadro
    results['N_He_atm'] = sol['He_total'][0]['atmosphere_moles']*avogadro
    results['N_He_int'] = sol['He_total'][0]['melt_moles']*avogadro

    if mantle_iron_dict:
        if mantle_iron_dict['type'] == 'dynamic':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro
                results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro
            else:
                mantle_iron_dict['mass_Fe'] = mantle_melt_fraction*Mp*mantle_iron_dict['Fe_mass_fraction']
                if mantle_iron_dict['mass_Fe'] > 0:
                    mantle_iron_dict['mass_Fe2'] = mantle_iron_dict['X_Fe2']*mantle_iron_dict['mass_Fe']
                    n_Fe2 = mantle_iron_dict['mass_Fe2']/M_Fe - 4*sol['O2_g'][0]['atmosphere_moles'] # oxidize Fe2+ to Fe3+
                    n_O2_atm = sol['O2_g'][0]['atmosphere_moles'] - 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2)
                    delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2)
                    sol['O2_g'][0]['atmosphere_moles'] = np.max([n_O2_atm,0])
                    results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro - 2*delta_n_O2_atm*avogadro
                    sol['O_total'][0]['atmosphere_moles'] = results['N_O_atm']/avogadro
                    results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro + 2*delta_n_O2_atm*avogadro
                    sol['O_total'][0]['melt_moles'] = results['N_O_int']/avogadro
                    mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*M_Fe, 0]) # update remaining Fe2 mass
                    mantle_iron_dict['X_Fe2'] = mantle_iron_dict['mass_Fe2']/mantle_iron_dict['mass_Fe']
                else:
                    results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro
                    results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro
        elif mantle_iron_dict['type'] == 'static':
            if mantle_iron_dict['mass_Fe2'] == 0:
                results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro
                results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro
            else:    
                n_Fe2 = mantle_iron_dict['mass_Fe2']/M_Fe - 4*sol['O2_g'][0]['atmosphere_moles'] # oxidize Fe2+ to Fe3+; this goes neg. when n_O2 > n_Fe2/4
                n_O2_atm = sol['O2_g'][0]['atmosphere_moles'] - 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2) # goes to zero when when n_O2 >= n_Fe2/4 (won't go neg.)
                delta_n_O2_atm = 0.25*(mantle_iron_dict['mass_Fe2']/M_Fe - n_Fe2) # calculates exactly the n_O2 reacted away
                sol['O2_g'][0]['atmosphere_moles'] = np.max([n_O2_atm,0]) # goes to zero even when n_Fe2 goes neg., which should put O2 back in atm. Confirmed: doesn't matter b/c of line results['N_O_atm'] = ...
                results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro - 2*delta_n_O2_atm*avogadro # takes away only O2 reacted from total O_atm inventory
                sol['O_total'][0]['atmosphere_moles'] = results['N_O_atm']/avogadro
                results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro + 2*delta_n_O2_atm*avogadro
                sol['O_total'][0]['melt_moles'] = results['N_O_int']/avogadro
                mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*M_Fe, 0]) # update remaining Fe2 mass
    else:
        results['N_O_atm'] = sol['O_total'][0]['atmosphere_moles']*avogadro
        results['N_O_int'] = sol['O_total'][0]['melt_moles']*avogadro
    results['N_C_atm'] = sol['C_total'][0]['atmosphere_moles']*avogadro
    results['N_C_int'] = sol['C_total'][0]['melt_moles']*avogadro
    results['M_atm'] = sol['atmosphere'][0]['mass']
    results['T_surface'] = T_surface
    results['T_surface_atmod'] = surface_temperature
    
    return results, sol, mantle_iron_dict
