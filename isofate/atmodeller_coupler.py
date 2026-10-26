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
from isofate.constants import *
from isofate.orbit_params import *
from isofate.isofunks import *

#other imports
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from atmodeller.constants import AVOGADRO
from atmodeller.engine import (
    get_atmosphere_log_molar_mass,
    get_atmosphere_log_volume,
    get_element_density,
    get_element_density_in_melt,
    get_log_activity,
    get_species_density_in_melt,
    get_total_pressure,
)
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import get_log_number_density_from_log_pressure, vmap_axes_spec

# logger = debug_logger()
# logger.setLevel(logging.INFO)

def AtmodellerCoupler(Teq, Mp, Rp, mu, melt_fraction, mantle_iron_dict,
                      N_H_atm, N_He_atm, N_O_atm, N_C_atm, N_N_atm, N_S_atm,
                      N_H_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, interior_atmosphere):

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
    mass_N: float = (N_N_atm + N_N_int)*mu_N
    mass_S: float = (N_S_atm + N_S_int)*mu_S
    mass_constraints = {
        "H": mass_H,
        "O": mass_O,
        "C": mass_C,
        "He": mass_He,
        "N": mass_N,
        "S": mass_S,
    }

    MASS_CUTOFF: float = 10**9
    mass_constraints = {
        "H": max(MASS_CUTOFF, mass_H),
        "O": max(MASS_CUTOFF, mass_O),
        "C": max(MASS_CUTOFF, mass_C),
        "He": max(MASS_CUTOFF, mass_He),
        "N": max(MASS_CUTOFF, mass_N),
        "S": max(MASS_CUTOFF, mass_S)
    }

    # solver = optx.Newton
    # solver_parameters = SolverParameters(solver=solver, throw=True)
    interior_atmosphere.solve(
    planet=planet,
    mass_constraints=mass_constraints,
    # solver_parameters=solver_parameters
)

    sol = output_asdict(interior_atmosphere.output)
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
                    results['N_O_atm'] = sol['element_O']['atmosphere_number'][0] - 2*delta_n_O2_atm*avogadro
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
                results['N_O_atm'] = sol['element_O']['atmosphere_number'][0] - 2*delta_n_O2_atm*avogadro # takes away only O2 reacted from total O_atm inventory
                sol['element_O']['atmosphere_moles'][0] = results['N_O_atm']/avogadro
                results['N_O_int'] = sol['element_O']['dissolved_number'][0] + 2*delta_n_O2_atm*avogadro
                sol['element_O']['dissolved_moles'][0] = results['N_O_int']/avogadro
                mantle_iron_dict['mass_Fe2'] = np.max([n_Fe2*M_Fe, 0]) # update remaining Fe2 mass
    else:
        results['N_O_atm'] = sol['element_O']['atmosphere_number'][0]
        results['N_O_int'] = sol['element_O']['dissolved_number'][0]
    results['N_C_atm'] = sol['element_C']['atmosphere_number'][0]
    results['N_C_int'] = sol['element_C']['dissolved_number'][0]
    results['N_N_atm'] = sol['element_N']['atmosphere_number'][0]
    results['N_N_int'] = sol['element_N']['dissolved_number'][0]
    results['N_S_atm'] = sol['element_S']['atmosphere_number'][0]
    results['N_S_int'] = sol['element_S']['dissolved_number'][0]
    results['M_atm'] = sol['atmosphere']['mass'][0]
    results['T_surface'] = T_surface
    results['T_surface_atmod'] = surface_temperature

    return results, sol, mantle_iron_dict


@eqx.filter_jit
def _output_arrays(traced_parameters, fixed_parameters, log_number_density, log_stability):
    traced_axes = vmap_axes_spec(traced_parameters)
    planet = traced_parameters.planet
    temperature_axes = vmap_axes_spec(planet.temperature)
    formula_matrix = jnp.asarray(fixed_parameters.formula_matrix)
    gas_species_mask = jnp.asarray(fixed_parameters.gas_species_mask)

    log_activity = eqx.filter_vmap(get_log_activity, in_axes=(traced_axes, None, 0))(
        traced_parameters, fixed_parameters, log_number_density)
    log_activity = jnp.where(fixed_parameters.active_stability(), log_activity - jnp.exp(log_stability), log_activity)
    log_volume = eqx.filter_vmap(get_atmosphere_log_volume, in_axes=(None, 0, vmap_axes_spec(planet)))(
        fixed_parameters, log_number_density, planet)
    volume = jnp.exp(log_volume)[:, None]

    element_gas = eqx.filter_vmap(get_element_density, in_axes=(None, 0))(
        formula_matrix, log_number_density*gas_species_mask)
    element_melt = eqx.filter_vmap(get_element_density_in_melt, in_axes=(traced_axes, None, None, 0, 0, 0))(
        traced_parameters, fixed_parameters, formula_matrix, log_number_density, log_activity, log_volume)
    species_melt = eqx.filter_vmap(get_species_density_in_melt, in_axes=(traced_axes, None, 0, 0, 0))(
        traced_parameters, fixed_parameters, log_number_density, log_activity, log_volume)

    total_pressure = eqx.filter_vmap(get_total_pressure, in_axes=(None, 0, temperature_axes))(
        fixed_parameters, log_number_density, planet.temperature)
    atmosphere_log_number_density = eqx.filter_vmap(
        get_log_number_density_from_log_pressure, in_axes=(0, temperature_axes))(
        jnp.log(total_pressure), planet.temperature)
    atmosphere_molar_mass = jnp.exp(eqx.filter_vmap(get_atmosphere_log_molar_mass, in_axes=(None, 0))(
        fixed_parameters, log_number_density))
    temperature = jnp.broadcast_to(planet.temperature, total_pressure.shape)

    return {
        'element_atmosphere_number': element_gas*volume,
        'element_dissolved_number': element_melt*volume,
        'species_atmosphere_moles': jnp.exp(log_number_density)*volume/AVOGADRO,
        'species_dissolved_moles': species_melt*volume/AVOGADRO,
        'fugacity': jnp.exp(log_activity),
        'atmosphere_mass': (jnp.exp(atmosphere_log_number_density)[:, None]*volume/AVOGADRO
                            *atmosphere_molar_mass[:, None]).ravel(),
        'log10_IW_1_bar': IronWustiteBuffer().log10_fugacity(temperature, 1.0),
    }


def output_asdict(output):
    '''
    The entries of atmodeller Output.asdict() that IsoFATE reads, computed in one compiled call.
    Relies on atmodeller 0.9.1 internals.
    '''
    arrays = {key: np.array(value) for key, value in _output_arrays(
        output._traced_parameters, output._fixed_parameters,
        jnp.asarray(output.log_number_density), jnp.asarray(output.log_stability)).items()}
    sol = {}
    gas_species_mask = output.gas_species_mask
    for ii, name in enumerate(output._species.get_species_names()):
        if gas_species_mask[ii]:
            sol[name] = {
                'atmosphere_moles': arrays['species_atmosphere_moles'][:, ii].copy(),
                'dissolved_moles': arrays['species_dissolved_moles'][:, ii].copy(),
                'fugacity': arrays['fugacity'][:, ii].copy(),
            }
    for ii, element in enumerate(output._species.get_unique_elements_in_species()):
        atmosphere_number = arrays['element_atmosphere_number'][:, ii].copy()
        dissolved_number = arrays['element_dissolved_number'][:, ii].copy()
        sol[f'element_{element}'] = {
            'atmosphere_number': atmosphere_number,
            'dissolved_number': dissolved_number,
            'atmosphere_moles': atmosphere_number/AVOGADRO,
            'dissolved_moles': dissolved_number/AVOGADRO,
        }
    sol['atmosphere'] = {'mass': arrays['atmosphere_mass']}
    if 'O2_g' in sol:
        sol['O2_g']['log10dIW_1_bar'] = np.log10(sol['O2_g']['fugacity']) - arrays['log10_IW_1_bar']
    return sol
