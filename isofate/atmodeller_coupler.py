'''
Collin Cherubim
June 30, 2025
Test script for atmodeller function for isocalc to call for isofate-atmodeller coupling
'''

import importlib.resources
from functools import cache, partial

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import optimistix as optx
from atmodeller import (
    INITIAL_LOG_NUMBER_DENSITY,
    INITIAL_LOG_STABILITY,
    TAU,
    InteriorAtmosphere,
    Planet,
    SolverParameters,
    Species,
    SpeciesCollection,
)
from atmodeller.constants import AVOGADRO
from atmodeller.containers import FugacityConstraints, MassConstraints, TracedParameters
from atmodeller.engine import (
    get_atmosphere_log_molar_mass,
    get_atmosphere_log_volume,
    get_element_density,
    get_element_density_in_melt,
    get_log_activity,
    get_species_density_in_melt,
    get_total_pressure,
    objective_function,
)
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import get_log_number_density_from_log_pressure, unit_conversion
from molmass import Formula

#isofate imports
from isofate.constants import *
from isofate.orbit_params import *
from isofate.isofunks import *

solubility_models = get_solubility_models()

njit = partial(nb.njit, error_model='numpy')
_atmosphere_descent = njit(atmosphere_descent)

with importlib.resources.as_file(
        importlib.resources.files('isofate.data').joinpath('melt_fraction_grid.npz')) as _path, np.load(_path) as _data:
    MELT_MASS_GRID = _data['mass_grid'].astype(float) # [Earth masses]
    MELT_TEMP_GRID = _data['temp_grid'].astype(float) # [K]
    MELT_FRACTION_GRID = np.ascontiguousarray(_data['psi_grid'].T) # [mass, temperature]

# atomic masses Atmodeller converts mass constraints to atom numbers with [kg/atom]. IsoFATE's mu_X differ by up to
# 1e-4, so using them here would add that fraction of each element's inventory on every Atmodeller call.
ATOM_MASS = {element: Formula(element).mass*unit_conversion.g_to_kg/AVOGADRO
             for element in ('H', 'He', 'O', 'C', 'N', 'S')}


def isofate_species():
    '''
    The Atmodeller species isocalc couples to.
    '''
    H2O_g = Species.create_gas("H2O", solubility=solubility_models["H2O_basalt_dixon95"])
    H2_g = Species.create_gas("H2", solubility=solubility_models["H2_basalt_hirschmann12"])
    O2_g = Species.create_gas("O2")
    CO_g = Species.create_gas("CO", solubility=solubility_models["CO_basalt_yoshioka19"])
    CO2_g = Species.create_gas("CO2", solubility=solubility_models["CO2_basalt_dixon95"])
    CH4_g = Species.create_gas("CH4", solubility=solubility_models["CH4_basalt_ardia13"])
    He_g = Species.create_gas("He", solubility=solubility_models["He_basalt_jambon86"])
    N2_g = Species.create_gas("N2", solubility=solubility_models["N2_basalt_libourel03"])
    S2_g = Species.create_gas("S2", solubility=solubility_models["S2_sulfide_basalt_boulliung23"])
    H2O4S_g = Species.create_gas("H2O4S")
    SO2_g = Species.create_gas("SO2")
    return SpeciesCollection((H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g, He_g, N2_g, S2_g, H2O4S_g, SO2_g))


@njit
def _interpolate_melt_fraction(m, T, mass_grid, temp_grid, psi):
    '''
    scipy RegularGridInterpolator(method='linear') of psi at (m, T); NaN where it would raise.
    '''
    if not (mass_grid[0] <= m <= mass_grid[-1] and temp_grid[0] <= T <= temp_grid[-1]):
        return np.nan
    i0 = min(np.searchsorted(mass_grid, m, side='right') - 1, mass_grid.size - 2)
    i1 = min(np.searchsorted(temp_grid, T, side='right') - 1, temp_grid.size - 2)
    y0 = (m - mass_grid[i0])/(mass_grid[i0 + 1] - mass_grid[i0])
    y1 = (T - temp_grid[i1])/(temp_grid[i1 + 1] - temp_grid[i1])
    result = 0.0
    result = result + psi[i0, i1]*(1 - y0)*(1 - y1)
    result = result + psi[i0, i1 + 1]*(1 - y0)*y1
    result = result + psi[i0 + 1, i1]*y0*(1 - y1)
    result = result + psi[i0 + 1, i1 + 1]*y0*y1
    return result


def melt_fraction_from_grid(Mp, T):
    '''
    isofunks.MeltFraction at a single point.
    '''
    psi = _interpolate_melt_fraction(Mp/Me, T, MELT_MASS_GRID, MELT_TEMP_GRID, MELT_FRACTION_GRID)
    if psi != psi:
        raise ValueError(f'melt fraction grid does not cover Mp = {Mp/Me} Me, T = {T} K')
    return psi


def AtmodellerCoupler(Teq, Mp, Rp, mu, melt_fraction, mantle_iron_dict,
                      N_H_atm, N_He_atm, N_O_atm, N_C_atm, N_N_atm, N_S_atm,
                      N_H_int, N_He_int, N_O_int, N_C_int, N_N_int, N_S_int, atmodeller_solver):

    results = {}
    gamma = 7/5
    planet_mass: float = Mp
    surface_radius: float = R_core(Mp)
    T_surface = _atmosphere_descent(Teq, mu, Rp, Mp, surface_radius, gamma)[1][0]
    surface_temperature: float = min(T_surface, 6000.0) # K
    if melt_fraction != False:
        mantle_melt_fraction: float = melt_fraction
    elif melt_fraction == False:
        mantle_melt_fraction: float = melt_fraction_from_grid(Mp, min(max(T_surface, 10.0), 16000.0))

    # element masses, in Atmodeller's atomic masses so the atom numbers it returns add up to the ones passed in
    mass_H: float = (N_H_atm + N_H_int)*ATOM_MASS['H']
    mass_O: float = (N_O_atm + N_O_int)*ATOM_MASS['O']
    mass_C: float = (N_C_atm + N_C_int)*ATOM_MASS['C']
    mass_He: float = (N_He_atm + N_He_int)*ATOM_MASS['He']
    mass_N: float = (N_N_atm + N_N_int)*ATOM_MASS['N']
    mass_S: float = (N_S_atm + N_S_int)*ATOM_MASS['S']
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

    sol = atmodeller_solver.solve(surface_temperature, mantle_melt_fraction, planet_mass, surface_radius,
                                  mass_constraints)
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


class _CompiledSystem:
    '''
    InteriorAtmosphere.solve for isofate_species and the outputs IsoFATE reads, as one compiled call.
    Newton runs on the log number densities alone, so the species must be gases without stability unknowns.
    Relies on atmodeller 0.9.1 internals.
    '''

    def __init__(self):
        self.interior_atmosphere = InteriorAtmosphere(isofate_species())
        species = self.interior_atmosphere.species
        fixed = self.interior_atmosphere.get_fixed_parameters()
        if np.any(fixed.active_stability()) or not np.all(fixed.gas_species_mask):
            raise ValueError('the compiled Atmodeller solve needs gas species without stability unknowns')
        self.species_names = species.get_species_names()
        self.elements = species.get_unique_elements_in_species()
        self.n_species = species.number
        self.log_molar_masses = np.log([Formula(element).mass*unit_conversion.g_to_kg for element in self.elements])
        self.cold_start = np.full(self.n_species, INITIAL_LOG_NUMBER_DENSITY)

        n, ne = self.n_species, len(self.elements)
        fugacity_constraints = FugacityConstraints.create(species, None)
        mass_constraints = MassConstraints.create(species, {element: 1.0 for element in self.elements}, 1)
        active_indices = jnp.where(jnp.concatenate((fugacity_constraints.active(), fixed.active_reactions(),
                                                    mass_constraints.active(), fixed.active_stability())))[0]
        log_stability = jnp.full(n, INITIAL_LOG_STABILITY)
        solver_parameters = SolverParameters()
        options = {'lower': species.get_lower_bound()[:n], 'upper': species.get_upper_bound()[:n],
                   'jac': solver_parameters.jac}

        def residual(log_number_density, args):
            return objective_function(jnp.concatenate((log_number_density, log_stability)), args)

        def traced_parameters(x):
            planet = Planet(surface_temperature=x[0], mantle_melt_fraction=x[1], planet_mass=x[2],
                            surface_radius=x[3])
            return TracedParameters(planet, fugacity_constraints, MassConstraints(x[4:4 + ne], self.elements))

        def outputs(traced, log_number_density):
            planet = traced.planet
            log_activity = get_log_activity(traced, fixed, log_number_density)
            log_volume = get_atmosphere_log_volume(fixed, log_number_density, planet)
            volume = jnp.exp(log_volume)
            element_gas = get_element_density(fixed.formula_matrix, log_number_density*fixed.gas_species_mask)
            element_melt = get_element_density_in_melt(traced, fixed, fixed.formula_matrix, log_number_density,
                                                       log_activity, log_volume)
            species_melt = get_species_density_in_melt(traced, fixed, log_number_density, log_activity, log_volume)
            total_pressure = get_total_pressure(fixed, log_number_density, planet.temperature)
            atmosphere_log_number_density = get_log_number_density_from_log_pressure(jnp.log(total_pressure),
                                                                                     planet.temperature)
            atmosphere_molar_mass = jnp.exp(get_atmosphere_log_molar_mass(fixed, log_number_density))
            return jnp.concatenate((
                log_number_density, element_gas*volume, element_melt*volume,
                jnp.exp(log_number_density)*volume/AVOGADRO, species_melt*volume/AVOGADRO, jnp.exp(log_activity),
                jnp.atleast_1d(jnp.exp(atmosphere_log_number_density)*volume/AVOGADRO*atmosphere_molar_mass),
                jnp.atleast_1d(IronWustiteBuffer().log10_fugacity(planet.temperature, 1.0))))

        def solve(x):
            traced = traced_parameters(x)
            args = {'traced_parameters': traced, 'active_indices': active_indices, 'tau': jnp.asarray(TAU),
                    'fixed_parameters': fixed, 'solver_parameters': solver_parameters}
            solution = optx.root_find(residual, solver_parameters.get_solver_instance(), x[4 + ne:], args=args,
                                      throw=False, max_steps=solver_parameters.max_steps, options=options)
            values = outputs(traced, solution.value)
            converged = (solution.result == optx.RESULTS.successful) & jnp.all(jnp.isfinite(values))
            return jnp.concatenate((jnp.atleast_1d(converged).astype(values.dtype), values))

        self.solve = jax.jit(solve)
        self.outputs = jax.jit(lambda x: outputs(traced_parameters(x), x[4 + ne:]))

    def pack(self, surface_temperature, mantle_melt_fraction, planet_mass, surface_radius, mass_constraints,
             log_number_density):
        x = np.empty(4 + len(self.elements) + self.n_species)
        x[:4] = surface_temperature, mantle_melt_fraction, planet_mass, surface_radius
        masses = np.array([mass_constraints[element] for element in self.elements], dtype=float)
        x[4:4 + len(self.elements)] = np.log(masses) + np.log(AVOGADRO) - self.log_molar_masses
        x[4 + len(self.elements):] = log_number_density
        return x

    def asdict(self, values):
        '''
        The entries of atmodeller Output.asdict() that IsoFATE reads, from the outputs of one solve.
        '''
        n, ne = self.n_species, len(self.elements)
        element_numbers = values[n:n + 2*ne].reshape(2, ne) # atmosphere, dissolved
        element_moles = element_numbers/AVOGADRO
        species = values[n + 2*ne:4*n + 2*ne].reshape(3, n) # atmosphere moles, dissolved moles, fugacity
        sol = {name: {'atmosphere_moles': species[0, ii:ii + 1], 'dissolved_moles': species[1, ii:ii + 1],
                      'fugacity': species[2, ii:ii + 1]} for ii, name in enumerate(self.species_names)}
        for ii, element in enumerate(self.elements):
            sol[f'element_{element}'] = {
                'atmosphere_number': element_numbers[0, ii:ii + 1], 'dissolved_number': element_numbers[1, ii:ii + 1],
                'atmosphere_moles': element_moles[0, ii:ii + 1], 'dissolved_moles': element_moles[1, ii:ii + 1]}
        sol['atmosphere'] = {'mass': values[-2:-1]}
        sol['O2_g']['log10dIW_1_bar'] = np.log10(sol['O2_g']['fugacity']) - values[-1:]
        return sol


@cache
def _compiled_system():
    return _CompiledSystem()


class AtmodellerSolver:
    '''
    Atmodeller solves for isofate_species, each starting from the previous solution.
    '''

    def __init__(self):
        self._system = _compiled_system()
        self._initial = self._system.cold_start

    def solve(self, surface_temperature, mantle_melt_fraction, planet_mass, surface_radius, mass_constraints):
        '''
        The entries of atmodeller Output.asdict() that IsoFATE reads, for mass constraints in kg by element.
        Falls back to InteriorAtmosphere.solve, which starts cold and retries with multistart, when the
        compiled solve does not converge.
        '''
        system = self._system
        planet = (surface_temperature, mantle_melt_fraction, planet_mass, surface_radius)
        result = np.array(system.solve(system.pack(*planet, mass_constraints, self._initial)))
        if result[0]:
            values = result[1:]
            self._initial = values[:system.n_species]
            return system.asdict(values)

        interior_atmosphere = system.interior_atmosphere
        interior_atmosphere.solve(planet=Planet(surface_temperature=surface_temperature,
                                                mantle_melt_fraction=mantle_melt_fraction, planet_mass=planet_mass,
                                                surface_radius=surface_radius),
                                  mass_constraints=mass_constraints)
        log_number_density = interior_atmosphere.output.log_number_density[0]
        if interior_atmosphere.output._solver_status[0]:
            self._initial = log_number_density
        return system.asdict(np.array(system.outputs(system.pack(*planet, mass_constraints, log_number_density))))
