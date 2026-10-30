'''
TOML configuration of an MCMC run: the defaults, the parameter and observable registries, reading a config and
its --set overrides, validation of every section (solve mode included), the resolved config.json, the output
directories and the README tables.

load_config(path, overrides) returns the normalized cfg dict, a plain, picklable dict (the spawned workers get a
copy). Internally everything is SI (star mass [kg], distances [m], period [s]), except the parameters and
observables, which keep the units of the PARAMS and OBSERVABLES registries below (e.g. Mp [M_earth], age [yr],
Mdot [kg/s]).

A relative output_root is relative to the config file's directory. Without output_root, results go to
./mcmc_results in the current working directory.
'''

import copy
import difflib
import importlib.resources
import json
import math
import os
import re
import tomllib

import numpy as np

from isofate.constants import Ls, Ms, Rs, au2m, s2day
from isofate.orbit_params import Insolation, Luminosity, SemiMajor

inf = math.inf
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs') # example configs and the template
TEMPLATE = os.path.join(CONFIG_DIR, 'template.toml') # every option at its default (python -m isofate.mcmc template)
MAX_N_STEPS = int(1e6) # larger needs model.allow_large_n_steps = true (worker memory grows with n_steps)
MAX_AGE = 1.4e10 # [yr] upper end of the age, about the age of the Universe


class ConfigError(ValueError):
    '''
    A problem with the configuration, with a message saying how to fix it
    The command line prints it as "error: ..." without a traceback and exits with code 2.
    '''


###_____Registries_____###

def _param(default, unit, label, limits, doc, optional = False):
    '''
    Returns one entry of the PARAMS registry

    Inputs:
        - default: value of a fixed parameter without a value in the config [unit]; None: required, or unset if optional
        - unit: unit in the TOML, in theta and in the outputs; '' for dimensionless [str]
        - label: matplotlib axis label [str]
        - limits: hard validity range (lo, hi, lo_open, hi_open) [tuple]; None: set by hard_limits
        - doc: meaning, the last column of the README params table [str]
        - optional: True when a default of None means unset (off), not required [Bool]

    Output: {default, unit, label, limits, doc, optional} [dict]
    '''
    return {'default': default, 'unit': unit, 'label': label, 'limits': limits, 'doc': doc, 'optional': optional}


# PARAMETER REGISTRY - every model input a config can fix or vary. The free ones, in this order, make up theta.
# limits = hard validity (lo, hi, lo_open, hi_open), True for an open (excluded) end; None: see hard_limits, since
# the valid Mp and age depend on the model settings. The doc strings make up the README params table.
PARAMS = {
    'log_f_atm': _param(-2.0, 'dex', r'$\log_{10} f_{\rm atm}$', (-inf, 0.0, True, True),
                        'initial atmospheric mass fraction, f_atm = 10**x'),
    'Mp': _param(None, 'M_earth', r'$M_p$', None, 'planet mass (required)'),
    'age': _param(5e9, 'yr', 'age', None, 'planet age = end of the run'),
    'log_metallicity': _param(0.0, 'dex', r'$\log_{10} Z$', (-inf, inf, True, True),
                              'scales O, C, N, S relative to H (ratios composition only)'),
    'log_OtoH_enhancement': _param(0.0, 'dex', r'$\log_{10}$ O/H enh.', (-inf, inf, True, True), 'extra O/H'),
    'log_CtoH_enhancement': _param(0.0, 'dex', r'$\log_{10}$ C/H enh.', (-inf, inf, True, True),
                                   'extra C/H (ratios composition only)'),
    'eps': _param(0.15, '', r'$\epsilon$', (0.0, 1.0, True, False), 'heating efficiency'),
    'log_F0_ratio': _param(-3.0, 'dex', r'$\log_{10} F_0/F_p$', (-inf, inf, True, True),
                           'initial XUV flux F0 = Fp*10**x'),
    'F_final': _param(0.0, 'W/m2', r'$F_{\rm final}$', (0.0, inf, False, True),
                      'XUV flux after t_sat when step_fn = true'),
    't_sat': _param(5e8, 'yr', r'$t_{\rm sat}$', (0.0, inf, True, True), 'XUV saturation time'),
    't_pms': _param(0.0, 'yr', r'$t_{\rm pms}$', (0.0, inf, False, True),
                    'end of the pre-main-sequence XUV boost (0: no boost)'),
    'pms_factor': _param(0.0, '', 'PMS factor', (0.0, inf, False, True),
                         'XUV boost at t0, falling to 1 at t_pms (0: none; must be > 0 when t_pms > t0)'),
    'beta': _param(-1.23, '', r'$\beta$', (-inf, inf, True, True), 'XUV power-law index (step_fn = false)'),
    'albedo': _param(0.0, '', r'$A_B$', (0.0, 1.0, False, True), 'Bond albedo, T = EqTemp(Fp, A)'),
    'Rp_fixed': _param(None, 'R_earth', r'$R_{p,\rm fixed}$', (0.0, inf, True, True),
                       'constant radius (isocalc Rp_override); needs rad_evol = false, which is then set. Unset: off',
                       optional = True),
}


def _obs(unit, label, floor = 0.0, ceiling = None, needs_atmodeller = False, input = False, doc = ''):
    '''
    Returns one entry of the OBSERVABLES registry

    Inputs:
        - unit: unit of value in the TOML and in the blobs; '' for dimensionless [str]
        - label: matplotlib axis label [str]
        - floor: in log10 space the model is clipped to [floor, ceiling] before the log, so a zero escape rate or
          ratio becomes log10(floor) instead of -inf; 0: no clipping (log10(0) = -inf) [unit]
        - ceiling: None means 1/floor, which caps the infinite ratios x/0 of dimensionless observables (worked out
          in _observables from the floor in use); the escape rates [kg/s] pass inf, no ceiling, since their
          values lie far above 1/floor [unit]
        - needs_atmodeller: True when only an Atmodeller solve gives the value [Bool]
        - input: True for an isocalc input (Mp, age, Teq): compared before the model runs, it acts as a prior [Bool]
        - doc: definition, a column of the README observables table [str]

    Output: {unit, label, floor, ceiling, needs_atmodeller, input, doc} [dict]
    '''
    return {'unit': unit, 'label': label, 'floor': floor, 'ceiling': ceiling, 'needs_atmodeller': needs_atmodeller,
            'input': input, 'doc': doc}


# Gases with a surface VMR observable: the 10 Atmodeller gases plus He
VMR_MOLECULES = ('H2', 'He', 'H2O', 'CO', 'CO2', 'CH4', 'N2', 'O2', 'S2', 'SO2', 'H2SO4')

# OBSERVABLE REGISTRY - what the model can be compared with. Units are those of value in the TOML and of the blobs;
# every value is taken at the last step of the run (age + one step). This order is the order of the blob fields
# (model.BLOB_FIELDS) and of the README observables table, which the doc strings make up.
OBSERVABLES = {
    'Mp': _obs('M_earth', r'$M_p$', input = True, doc = 'planet mass (input: acts as a prior on Mp)'),
    'age': _obs('yr', 'age', input = True, doc = 'age (input: acts as a prior on age)'),
    'Teq': _obs('K', r'$T_{\rm eq}$', input = True,
                doc = 'equilibrium temperature EqTemp(Fp, albedo), full redistribution (input)'),
    'Rp': _obs('R_earth', r'$R_p$', doc = 'final radius Rp[-1] (Lopez & Fortney; the core radius when stripped)'),
    'fatm': _obs('', r'$f_{\rm atm}$', 1e-12, doc = 'final atmospheric mass fraction fatm[-1] (0 when stripped)'),
    'Mdot': _obs('kg/s', r'$\dot M$', 1e-3, inf, doc = 'total escape rate, sum over X of Phi_X mu_X 4 pi Rp^2'),
    'Mdot_H': _obs('kg/s', r'$\dot M_{\rm H}$', 1e-3, inf, doc = 'H escape rate Phi_H mu_H 4 pi Rp^2'),
    'Mdot_He': _obs('kg/s', r'$\dot M_{\rm He}$', 1e-3, inf,
                    doc = 'He escape rate Phi_He mu_He 4 pi Rp^2 (0 when the flux is subcritical)'),
    'Mdot_HHe': _obs('kg/s', r'$\dot M_{\rm H+He}$', 1e-3, inf,
                     doc = 'H+He escape rate (Phi_H mu_H + Phi_He mu_He) 4 pi Rp^2, the mass-loss rate of p-winds'),
    'outflow_H_fraction': _obs('', r'H/(H+He)$_{\rm out}$', 1e-15,
                               doc = 'H number fraction of the H+He outflow, Phi_H/(Phi_H + Phi_He) (p-winds '
                                     'h_fraction); 1 when He does not escape, NaN without an H/He wind'),
    'outflow_He_fraction': _obs('', r'He/(H+He)$_{\rm out}$', 1e-15,
                                doc = 'He number fraction of the outflow, Phi_He/(Phi_H + Phi_He); NaN without an '
                                      'H/He wind'),
    'escape_He_H': _obs('', r'$\Phi_{\rm He}/\Phi_{\rm H}$', 1e-12,
                        doc = 'legacy: Phi_He/Phi_H (0/0: NaN); prefer outflow_He_fraction'),
    'escape_H_He': _obs('', r'$\Phi_{\rm H}/\Phi_{\rm He}$', 1e-12,
                        doc = "legacy: Phi_H/Phi_He, the old script's flux_ratio (inf when He does not escape); "
                              'prefer outflow_H_fraction'),
    'H_He': _obs('', 'H/He', 1e-12, doc = '(N_H + N_D)/N_He in the atmosphere [atoms]'),
    'C_O': _obs('', 'C/O', 1e-12,
                doc = 'atmospheric C/O: Atmodeller gas totals of the final call, else kernel N_C/N_O'),
    'D_H': _obs('', 'D/H', 1e-12, doc = 'atmospheric N_D/N_H from the escape kernel (Atmodeller lumps D into H)'),
    'mmw': _obs('g/mol', r'$\mu$', needs_atmodeller = True,
                doc = 'mean molecular weight of the 10 Atmodeller gases plus He'),
}
# One VMR per gas; He is not an Atmodeller gas, so its amount comes from conservation of the He budget
for _mol in VMR_MOLECULES:
    OBSERVABLES[f'VMR_{_mol}'] = _obs('', f'VMR {_mol}', 1e-15, needs_atmodeller = True,
                                      doc = f'volume mixing ratio of {_mol} at the surface (Atmodeller)'
                                      + (', He from conservation' if _mol == 'He' else ''))
OBSERVABLES['T_surf'] = _obs('K', r'$T_{\rm surf}$', needs_atmodeller = True,
                             doc = 'surface temperature at the base of the adiabat (not directly observable)')


###_____Allowed keys, defaults and choices_____###

# Allowed keys of the top level, of [star], [planet], [params.NAME] and [observables.NAME]
TOP_KEYS = ('name', 'description', 'output_root', 'star', 'planet', 'model', 'params', 'observables', 'solve',
            'sampler', 'parallel', 'resolution')
STAR_KEYS = ('mass', 'radius', 'teff', 'luminosity') # [M_sun], [R_sun], [K], [L_sun]
PLANET_KEYS = ('period', 'semi_major_axis', 'Fp') # [days], [AU], [W/m2]
PARAM_KEYS = ('vary', 'value', 'prior', 'bounds', 'mu', 'sigma', 'init_scale')
OBS_KEYS = ('use', 'value', 'sigma', 'space', 'kind', 'floor', 'sigma_model')

# DEFAULTS of the other sections; configs/template.toml explains every key. _section coerces a value to the type of
# its default (bool, int, float, list of whole numbers) and leaves string defaults such as nwalkers = 'auto' to the
# section's own checks; [model] is checked key by key in _model
MODEL_DEFAULTS = {
    'n_steps': 100000, 'n_atmodeller': 100, 'mechanism': 'XUV', 'flux_model': 'power law', 'stellar_type': 'M1',
    'activity': 'medium', 'step_fn': False, 'RR': True, 'rad_evol': True, 'thermal': True, 'dynamic_phi': True,
    't0': 1e6, 'rho_rcb': 1.0, 'melt_fraction_override': False, 'mantle_iron': False, 'composition': 'ratios',
    'backend': 'isocalc', 'allow_large_n_steps': False}
SAMPLER_DEFAULTS = {
    'nwalkers': 'auto', 'max_steps': 5000, 'checkpoint_every': 50, 'print_every': 10, 'moves': 'de',
    'seed': 12345, 'init': 'prescan', 'prescan_factor': 8, 'bisect_param': 'log_f_atm', 'bisect_observable': 'Rp',
    'bisect_draws': 'auto', 'bisect_iters': 22, 'bisect_tol': 0.1, 'bisect_fallback': True, 'reset_stuck_at': 0,
    'stop_on_convergence': True, 'check_every': 100, 'n_tau': 50.0, 'tau_rtol': 0.01, 'tau_checks': 2, 'min_steps': 200,
    'burn_factor': 3.0, 'tracks': 200, 'track_points': 300, 'max_unexpected_errors': 10}
SOLVE_DEFAULTS = {'param': '', 'observable': 'Rp', 'range_sigma': 6.0, 'tol': 0.01, 'max_evals': 40, 'warm_start': True,
                  'derivative': 'forward', 'refit_steps': [25, 50, 100, 200, 400]}
EMPTY_LISTS = ('refit_steps',) # list options that may be empty (refit_steps = []: never refit)
PARALLEL_DEFAULTS = {'workers': 6, 'max_memory_gb': 8.0, 'check_available': True, 'cache': True, 'warmup': True}
RESOLUTION_DEFAULTS = {
    'n_steps': [10000, 30000, 100000, 300000, 1000000], 'n_atmodeller': [1000, 300, 100, 30, 10, 3, 1],
    'points': 6, 'source': 'auto', 'tol': 0.1, 'dlogl_tol': 0.5, 'perturb': 1e-6, 'shift_tol': 1e-3,
    'max_eval_seconds': 300.0}

# Allowed values of the string options (_choice); mantle_iron_type is model.mantle_iron.type
CHOICES = {
    # CPML: core-powered mass loss; 'phi kill': an escape flux that removes the whole atmosphere (work in progress)
    'mechanism': ('XUV', 'CPML', 'XUV+CPML', 'phi kill'),
    'flux_model': ('power law', 'phoenix', 'Johnstone'),
    'activity': ('low', 'medium', 'high'),
    'composition': ('ratios', 'legacy'),
    'backend': ('isocalc', 'toy'),
    'prior': ('uniform', 'loguniform', 'normal'),
    'space': ('linear', 'log10'),
    'kind': ('detection', 'upper', 'lower'),
    # de: DE 0.7 + DE(gamma0 = 1) 0.1 + DESnooker 0.2; de_only: DE 0.9 + DE(gamma0 = 1) 0.1; stretch: emcee's default
    'moves': ('de', 'stretch', 'de_only'),
    'init': ('prescan', 'prior', 'ball', 'bisect', 'solve'),
    'mantle_iron_type': ('static', 'dynamic'),
}


###_____Hard validity of the parameters_____###

_MELT_MASS_RANGE = None # (min, max) planet mass of the melt grid [M_earth], read once by melt_mass_range


def melt_mass_range():
    '''
    Returns the planet-mass range covered by the melt fraction grid, isofate/data/melt_fraction_grid.npz
    Atmodeller needs the mantle melt fraction whenever it runs without model.melt_fraction_override; the coupler
    interpolates it on this grid in Mp and surface temperature and raises outside it, so Mp must stay inside this
    range then. The file is read once.

    Output: (min, max) planet mass of the grid [M_earth]
    '''
    global _MELT_MASS_RANGE
    if _MELT_MASS_RANGE is None:
        path = importlib.resources.files('isofate.data').joinpath('melt_fraction_grid.npz')
        # as_file gives a real file path, even when the package is imported from a zip
        with importlib.resources.as_file(path) as p, np.load(p) as data:
            _MELT_MASS_RANGE = (float(data['mass_grid'].min()), float(data['mass_grid'].max()))
    return _MELT_MASS_RANGE


def hard_limits(name, model):
    '''
    Returns the physical validity range of a registry parameter: where the model can run, whatever the prior
    Prior bounds must lie inside it. Mp and age depend on the model settings: with Atmodeller on and no melt
    fraction override, Mp must lie inside the melt grid, ends included (melt_mass_range), else Mp > 0; the age must
    lie in (t0, MAX_AGE]. Names outside PARAMS, such as the sampled coordinate u_<observable> of solve mode, are
    unbounded (their support is their bounds).

    Inputs:
        - name: parameter name [str]
        - model: the model settings, cfg['model'] [dict]

    Output: (lo, hi, lo_open, hi_open), lo_open and hi_open True for an excluded end [tuple]
    '''
    if name not in PARAMS:
        return (-inf, inf, True, True)
    if name == 'Mp':
        if model['n_atmodeller'] > 0 and model['melt_fraction_override'] is False:
            return melt_mass_range() + (False, False)
        return (0.0, inf, True, True)
    if name == 'age':
        return (model['t0'], MAX_AGE, True, False) # the run goes from t0 to the age
    return PARAMS[name]['limits']


def within_limits(name, x, model):
    '''
    Returns True when x lies inside the hard validity range of name (hard_limits); None, NaN and inf never do
    '''
    lo, hi, lo_open, hi_open = hard_limits(name, model)
    if x is None or not math.isfinite(x):
        return False
    above = x > lo if lo_open else x >= lo
    below = x < hi if hi_open else x <= hi
    return above and below


def _limits_text(name, model):
    '''
    Returns the hard validity range as text for the error messages, e.g. '(0, 1]'
    '''
    lo, hi, lo_open, hi_open = hard_limits(name, model)
    return f"{'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}"


###_____Small validators_____###

def _check_keys(d, allowed, where):
    '''
    Checks that d is a table and that every key of it is allowed
    An unknown key raises ConfigError suggesting up to 3 close names (case-insensitive), e.g. "unknown key
    'n_step' in [model]; did you mean 'n_steps'?", or listing the allowed keys when none is close.

    Inputs:
        - d: a table of the raw TOML [dict]
        - allowed: the allowed keys; a dict gives its keys
        - where: the table, for the message, e.g. '[model]' [str]
    '''
    if not isinstance(d, dict):
        raise ConfigError(f'{where} must be a table, got {d!r}')
    allowed = list(allowed)
    lower = {a.lower(): a for a in allowed} # match in lower case, suggest in the original case
    for key in d:
        if key in allowed:
            continue
        close = difflib.get_close_matches(key.lower(), list(lower), n = 3, cutoff = 0.6)
        if close:
            hint = '; did you mean ' + ' or '.join(repr(lower[c]) for c in close) + '?'
        else:
            hint = '; allowed: ' + ', '.join(allowed)
        raise ConfigError(f"unknown key '{key}' in {where}{hint}")


def _float(value, where, positive = False, nonnegative = False):
    '''
    Returns value as a float, or raises ConfigError
    Booleans and NaN are refused. inf passes (-inf only without positive or nonnegative), so callers that need a
    finite number check it themselves.

    Inputs:
        - value: the TOML value
        - where: dotted key for the message, e.g. 'star.mass' [str]
        - positive: True to require value > 0 [Bool]
        - nonnegative: True to require value >= 0 [Bool]

    Output: value [float]
    '''
    # bool is a subclass of int in Python: without this, true would pass as 1
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'{where} must be a number, got {value!r}')
    value = float(value)
    if math.isnan(value):
        raise ConfigError(f'{where} is NaN')
    if positive and not value > 0:
        raise ConfigError(f'{where} must be > 0, got {value:g}')
    if nonnegative and not value >= 0:
        raise ConfigError(f'{where} must be >= 0, got {value:g}')
    return value


def _int(value, where, minimum = None):
    '''
    Returns value as an int, or raises ConfigError
    Floats with whole values are accepted (n_steps = 1e5 is a float in TOML); NaN and inf are not, since int(inf)
    would raise OverflowError, a traceback instead of this message. Only floats are tested for finiteness: TOML
    integers can be arbitrarily large, and math.isfinite(10**400) raises OverflowError itself. Booleans are refused.

    Inputs:
        - value: the TOML value
        - where: dotted key for the message, e.g. 'model.n_steps' [str]
        - minimum: smallest allowed value; None: no minimum [int]

    Output: value [int]
    '''
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value)) or value != int(value)):
        raise ConfigError(f'{where} must be a whole number, got {value!r}')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f'{where} must be >= {minimum}, got {value}')
    return value


def _bool(value, where):
    '''
    Returns value if it is a TOML boolean (true or false), else raises ConfigError; 0 and 1 are refused
    '''
    if not isinstance(value, bool):
        raise ConfigError(f'{where} must be true or false, got {value!r}')
    return value


def _choice(value, key, where):
    '''
    Returns value if it is one of CHOICES[key], else raises ConfigError suggesting the closest choice
    '''
    if value not in CHOICES[key]:
        close = difflib.get_close_matches(str(value), CHOICES[key], n = 1)
        hint = f'; did you mean {close[0]!r}?' if close else ''
        raise ConfigError(f'{where} must be one of {", ".join(map(repr, CHOICES[key]))}, got {value!r}{hint}')
    return value


###_____Sections_____###

def _output_root(value, base):
    '''
    Returns the absolute path of the results folder
    None (no output_root in the config) gives ./mcmc_results in the current working directory; a relative path
    is relative to base; ~ is expanded.

    Inputs:
        - value: output_root of the config, or None [str]
        - base: the config file's directory (the working directory for a raw dict) [str]

    Output: absolute path [str]
    '''
    if value is None:
        return os.path.abspath(os.path.join(os.getcwd(), 'mcmc_results'))
    if not isinstance(value, str):
        raise ConfigError(f'output_root must be a path string, got {value!r}')
    path = os.path.expanduser(value)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.abspath(path)


def _star(raw):
    '''
    Validates [star] and converts it to SI
    mass is required. A given luminosity is used as it is; else it comes from radius and teff,
    L = 4 pi R^2 sigma Teff^4, and without those it is an error.

    Inputs:
        - raw: the raw [star] table, in solar units and K [dict]

    Output: the star settings [dict]
        - M: stellar mass [kg]
        - R: stellar radius [m]; None when not given
        - Teff: effective temperature [K]; None when not given
        - L: bolometric luminosity [W]
    '''
    _check_keys(raw, STAR_KEYS, '[star]')
    if 'mass' not in raw:
        raise ConfigError('[star] mass [M_sun] is required')
    M = _float(raw['mass'], 'star.mass', positive = True)*Ms # [kg]
    R = _float(raw['radius'], 'star.radius', positive = True)*Rs if 'radius' in raw else None # [m]
    Teff = _float(raw['teff'], 'star.teff', positive = True) if 'teff' in raw else None # [K]
    if 'luminosity' in raw:
        L = _float(raw['luminosity'], 'star.luminosity', positive = True)*Ls # [W]
    elif R is not None and Teff is not None:
        L = float(Luminosity(R, Teff)) # 4 pi R^2 sigma Teff^4 [W]
    else:
        raise ConfigError('[star] needs luminosity [L_sun], or radius [R_sun] and teff [K]')
    return {'M': M, 'R': R, 'Teff': Teff, 'L': L}


def _planet(raw, star, warnings):
    '''
    Validates [planet] and converts it to SI
    The semi-major axis comes from Kepler's third law for the period and the stellar mass, unless semi_major_axis
    is given; when both are given and differ by more than 1 %, semi_major_axis wins, with a warning. One of the two
    is required. Fp defaults to the bolometric insolation L/(4 pi d^2).

    Inputs:
        - raw: the raw [planet] table [dict]
        - star: the star settings (_star) [dict]
        - warnings: list the warnings are appended to [list]

    Output: the planet settings [dict]
        - P: orbital period [s]; None when not given
        - d: semi-major axis [m]
        - Fp: bolometric insolation [W/m2]
    '''
    _check_keys(raw, PLANET_KEYS, '[planet]')
    P = _float(raw['period'], 'planet.period', positive = True)/s2day if 'period' in raw else None # [s]
    if 'semi_major_axis' in raw:
        d = _float(raw['semi_major_axis'], 'planet.semi_major_axis', positive = True)*au2m # [m]
        mismatch = abs(SemiMajor(star['M'], P)/d - 1) if P is not None else 0.0 # relative to Kepler's law
        if mismatch > 0.01:
            warnings.append(f"semi_major_axis differs by {100*mismatch:.1f}% from Kepler's law for the period and "
                            'stellar mass; using semi_major_axis')
    elif P is not None:
        d = float(SemiMajor(star['M'], P)) # Kepler's third law [m]
    else:
        raise ConfigError('[planet] needs period [days] or semi_major_axis [AU]')
    Fp = _float(raw['Fp'], 'planet.Fp', positive = True) if 'Fp' in raw else float(Insolation(star['L'], d)) # [W/m2]
    return {'P': P, 'd': d, 'Fp': Fp}


def _model(raw, warnings):
    '''
    Validates [model], filling in MODEL_DEFAULTS
    Checks the flags and the choices; n_steps >= 1, and at most MAX_N_STEPS unless allow_large_n_steps (the memory
    of a worker grows with n_steps); 0 <= n_atmodeller <= n_steps; t0 > 0 and rho_rcb > 0. Warns for the uncoupled
    model (n_atmodeller = 0) and for flux_model = 'Johnstone'. params.Rp_fixed may switch rad_evol off later
    (_params).

    Inputs:
        - raw: the raw [model] table [dict]
        - warnings: list the warnings are appended to [list]

    Output: the model settings, the keys of MODEL_DEFAULTS [dict], among them
        - n_steps: time steps of a run [int]
        - n_atmodeller: time steps between Atmodeller calls; 0: Atmodeller off [int]
        - t0: start of the run [yr]
        - rho_rcb: gas density at the radiative-convective boundary, for CPML [kg/m3]
        - melt_fraction_override: false (melt fraction from the grid), or a melt fraction in (0, 1] [ndim]
        - mantle_iron: false, or {'type': 'static' or 'dynamic', 'Fe_mass_fraction': x}: mantle Fe that reacts
          with O2 [dict]
    '''
    _check_keys(raw, MODEL_DEFAULTS, '[model]')
    model = dict(MODEL_DEFAULTS)
    model.update(raw)
    for key in ('step_fn', 'RR', 'rad_evol', 'thermal', 'dynamic_phi', 'allow_large_n_steps'):
        model[key] = _bool(model[key], f'model.{key}')
    for key in ('mechanism', 'flux_model', 'activity', 'composition', 'backend'):
        _choice(model[key], key, f'model.{key}')
    # stellar_type is free text: only flux_model = 'Johnstone' reads it
    if not isinstance(model['stellar_type'], str):
        raise ConfigError(f"model.stellar_type must be a string, got {model['stellar_type']!r}")
    model['n_steps'] = _int(model['n_steps'], 'model.n_steps', minimum = 1)
    if model['n_steps'] > MAX_N_STEPS and not model['allow_large_n_steps']:
        raise ConfigError(f"model.n_steps = {model['n_steps']:.3g} is above {MAX_N_STEPS:.0e}; set "
                          'model.allow_large_n_steps = true if you really mean it (memory: about 1.1 GB per 1e6 '
                          'steps per worker)')
    model['n_atmodeller'] = _int(model['n_atmodeller'], 'model.n_atmodeller', minimum = 0)
    if model['n_atmodeller'] > model['n_steps']:
        raise ConfigError(f"model.n_atmodeller = {model['n_atmodeller']} must be <= n_steps = {model['n_steps']}")
    if model['n_atmodeller'] == 0:
        warnings.append('uncoupled: Atmodeller off (n_atmodeller = 0)')
    model['t0'] = _float(model['t0'], 'model.t0', positive = True)
    model['rho_rcb'] = _float(model['rho_rcb'], 'model.rho_rcb', positive = True)
    if model['flux_model'] == 'Johnstone':
        warnings.append("flux_model = 'Johnstone' reads a hard-coded data path and its phi_E has a units bug "
                        '(no 1/(4 Vpot)); do not use it for fits')
    # melt_fraction_override = false: the melt fraction comes from the grid, which limits Mp (hard_limits)
    melt = model['melt_fraction_override']
    if melt is not False:
        melt = _float(melt, 'model.melt_fraction_override')
        if not 0 < melt <= 1:
            raise ConfigError('model.melt_fraction_override must be false or in (0, 1] (isocalc treats 0.0 as '
                              f'false and uses the melt grid), got {melt:g}')
        model['melt_fraction_override'] = melt
    # Mantle Fe that reacts with O2, passed to isocalc as mantle_iron_dict
    iron = model['mantle_iron']
    if iron is not False:
        _check_keys(iron, ('type', 'Fe_mass_fraction'), '[model.mantle_iron]')
        if 'type' not in iron or 'Fe_mass_fraction' not in iron:
            raise ConfigError('model.mantle_iron must be false or {type = "static"|"dynamic", Fe_mass_fraction = x}')
        _choice(iron['type'], 'mantle_iron_type', 'model.mantle_iron.type')
        fraction = _float(iron['Fe_mass_fraction'], 'model.mantle_iron.Fe_mass_fraction', nonnegative = True)
        model['mantle_iron'] = {'type': iron['type'], 'Fe_mass_fraction': fraction}
    return model


def _bounds(value, where):
    '''
    Returns bounds as [lo, hi] floats with lo < hi; None stays None; anything else raises ConfigError
    '''
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f'{where} must be [lo, hi], got {value!r}')
    lo, hi = _float(value[0], where), _float(value[1], where)
    if not lo < hi:
        raise ConfigError(f'{where} = [{lo:g}, {hi:g}] needs lo < hi')
    return [lo, hi]


def _params(raw, model, warnings):
    '''
    Validates the [params.NAME] tables, for every parameter of the registry, in registry order
    A parameter the config leaves out is fixed at its registry default (Mp has none: it is required, and the
    optional Rp_fixed stays unset); a bare number, NAME = x, fixes it at x.
    A free parameter needs bounds for a uniform or loguniform prior (bounds > 0 for loguniform), and mu and sigma
    for a normal one (bounds then truncate it). Its bounds, or mu of a normal prior without bounds, must lie inside
    the hard validity range (hard_limits). A free parameter without a value gets the centre of its prior: mu
    (normal, when mu lies inside the bounds), the geometric mean of the bounds (loguniform), else their midpoint.
    The legacy composition takes only log_OtoH_enhancement, with a warning when it is not 0. Setting Rp_fixed
    switches model['rad_evol'] to false, in place, with a warning.

    Inputs:
        - raw: the raw [params] table [dict]
        - model: the model settings (_model); rad_evol may be changed [dict]
        - warnings: list the warnings are appended to [list]

    Output: {name: entry} for every name in PARAMS [dict]; each entry holds
        - vary: True for a free parameter [Bool]
        - value: the fixed value, or the centre of a free one (init = 'ball', check); None for an unset optional one
        - prior: 'uniform', 'loguniform' or 'normal'; 'fixed' when not free [str]
        - bounds: [lo, hi], or None [list]
        - mu, sigma: of a normal prior; None when not given
        - init_scale: spread of init = 'ball'; None for the default
        - unit, label: from the registry [str]
    '''
    _check_keys(raw, PARAMS, '[params]')
    out = {}
    for name, reg in PARAMS.items():
        where = f'params.{name}'
        entry = raw.get(name, {}) # left out: fixed at the registry default
        if not isinstance(entry, dict): # shorthand: name = value, a fixed parameter
            entry = {'value': entry}
        _check_keys(entry, PARAM_KEYS, f'[{where}]')
        vary = _bool(entry.get('vary', False), f'{where}.vary')
        prior = _choice(entry.get('prior', 'uniform'), 'prior', f'{where}.prior')
        bounds = _bounds(entry.get('bounds'), f'{where}.bounds')
        mu = _float(entry['mu'], f'{where}.mu') if entry.get('mu') is not None else None
        sigma = _float(entry['sigma'], f'{where}.sigma', positive = True) if entry.get('sigma') is not None else None
        init_scale = None
        if entry.get('init_scale') is not None:
            init_scale = _float(entry['init_scale'], f'{where}.init_scale', positive = True)
        value = _float(entry['value'], f'{where}.value') if entry.get('value') is not None else None
        limits = _limits_text(name, model)
        if vary: # a free parameter: a complete prior, inside the hard limits
            if prior in ('uniform', 'loguniform') and bounds is None:
                raise ConfigError(f'{where}: a {prior} prior needs bounds = [lo, hi]')
            if prior == 'loguniform' and bounds[0] <= 0:
                raise ConfigError(f'{where}: a loguniform prior needs bounds > 0, got {bounds}')
            if prior == 'normal' and (mu is None or sigma is None):
                raise ConfigError(f'{where}: a normal prior needs mu and sigma')
            if bounds is not None and not (within_limits(name, bounds[0], model)
                                           and within_limits(name, bounds[1], model)):
                raise ConfigError(f'{where}.bounds = {bounds} lie outside the valid range {limits} of {name}')
            if prior == 'normal' and bounds is None and not within_limits(name, mu, model):
                raise ConfigError(f'{where}.mu = {mu:g} lies outside the valid range {limits} of {name}')
            if value is None: # the centre of the prior, where init = 'ball' and check start
                if prior == 'normal' and (bounds is None or bounds[0] <= mu <= bounds[1]):
                    value = mu
                elif prior == 'loguniform':
                    value = math.sqrt(bounds[0]*bounds[1])
                else:
                    value = 0.5*(bounds[0] + bounds[1])
            elif bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ConfigError(f'{where}.value = {value:g} lies outside its bounds {bounds}')
        else: # fixed: the value given, else the registry default
            prior = 'fixed'
            if value is None:
                value = reg['default']
            if value is None and not reg['optional']:
                raise ConfigError(f'{name} is required: set [params.{name}] value (in {reg["unit"]}), or vary it')
        if value is not None and not within_limits(name, value, model):
            raise ConfigError(f'{where}.value = {value:g} lies outside the valid range {limits} of {name}')
        out[name] = {'vary': vary, 'value': value, 'prior': prior, 'bounds': bounds, 'mu': mu, 'sigma': sigma,
                     'init_scale': init_scale, 'unit': reg['unit'], 'label': reg['label']}

    # The legacy recipe of isofate_MCMC.py has only an O/H enhancement
    if model['composition'] == 'legacy':
        for name in ('log_metallicity', 'log_CtoH_enhancement'):
            if out[name]['vary'] or out[name]['value'] != 0:
                raise ConfigError(f'{name} needs model.composition = "ratios"; the legacy composition only has '
                                  'log_OtoH_enhancement')
        p = out['log_OtoH_enhancement']
        if p['vary'] or p['value'] != 0:
            warnings.append('legacy composition: the O/H enhancement takes O mass out of the H budget, so the '
                            'actual O/H is not 10**x times protosolar (13x too high at x = 2) and N_H < 0 above '
                            'x = 2.03 (status 4); composition = "ratios" is linear in x')
    p = out['Rp_fixed']
    if (p['vary'] or p['value'] is not None) and model['rad_evol']:
        model['rad_evol'] = False # isocalc applies Rp_override only without radius evolution
        warnings.append('params.Rp_fixed is set: model.rad_evol is switched to false, so the radius stays at '
                        'Rp_fixed for the whole run (no Lopez & Fortney radius) and the Rp observable acts as a '
                        'prior on Rp_fixed')
    return out


def _xuv_step_warning(cfg, warnings):
    '''
    Warns when the step-function XUV flux drops inside the age prior
    With step_fn = true the XUV flux stays at F0 until t_sat and then drops to F_final. Ages just below t_sat
    still see the saturated flux today, which can make a separate, narrow mode of the posterior that prior draws
    rarely hit (on LHS 1140 b a 28-Myr sliver of ages below t_sat made a spurious second mode). Checked only for
    an XUV mechanism with a free age and a fixed t_sat.

    Inputs:
        - cfg: the cfg dict being built, with planet, model and params [dict]
        - warnings: list the warning is appended to [list]
    '''
    m, p = cfg['model'], cfg['params']
    age, t_sat = p['age'], p['t_sat']
    if not m['step_fn'] or 'XUV' not in m['mechanism'] or not age['vary'] or t_sat['vary']:
        return
    lo, hi = age['bounds'] if age['bounds'] is not None else hard_limits('age', m)[:2] # [yr]; no bounds: (t0, MAX_AGE]
    if not lo < t_sat['value'] < hi: # every age of the prior sees the same side of the drop
        return
    F0, F_final = cfg['planet']['Fp']*10**p['log_F0_ratio']['value'], p['F_final']['value'] # [W/m2]
    ratio = f' ({F0/F_final:.3g}x)' if F_final > 0 else '' # the size of the drop
    below = 100*(t_sat['value'] - lo)/(hi - lo) # [%] of the age prior before the drop
    warnings.append(f"step_fn = true: the XUV flux drops from F0 = {F0:.3g} to F_final = {F_final:.3g} W/m2{ratio} at "
                    f"t_sat = {t_sat['value']:.4g} yr, inside the age prior [{lo:.4g}, {hi:.4g}] yr. Ages below t_sat "
                    f'({below:.2g}% of the age prior) still see the saturated flux today and can form a separate mode '
                    'of the posterior that prior draws rarely hit; check it (e.g. scan the age around t_sat), or move '
                    'the age bound or t_sat if a saturated present-day flux is not physical')


def _pms_check(cfg, notes):
    '''
    Checks the pre-main-sequence XUV boost
    isofunks.Fxuv starts at pms_factor*F0 at t0 and falls as a power law to F0 at t_pms, so the boost is on only
    when t_pms > t0, and then pms_factor must be > 0 (the power law takes log10(pms_factor): 0 makes the flux NaN).
    By default both are 0, i.e. no boost.

    Inputs:
        - cfg: the cfg dict being built, with model and params [dict]
        - notes: list a note is appended to when pms_factor is set but has no effect [list]
    '''
    t0, p = cfg['model']['t0'], cfg['params'] # [yr]

    def extremes(name):
        # (lowest, highest) value a parameter can take: its value when fixed, its prior support when free
        e = p[name]
        if not e['vary']:
            return e['value'], e['value']
        return tuple(e['bounds']) if e['bounds'] is not None else hard_limits(name, cfg['model'])[:2]
    t_hi = extremes('t_pms')[1] # [yr]
    f_lo, f_hi = extremes('pms_factor')
    if t_hi > t0 and not f_lo > 0:
        raise ConfigError(f'the pre-main-sequence XUV boost is on (t_pms up to {t_hi:g} yr > t0 = {t0:g} yr), but '
                          f'pms_factor can be {f_lo:g}: it must be > 0 there (e.g. 100 for mid-to-late M dwarfs, '
                          'Ramirez & Kaltenegger 2014), or set t_pms = 0 for no boost')
    if t_hi <= t0 and f_hi > 0:
        notes.append(f'pms_factor = {f_hi:g} has no effect: the pre-main-sequence XUV boost is off while t_pms <= t0 '
                     f'({t_hi:g} yr)')


def _sigma(value, where):
    '''
    Returns a sigma: a positive number, or a (lo, hi) tuple of positive numbers for asymmetric errors (lo applies
    when the model lies below the data, hi above)
    '''
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ConfigError(f'{where} must be a number or [lo, hi], got {value!r}')
        return (_float(value[0], where, positive = True), _float(value[1], where, positive = True))
    return _float(value, where, positive = True)


def _observables(raw, cfg, warnings):
    '''
    Validates the [observables.NAME] tables and keeps the active ones (use = true), in registry order
    An active observable needs value and sigma; value must be finite, and > 0 in log10 space (sigma is then in
    dex). An observable that needs Atmodeller is an error with n_atmodeller = 0. Warns for T_surf (not directly
    observable) and for the Mp or age observable on a fixed parameter (it only adds a constant). Inactive tables
    are checked for unknown keys too, so a typo never goes unnoticed.

    Inputs:
        - raw: the raw [observables] table [dict]
        - cfg: the cfg dict being built, with model and params [dict]
        - warnings: list the warnings are appended to [list]

    Output: {name: entry} for the active observables [dict]; each entry holds
        - use: True [Bool]
        - value: the data, in the registry unit (also in log10 space)
        - sigma: a number or (lo, hi); in the unit of value in linear space, in dex in log10 space
        - space: 'linear' or 'log10' [str]
        - kind: 'detection', 'upper' or 'lower' (a one-sided limit at value with width sigma) [str]
        - floor, ceiling: log10-space clipping range of the model value; ceiling 1/floor unless the registry
          sets one, inf for floor 0
        - sigma_model: extra model error, added in quadrature, in the space of sigma
        - unit, label, needs_atmodeller, input: from the registry
    '''
    _check_keys(raw, OBSERVABLES, '[observables]')
    out = {}
    for name, reg in OBSERVABLES.items():
        if name not in raw:
            continue
        where = f'observables.{name}'
        entry = raw[name]
        _check_keys(entry, OBS_KEYS, f'[{where}]')
        if not _bool(entry.get('use', False), f'{where}.use'): # inactive: may omit value and sigma
            continue
        for key in ('value', 'sigma'):
            if entry.get(key) is None:
                raise ConfigError(f'{where}: use = true needs {key}')
        value = _float(entry['value'], f'{where}.value')
        if math.isinf(value):
            raise ConfigError(f'{where}.value must be finite')
        sigma = _sigma(entry['sigma'], f'{where}.sigma')
        space = _choice(entry.get('space', 'linear'), 'space', f'{where}.space')
        kind = _choice(entry.get('kind', 'detection'), 'kind', f'{where}.kind')
        # A config's floor overrides the registry's
        floor = _float(entry.get('floor', reg['floor']), f'{where}.floor', nonnegative = True)
        sigma_model = _float(entry.get('sigma_model', 0.0), f'{where}.sigma_model', nonnegative = True)
        if space == 'log10' and value <= 0:
            raise ConfigError(f'{where}: space = "log10" needs value > 0 (value is in {reg["unit"] or "linear"} '
                              'units; sigma is in dex)')
        if reg['needs_atmodeller'] and cfg['model']['n_atmodeller'] == 0:
            raise ConfigError(f'{where} needs Atmodeller: set model.n_atmodeller > 0 or switch it off')
        if name == 'T_surf':
            warnings.append('T_surf is the surface temperature at the base of the adiabat: not directly observable')
        if reg['input'] and name in cfg['params'] and not cfg['params'][name]['vary']:
            warnings.append(f'observable {name} compares to a fixed parameter: it only adds a constant')
        # 1/floor caps the inf of a ratio x/0; it follows the floor in use, so a config's own floor moves it too
        ceiling = reg['ceiling'] if reg['ceiling'] is not None else (1/floor if floor > 0 else inf)
        out[name] = {'use': True, 'value': value, 'sigma': sigma, 'space': space, 'kind': kind, 'floor': floor,
                     'ceiling': ceiling, 'sigma_model': sigma_model, 'unit': reg['unit'], 'label': reg['label'],
                     'needs_atmodeller': reg['needs_atmodeller'], 'input': reg['input']}
    return out


def _section(raw, defaults, where):
    '''
    Validates a plain section: fills in the defaults and coerces each value to the type of its default
    bool, int and float defaults go through _bool, _int and _float; a list default takes a list of whole numbers,
    which must not be empty unless its key is in EMPTY_LISTS. String defaults (e.g. nwalkers = 'auto') are left to
    the caller.

    Inputs:
        - raw: the raw table [dict]
        - defaults: the defaults of the section, e.g. SAMPLER_DEFAULTS [dict]
        - where: the table, for the messages, e.g. '[sampler]' [str]

    Output: the section settings [dict]
    '''
    _check_keys(raw, defaults, where)
    out = dict(defaults)
    out.update(raw)
    for key, default in defaults.items():
        name = f'{where.strip("[]")}.{key}'
        value = out[key]
        if isinstance(default, bool): # before int: bool is a subclass of int
            out[key] = _bool(value, name)
        elif isinstance(default, int):
            out[key] = _int(value, name)
        elif isinstance(default, float):
            out[key] = _float(value, name)
        elif isinstance(default, list):
            if not isinstance(value, (list, tuple)):
                none = ' (or [] for none)' if key in EMPTY_LISTS else ''
                raise ConfigError(f'{name} must be a list{none}, got {value!r}')
            if not value and key not in EMPTY_LISTS:
                raise ConfigError(f'{name} must be a non-empty list, got {value!r}')
            out[key] = [_int(v, name) for v in value]
    return out


def _registry_name(value, registry, where):
    '''
    Returns value if it names an entry of registry (PARAMS or OBSERVABLES), else raises ConfigError suggesting
    the closest name
    '''
    if isinstance(value, str) and value in registry:
        return value
    close = difflib.get_close_matches(str(value), list(registry), n = 1)
    hint = f'; did you mean {close[0]!r}?' if close else ''
    raise ConfigError(f'{where} = {value!r} is not a known name{hint}')


def bisect_target(cfg, s = None):
    '''
    Returns the target of the bisection initializer (init = 'bisect'), checked
    The bisection moves sampler.bisect_param of prior draws until sampler.bisect_observable matches its data, so
    the parameter must be free and the observable active and computed by the model (not an input); else
    ConfigError.

    Inputs:
        - cfg: the cfg dict [dict]
        - s: the [sampler] settings; default cfg['sampler'] (_sampler passes its own before that exists) [dict]

    Output: (param, observable) [tuple]
    '''
    s = cfg['sampler'] if s is None else s
    param, name = s['bisect_param'], s['bisect_observable']
    if param not in cfg['free']:
        raise ConfigError(f'sampler.bisect_param = {param!r} must be a free parameter (free: '
                          f"{', '.join(cfg['free']) or 'none'}); set --bisect-param or params.{param}.vary = true")
    model_obs = [n for n, o in cfg['observables'].items() if not o['input']]
    if name not in model_obs:
        raise ConfigError(f'sampler.bisect_observable = {name!r} must be an active observable computed by the model '
                          f"(active: {', '.join(model_obs) or 'none'}); set --bisect-observable or switch it on")
    return param, name


###_____Solve mode_____###

def u_name(observable):
    '''
    Returns the name of the sampled coordinate of solve mode, u_<observable> (e.g. u_Rp)
    '''
    return f'u_{observable}'


def solve_on(cfg):
    '''
    Returns True when solve mode is on ([solve] param set); a cfg saved before solve mode existed has no [solve],
    so it is off
    '''
    return bool(cfg.get('solve', {}).get('active'))


def sampled_names(cfg):
    '''
    Returns the names of the sampler's coordinates (theta of the chain) [list]
    These are the free parameters, with u_<observable> in place of the solved parameter in solve mode; a cfg saved
    without cfg['sampled'] gives the free parameters.
    '''
    return list(cfg.get('sampled', cfg['free']))


def _bare_core_warning(cfg, o, bounds, warnings):
    '''
    Warns when solve mode on Rp along log_f_atm leaves out the stripped planets
    Below the stripping edge (the initial log_f_atm under which the planet loses its whole atmosphere before the
    end of the run) every log_f_atm gives Rp = the bare-core radius, (Mp/M_earth)^(1/4) R_earth (Lopez & Fortney
    2014): a plateau of Rp, which cannot be inverted for log_f_atm. When the bare-core radius over the Mp prior
    (its bounds, else the Mp value) reaches into the u support, solve mode leaves out the posterior mass of
    stripped planets, which plain sampling includes. The warning gives how many sigma the bare core at the Mp
    value lies from the Rp datum.

    Inputs:
        - cfg: the cfg dict being built [dict]
        - o: the Rp observable, cfg['observables']['Rp'] [dict]
        - bounds: the u support [lo, hi], in the likelihood space of Rp [list]
        - warnings: list the warning is appended to [list]
    '''
    from isofate.constants import Me, Re
    from isofate.isofunks import R_core
    from isofate.mcmc import likelihood
    p = cfg['params']['Mp']
    masses = p['bounds'] if p['vary'] and p['bounds'] is not None else [p['value']]
    cores = [R_core(m*Me)/Re for m in masses] # [R_earth]; R_core rises with Mp, so the ends of the prior suffice
    if max(cores) < bounds[0] or min(cores) > bounds[1]: # the plateau lies outside the u support
        return
    core = R_core(p['value']*Me)/Re
    # Distance of the bare core from the datum [sigma], with the sigma of the side it lies on
    d = (core - likelihood.observed(o))/likelihood.sigma_eff(o, above = core > likelihood.observed(o))
    span = f'{min(cores):.4g}-{max(cores):.4g} R_earth over the Mp prior' if len(cores) > 1 else f'{core:.4g} R_earth'
    warnings.append(f'solve mode on Rp along log_f_atm: a planet that loses its atmosphere has Rp = the bare-core '
                    f'radius ({span}) for every log_f_atm below the stripping edge. That plateau lies inside the u '
                    f'range [{bounds[0]:.4g}, {bounds[1]:.4g}] and cannot be inverted, so solve mode leaves out the '
                    f'posterior mass of stripped planets, which plain sampling includes (at Mp = {p["value"]:.4g} the '
                    f'bare core is {d:+.2g} sigma from the Rp datum). Sample without solve mode unless stripped '
                    'planets are ruled out (see "Solve mode" in the README)')


def _solve(raw, cfg, warnings, notes):
    '''
    Validates [solve], the settings of solve mode
    Solve mode samples u = T(model value of solve.observable), T the transform to its likelihood space (log10 for a
    log10-space observable), in place of the free parameter solve.param, which each evaluation then solves for.
    It is meant for one precise datum that pins one parameter to a thin, curved ridge, on which emcee's moves stall.
    Sampling u instead of the parameter P changes the density by the Jacobian |dP*/du| at the root P*, which
    solve.py adds, so the posterior over the physical parameters is unchanged (see "Solve mode" in the README). Off
    when solve.param = "" (the default).
    solve.param must be a free parameter and solve.observable an active observable computed by the model (not an
    input); tol is a fraction of sigma in (0, 1), max_evals >= 6 and derivative 'forward' or 'central'.
    The support of u is [v - k sigma_lo, v + k sigma_hi], v the datum in the likelihood space, sigma_lo and sigma_hi
    its effective sigma on each side and k = solve.range_sigma: k = 6 drops only points whose term is below -18. A
    limit keeps only its allowed side, [v - k sigma, v] (upper) or [v, v + k sigma] (lower), with a warning. Warns
    too when the support reaches the log10 floor of the observable, and for Rp along log_f_atm when the bare-core
    plateau lies inside it (_bare_core_warning).

    Inputs:
        - raw: the raw [solve] table [dict]
        - cfg: the cfg dict being built, with free and observables; cfg['sampled'] is set here [dict]
        - warnings: list the warnings are appended to [list]
        - notes: list the notes (information, not warnings) are appended to [list]

    Output: the solve settings [dict]: the keys of SOLVE_DEFAULTS (refit_steps sorted, without repeats), plus
        - active: True in solve mode [Bool]
        - index: position of param in cfg['free'], and of u in theta; None when off [int]
        - name: u_<observable>; None when off [str]
        - bounds: the support [lo, hi] of u, in the likelihood space of the observable; None when off [list]
        - label, unit: axis label and unit of u (dex in log10 space); None when off [str]
    '''
    from isofate.mcmc import likelihood
    s = _section(raw, SOLVE_DEFAULTS, '[solve]') # refit_steps is a list of whole numbers from here on
    for key in ('param', 'observable'):
        if not isinstance(s[key], str):
            raise ConfigError(f'solve.{key} must be a name (a string), got {s[key]!r}')
    if s['derivative'] not in ('forward', 'central'):
        raise ConfigError(f"solve.derivative must be \"forward\" or \"central\", got {s['derivative']!r}")
    s['refit_steps'] = sorted({_int(k, 'solve.refit_steps', minimum = 1) for k in s['refit_steps']})
    # The derived keys, None while solve mode is off
    s.update(active = bool(s['param']), index = None, name = None, bounds = None, label = None, unit = None)
    cfg['sampled'] = list(cfg['free']) # u takes the place of param below
    if not s['active']:
        return s
    if not s['warm_start'] and 'refit_steps' in raw and s['refit_steps']:
        notes.append('solve.warm_start = false: solve.refit_steps is unused (every solve starts from both prior '
                     'bounds, so there is no table to refit)')
    param = _registry_name(s['param'], PARAMS, 'solve.param')
    name = _registry_name(s['observable'], OBSERVABLES, 'solve.observable')
    if param not in cfg['free']:
        free = ', '.join(cfg['free']) or 'none'
        raise ConfigError(f'solve.param = {param!r} must be a free parameter (free: {free}); set params.{param}.vary = '
                          'true with a prior')
    model_obs = [n for n, o in cfg['observables'].items() if not o['input']]
    if name not in model_obs:
        raise ConfigError(f'solve.observable = {name!r} must be an active observable computed by the model (active: '
                          f"{', '.join(model_obs) or 'none'}); switch it on with value and sigma")
    if not s['range_sigma'] > 0:
        raise ConfigError(f"solve.range_sigma must be > 0, got {s['range_sigma']:g}")
    if not 0 < s['tol'] < 1:
        raise ConfigError(f"solve.tol must be in (0, 1) (a fraction of sigma), got {s['tol']:g}")
    s['max_evals'] = _int(s['max_evals'], 'solve.max_evals', minimum = 6)
    # Support of u: range_sigma sigmas around the datum, in the likelihood space (sigma_model included)
    o = cfg['observables'][name]
    k, v = s['range_sigma'], likelihood.observed(o)
    s_lo, s_hi = likelihood.sigma_eff(o, above = False), likelihood.sigma_eff(o, above = True)
    if o['kind'] == 'upper':
        bounds = [v - k*s_lo, v]
    elif o['kind'] == 'lower':
        bounds = [v, v + k*s_hi]
    else:
        bounds = [v - k*s_lo, v + k*s_hi]
    if o['kind'] != 'detection':
        warnings.append(f"solve mode on the {o['kind']} limit of {name}: u is sampled only in [{bounds[0]:.4g}, "
                        f"{bounds[1]:.4g}], between the limit and {k:g} sigma on its allowed side, so model values "
                        'beyond that, which the limit also allows, and the tail past the limit are cut away; solve '
                        'mode is meant for detections')
    # Every model value below the floor is clipped to it: a plateau of T at log10(floor)
    if o['space'] == 'log10' and o['floor'] > 0 and bounds[0] <= math.log10(o['floor']):
        warnings.append(f'solve mode: the u range of {name} reaches its log10 floor ({o["floor"]:g}), where the model '
                        'value cannot be inverted; lower solve.range_sigma or the floor')
    # With Rp_fixed the radius is constant: no bare-core plateau
    if name == 'Rp' and param == 'log_f_atm' and cfg['params']['Rp_fixed']['value'] is None:
        _bare_core_warning(cfg, o, bounds, warnings)
    reg = OBSERVABLES[name]
    log = o['space'] == 'log10'
    s.update(param = param, observable = name, index = cfg['free'].index(param), name = u_name(name),
             bounds = bounds, label = ('log$_{10}$ ' if log else '') + reg['label'] + ' (sampled)',
             unit = 'dex' if log else reg['unit'])
    cfg['sampled'][s['index']] = s['name'] # u in place of param in theta
    return s


###_____Sampler, parallel and resolution_____###

def _sampler(raw, cfg, warnings, notes):
    '''
    Validates [sampler], filling in SAMPLER_DEFAULTS
    In solve mode init defaults to 'solve', and 'bisect' becomes 'solve' with a note; init = 'solve' without solve
    mode is an error. nwalkers is 'auto' or an even number >= 2 ndim. The bisection names are always checked
    against the registries; with init = 'bisect' also their roles (bisect_target), and a limit as the target gives
    a warning, since the walkers then start on the boundary of the allowed region.
    The convergence settings: tau, the integrated autocorrelation time, is about the number of steps between two
    independent samples of a walker (Sokal 1997). It is estimated every check_every steps, and with
    stop_on_convergence the run stops once it has at least min_steps steps, more than n_tau max(tau), and tau has
    changed by less than max(tau_rtol, its statistical noise) at tau_checks checks in a row. The summary discards
    the first burn_factor max(tau) steps (at most half the chain) as burn-in.

    Inputs:
        - raw: the raw [sampler] table [dict]
        - cfg: the cfg dict being built, with free, observables and solve [dict]
        - warnings: list the warnings are appended to [list]
        - notes: list the notes (information, not warnings) are appended to [list]

    Output: the sampler settings, the keys of SAMPLER_DEFAULTS [dict]
    '''
    ndim = len(cfg['free'])
    s = _section(raw, SAMPLER_DEFAULTS, '[sampler]')
    if solve_on(cfg):
        if 'init' not in raw:
            s['init'] = 'solve' # the default in solve mode
        elif s['init'] == 'bisect': # the solve-mode init already starts every walker on a root
            s['init'] = 'solve'
            notes.append("sampler.init = 'bisect' is redundant in solve mode: init = 'solve' is used (draws of the "
                         'other parameters, u from the data distribution, each solved for '
                         f"{cfg['solve']['param']})")
    elif s['init'] == 'solve':
        raise ConfigError("sampler.init = 'solve' needs solve mode: set [solve] param (e.g. param = \"log_f_atm\", "
                          'observable = "Rp")')
    # emcee's ensemble moves update one group of walkers from the others (two halves for DE and stretch; Goodman &
    # Weare 2010, Foreman-Mackey et al 2013) and refuse fewer than 2 ndim walkers
    if s['nwalkers'] != 'auto':
        s['nwalkers'] = _int(s['nwalkers'], 'sampler.nwalkers')
        if s['nwalkers'] % 2 or s['nwalkers'] < 2*ndim:
            raise ConfigError(f"sampler.nwalkers = {s['nwalkers']} must be even and >= 2*ndim = {2*ndim}")
    _choice(s['moves'], 'moves', 'sampler.moves')
    _choice(s['init'], 'init', 'sampler.init')
    # _section made these whole numbers or floats already: only their ranges are checked here
    for key in ('max_steps', 'checkpoint_every', 'print_every', 'prescan_factor', 'check_every', 'bisect_iters',
                'tau_checks'):
        _int(s[key], f'sampler.{key}', minimum = 1)
    for key in ('reset_stuck_at', 'min_steps', 'tracks', 'max_unexpected_errors'):
        _int(s[key], f'sampler.{key}', minimum = 0)
    _int(s['track_points'], 'sampler.track_points', minimum = 2)
    for key in ('n_tau', 'tau_rtol', 'burn_factor'):
        _float(s[key], f'sampler.{key}', nonnegative = True)
    # Misspelt names fail now, even when init is not 'bisect'
    _registry_name(s['bisect_param'], PARAMS, 'sampler.bisect_param')
    _registry_name(s['bisect_observable'], OBSERVABLES, 'sampler.bisect_observable')
    if s['bisect_draws'] != 'auto':
        s['bisect_draws'] = _int(s['bisect_draws'], 'sampler.bisect_draws', minimum = 1)
    _float(s['bisect_tol'], 'sampler.bisect_tol', positive = True)
    if s['init'] == 'bisect':
        _, name = bisect_target(cfg, s)
        if cfg['observables'][name]['kind'] != 'detection':
            warnings.append(f"sampler.init = 'bisect' on the {cfg['observables'][name]['kind']} limit of {name}: the "
                            'bisection targets the limit value, so the walkers start on the boundary of the allowed '
                            'region')
    return s


def _parallel(raw):
    '''
    Validates [parallel], filling in PARALLEL_DEFAULTS: workers >= 1 and max_memory_gb > 0, the memory budget of
    all workers [GB] (the pool lowers the worker count to fit it)
    '''
    p = _section(raw, PARALLEL_DEFAULTS, '[parallel]')
    _int(p['workers'], 'parallel.workers', minimum = 1)
    _float(p['max_memory_gb'], 'parallel.max_memory_gb', positive = True)
    return p


def _resolution(raw, model):
    '''
    Validates [resolution], the settings of the resolution study (python -m isofate.mcmc resolution), filling in
    RESOLUTION_DEFAULTS
    The study runs test points at every (n_steps, n_atmodeller) cell of a grid and compares each cell with the
    finest one, in units of the measurement sigmas (see "Choosing n_steps and n_atmodeller" in the README). Every
    n_steps of the grid is >= 1 and at most MAX_N_STEPS unless model.allow_large_n_steps; every n_atmodeller >= 0;
    points >= 1; source one of RESOLUTION_SOURCES or 'chain:RUN_DIR'; tol, dlogl_tol, max_eval_seconds and
    shift_tol > 0; perturb >= 0.

    Inputs:
        - raw: the raw [resolution] table [dict]
        - model: the model settings (_model) [dict]

    Output: the resolution settings, the keys of RESOLUTION_DEFAULTS [dict]
    '''
    r = _section(raw, RESOLUTION_DEFAULTS, '[resolution]')
    for n in r['n_steps']:
        _int(n, 'resolution.n_steps', minimum = 1)
        if n > MAX_N_STEPS and not model['allow_large_n_steps']:
            raise ConfigError(f'resolution.n_steps contains {n:.3g} > {MAX_N_STEPS:.0e}; set '
                              'model.allow_large_n_steps = true if you really mean it')
    for n in r['n_atmodeller']:
        _int(n, 'resolution.n_atmodeller', minimum = 0)
    _int(r['points'], 'resolution.points', minimum = 1)
    if not isinstance(r['source'], str) or not (r['source'] in RESOLUTION_SOURCES or
                                                r['source'].startswith('chain:')):
        raise ConfigError(f"resolution.source must be one of {', '.join(map(repr, RESOLUTION_SOURCES))} or "
                          f"'chain:RUN_DIR', got {r['source']!r}")
    for key in ('tol', 'dlogl_tol', 'max_eval_seconds', 'shift_tol'):
        _float(r[key], f'resolution.{key}', positive = True)
    _float(r['perturb'], 'resolution.perturb', nonnegative = True)
    return r


# Test-point sources of a resolution study; 'chain:RUN_DIR' (posterior samples of a run) is allowed too
RESOLUTION_SOURCES = ('auto', 'prescan', 'prior', 'bisect', 'solve')


def resolution_source(cfg, source = None, solve = None):
    '''
    Returns the test-point source of a resolution study, checked
    'auto' is 'solve' for a study in solve mode; else 'bisect' when sampler.init is 'bisect' or 'solve' (the test
    points then sit where the walkers start), else 'prescan'. 'solve' needs a study in solve mode and 'bisect' one
    without it; 'bisect' also needs a valid bisection target (bisect_target). Anything else raises ConfigError.

    Inputs:
        - cfg: the cfg dict [dict]
        - source: the source asked for (--source); default resolution.source [str]
        - solve: True for a study in solve mode (--solve, --no-solve); default solve_on(cfg) [Bool]

    Output: 'prescan', 'prior', 'bisect', 'solve' or 'chain:RUN_DIR' [str]
    '''
    solve = solve_on(cfg) if solve is None else bool(solve)
    source = cfg['resolution']['source'] if source is None else str(source)
    if source == 'auto':
        source = 'solve' if solve else 'bisect' if cfg['sampler']['init'] in ('bisect', 'solve') else 'prescan'
    if source not in RESOLUTION_SOURCES[1:] and not source.startswith('chain:'): # 'auto' is resolved by now
        raise ConfigError(f"source must be 'prescan', 'prior', 'bisect', 'solve' or 'chain:RUN_DIR', got {source!r}")
    if source == 'solve' and not solve:
        raise ConfigError("source = 'solve' needs a study in solve mode ([solve] param set, without --no-solve)")
    if source == 'bisect' and solve:
        raise ConfigError("source = 'bisect' is for studies without solve mode: use --source solve (the default), or "
                          '--no-solve')
    if source == 'bisect':
        bisect_target(cfg) # a bad bisection target fails before any model run
    return source


###_____Loading_____###

def normalize(raw, config_path = None):
    '''
    Validates a raw TOML dict and returns the normalized cfg dict
    Works on a copy of raw. The sections are validated in the order they depend on each other: star, planet (needs
    the star mass and luminosity), model, params (need the model settings for their hard limits), observables,
    solve, sampler (its init depends on solve mode), parallel, resolution. Warns when output_root lies inside the
    IsoFATE git checkout, when no parameter is free and when no observable is active (the posterior is the prior).

    Inputs:
        - raw: the TOML as a dict, overrides already applied [dict]
        - config_path: path of the config file; None for a raw dict, which takes the name 'run' and the working
          directory as the base of relative paths [str]

    Output: cfg [dict]
        - name: run name, used in paths (letters, digits and _.+-); default the file stem [str]
        - description [str]
        - config_path: absolute path of the config, or None [str]
        - output_root: absolute path of the results folder [str]
        - star, planet: the SI settings of _star and _planet [dict]
        - model, solve, sampler, parallel, resolution: the validated sections [dict]
        - params: every registry parameter (_params) [dict]
        - free: names of the free parameters, in theta order [list]
        - sampled: names of the sampler's coordinates, u_<observable> in place of the solved parameter [list]
        - observables: the active observables (_observables) [dict]
        - overrides: [] here; load_config records the --set strings [list]
        - warnings: messages shown before a run [list]
        - notes: information, not warnings (printed by check, logged by run) [list]
    '''
    raw = copy.deepcopy(raw)
    _check_keys(raw, TOP_KEYS, 'the top level')
    warnings = []
    config_path = os.path.abspath(config_path) if config_path else None
    base = os.path.dirname(config_path) if config_path else os.getcwd() # relative paths start here
    stem = os.path.splitext(os.path.basename(config_path))[0] if config_path else 'run'
    name = raw.get('name', stem)
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.+-]+', name):
        raise ConfigError(f'name must be a short string of letters, digits and _.+- (used in paths), got {name!r}')
    cfg = {'name': name, 'description': str(raw.get('description', '')), 'config_path': config_path,
           'output_root': _output_root(raw.get('output_root'), base)}
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # the IsoFATE checkout
    # Results inside a git checkout clutter it; an installed package has no .git, so no warning then
    if os.path.isdir(os.path.join(repo, '.git')) and (cfg['output_root'] + os.sep).startswith(repo + os.sep):
        warnings.append(f"output_root {cfg['output_root']} is inside the IsoFATE repository (output_root "
                        f"{'not set: ./mcmc_results' if 'output_root' not in raw else 'in the config'}); set "
                        'output_root to a results folder outside it')
    cfg['star'] = _star(raw.get('star', {}))
    cfg['planet'] = _planet(raw.get('planet', {}), cfg['star'], warnings)
    cfg['model'] = _model(raw.get('model', {}), warnings)
    cfg['params'] = _params(raw.get('params', {}), cfg['model'], warnings)
    cfg['free'] = [name for name in PARAMS if cfg['params'][name]['vary']] # theta order: the registry order
    _xuv_step_warning(cfg, warnings)
    notes = []
    _pms_check(cfg, notes)
    cfg['observables'] = _observables(raw.get('observables', {}), cfg, warnings)
    cfg['solve'] = _solve(raw.get('solve', {}), cfg, warnings, notes)
    cfg['sampler'] = _sampler(raw.get('sampler', {}), cfg, warnings, notes)
    cfg['parallel'] = _parallel(raw.get('parallel', {}))
    cfg['resolution'] = _resolution(raw.get('resolution', {}), cfg['model'])
    if not cfg['free']:
        warnings.append('no free parameters (set vary = true in a [params.NAME] table)')
    if not cfg['observables']:
        warnings.append('no active observables: the posterior is the prior')
    cfg['overrides'] = []
    cfg['warnings'] = warnings
    cfg['notes'] = notes # information, not warnings (printed by check, logged by run)
    return cfg


def parse_value(text):
    '''
    Returns the TOML value of an override's text: '1e5' -> 1e5, 'true' -> True, '[1, 2]' -> [1, 2],
    '"XUV"' -> 'XUV'; anything that is not a TOML value is taken as a string ('toy' -> 'toy')
    '''
    try:
        return tomllib.loads(f'v = {text}')['v']
    except tomllib.TOMLDecodeError:
        return text


def _override_items(overrides):
    '''
    Returns the overrides as (dotted key, value) pairs
    The string values of a dict are parsed as TOML (parse_value), its other values kept. Each string of a list
    must look like section.key=value, else ConfigError.
    '''
    if isinstance(overrides, dict):
        return [(key, parse_value(value) if isinstance(value, str) else value) for key, value in overrides.items()]
    items = []
    for text in overrides:
        key, sep, value = text.partition('=') # at the first '=': a value may contain '=' itself
        if not sep or not key.strip():
            raise ConfigError(f"override {text!r} must look like section.key=value")
        items.append((key.strip(), parse_value(value.strip())))
    return items


def apply_overrides(raw, overrides):
    '''
    Sets dotted keys ('model.n_steps', 'params.age.bounds', ...) in the raw TOML dict, in place
    Missing tables are created. A parameter written as a bare number (params.NAME = x) first becomes the table
    {value = x}, so that params.NAME.vary=true works on it; any other key on the path that is not a table raises
    ConfigError.

    Inputs:
        - raw: the raw TOML dict, changed in place [dict]
        - overrides: list of 'dotted.key=value' strings (--set), or a dict {dotted_key: value}; None: none

    Output: raw, the same dict [dict]
    '''
    for key, value in _override_items(overrides or []):
        parts = key.split('.')
        node = raw
        for i, part in enumerate(parts[:-1]): # walk down the tables of the path
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                if i == 1 and parts[0] == 'params': # the shorthand name = value becomes a table
                    child = node[part] = {'value': child}
                else:
                    raise ConfigError(f'override {key}: {".".join(parts[:i + 1])} is not a table')
            node = child
        node[parts[-1]] = value
    return raw


def _override_strings(overrides):
    '''
    Returns the overrides as 'key=value' strings with TOML values, which parse back as --set arguments (they go to
    cfg['overrides'], and from there to config.json and the top of a run's config.toml)
    '''
    return [f'{key}={_toml_value(value)}' for key, value in _override_items(overrides or [])]


def read_toml(path, overrides = None):
    '''
    Reads a config file as a raw TOML dict, with the overrides applied (synth uses it to write the file back)
    A TOML syntax error raises ConfigError with the file name and tomllib's message.

    Inputs:
        - path: the config file [str]
        - overrides: as in apply_overrides

    Output: the raw TOML dict [dict]
    '''
    with open(path, 'rb') as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f'{path}: {e}') from None
    return apply_overrides(raw, overrides)


def load_config(path, overrides = None):
    '''
    Loads, validates and normalizes a TOML config (or a raw dict with the same structure)
    Every command that reads a TOML config loads it here. The overrides are applied to the raw TOML before
    validation and recorded in cfg['overrides'] as --set strings. A dict is copied first, so the caller's dict is
    not changed.

    Inputs:
        - path: the config file, or a raw dict [str or dict]
        - overrides: list of 'dotted.key=value' strings (--set), or a dict {dotted_key: value}; None: none

    Output: the normalized cfg dict (see normalize) [dict]
    '''
    if isinstance(path, dict):
        raw, config_path = apply_overrides(copy.deepcopy(path), overrides), None
    else:
        raw, config_path = read_toml(path, overrides), path
    cfg = normalize(raw, config_path)
    cfg['overrides'] = _override_strings(overrides)
    return cfg


###_____nwalkers_____###

MIN_AUTO_WALKERS = 16 # smaller ensembles mix poorly (8 walkers: DESnooker groups of 2)


def auto_nwalkers(ndim, workers, moves = 'de'):
    '''
    Returns the automatic number of walkers (sampler.nwalkers = 'auto')
    max(4*ndim, 16), rounded up to a multiple of 4*workers when the moves include DESnooker (moves = 'de'; it maps
    the walkers in 4 groups), else of 2*workers (the two halves of the red-blue moves), so every worker is busy in
    each map. 6 workers, 3-6 dims: 24; 1-2 workers and up to 4 dims: 16.

    Inputs:
        - ndim: number of free parameters [int]
        - workers: number of worker processes [int]
        - moves: sampler.moves [str]

    Output: nwalkers [int]
    '''
    step = 4*workers if moves == 'de' else 2*workers # every group of the ensemble a multiple of workers
    return int(math.ceil(max(4*ndim, MIN_AUTO_WALKERS)/step)*step)


def choose_nwalkers(cfg, workers):
    '''
    Returns the number of walkers of a run: sampler.nwalkers, or the auto rule for the actual number of workers
    Raises ConfigError without free parameters, and for an odd number or fewer than 2 ndim walkers.

    Inputs:
        - cfg: the cfg dict [dict]
        - workers: the number of workers the pool got (the memory budget can lower it) [int]

    Output: nwalkers [int]
    '''
    ndim = len(cfg['free'])
    if ndim == 0:
        raise ConfigError('no free parameters to sample (set vary = true in a [params.NAME] table)')
    n = cfg['sampler']['nwalkers']
    if n == 'auto':
        n = auto_nwalkers(ndim, workers, cfg['sampler']['moves'])
    if n % 2 or n < 2*ndim:
        raise ConfigError(f'nwalkers = {n} must be even and >= 2*ndim = {2*ndim}')
    return n


###_____Documentation_____###

def _limits_doc(name):
    '''
    Returns the valid range of a parameter as README text, e.g. '(0, 1]', '>= 0' or 'any'; the ranges of Mp and
    age depend on the model settings and are given in words
    '''
    if name == 'Mp':
        lo, hi = melt_mass_range()
        return f'[{lo:g}, {hi:g}] with Atmodeller and no melt_fraction_override (melt grid), else > 0'
    if name == 'age':
        return f'(t0, {MAX_AGE:g}]'
    lo, hi, lo_open, hi_open = PARAMS[name]['limits']
    if lo == -inf and hi == inf:
        return 'any'
    if hi == inf:
        return f"{'>' if lo_open else '>='} {lo:g}"
    if lo == -inf:
        return f"{'<' if hi_open else '<='} {hi:g}"
    return f"{'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}"


def markdown_tables():
    '''
    Returns the README's markdown tables of the PARAMS and OBSERVABLES registries, so the documentation cannot
    drift from the code (tests/test_mcmc_cli.py compares them with the README)

    Output: (params, observables) [tuple]
        - params: table of name, unit, default, valid range and meaning [str]
        - observables: table of name, unit, definition, log10 floor and needs Atmodeller [str]
    '''
    rows = ['| name | unit | default | valid range | meaning |', '|---|---|---|---|---|']
    for name, p in PARAMS.items():
        default = f"{p['default']:g}" if p['default'] is not None else 'unset' if p['optional'] else 'required'
        rows.append(f"| `{name}` | {p['unit'] or '-'} | {default} | {_limits_doc(name)} | {p['doc']} |")
    params = '\n'.join(rows)
    rows = ['| name | unit | definition | log10 floor | needs Atmodeller |', '|---|---|---|---|---|']
    for name, o in OBSERVABLES.items():
        if o['input']:
            floor = 'input'
        elif o['floor'] == 0:
            floor = '-'
        else:
            floor = f"{o['floor']:g}" + (' (no ceiling)' if o['ceiling'] == inf else '')
        rows.append(f"| `{name}` | {o['unit'] or '-'} | {o['doc']} | {floor} | "
                    f"{'yes' if o['needs_atmodeller'] else 'no'} |")
    return params, '\n'.join(rows)


###_____Output directories_____###

def new_output_dir(base, first = 1):
    '''
    Creates and returns a new output directory: base, else base_<first>, base_<first + 1>, ...
    Claimed atomically (os.makedirs without exist_ok), so two processes started in the same second, whose base
    carries the same time stamp, never share one.

    Inputs:
        - base: the directory wanted, e.g. <output_root>/<name>/mcmc_<stamp> [str]
        - first: suffix of the first alternative [int]

    Output: absolute path of the new directory [str]
    '''
    path, k = base, first
    while True:
        try:
            os.makedirs(path) # also creates <output_root>/<name>; fails if another process got there first
            return os.path.abspath(path)
        except FileExistsError:
            path, k = f'{base}_{k}', k + 1


def empty_output_dir(path, what):
    '''
    Returns an output directory given explicitly: created if missing, refused (FileExistsError) if not empty

    Inputs:
        - path: the directory [str]
        - what: what writes there, for the message, e.g. 'a scan' [str]

    Output: absolute path [str]
    '''
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(f'{path} exists and is not empty; {what} never overwrites one')
    os.makedirs(path, exist_ok = True)
    return os.path.abspath(path)


###_____Writing_____###

_BARE_KEY = re.compile(r'[A-Za-z0-9_-]+') # TOML bare keys; any other key is written quoted


def _toml_key(key):
    '''
    Returns key as a TOML key: bare when it may be, else a quoted string (TOML basic strings share the escapes of
    json.dumps)
    '''
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii = False)


def _toml_value(v):
    '''
    Returns v as TOML text, so that tomllib reads it back unchanged
    Takes bool, int, float (nan and +-inf included), str, list, tuple, array and dict (an inline table, None
    values left out), numpy scalars too; anything else raises TypeError. Floats keep their shortest exact form.
    '''
    if isinstance(v, (bool, np.bool_)): # before int: bool is a subclass of int
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if math.isnan(v): # TOML has nan, inf and -inf
            return 'nan'
        if math.isinf(v):
            return 'inf' if v > 0 else '-inf'
        if abs(v) >= 1e5: # 3.5e9 rather than 3500000000.0 (still the shortest exact form)
            return np.format_float_scientific(v, unique = True, trim = '-', exp_digits = 1).replace('+', '')
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii = False)
    if isinstance(v, (list, tuple, np.ndarray)):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    if isinstance(v, dict): # an inline table: a dict inside a list, or an override value
        return '{' + ', '.join(f'{_toml_key(k)} = {_toml_value(x)}' for k, x in v.items() if x is not None) + '}'
    raise TypeError(f'cannot write {v!r} ({type(v).__name__}) as TOML')


def dump_toml(d):
    '''
    Returns d as TOML text: a minimal emitter for nested dicts of str, bool, int, float and lists (None values are
    left out)
    Each nested dict becomes a [table], written after the scalars of its parent, and a dict inside a list an inline
    table; tomllib reads the text back to d (tested on the template and the example configs too).

    Inputs:
        - d: the dict to write [dict]

    Output: TOML text, ending in a newline [str]
    '''
    lines = []

    def table(node, path):
        '''
        Appends one table to lines: its header, its scalars, then its subtables
        A table with only subtables needs no header of its own; an empty one gets just the header, so that it
        reads back as an empty table.
        '''
        scalars = [(k, v) for k, v in node.items() if not isinstance(v, dict) and v is not None]
        tables = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        if path and (scalars or not tables):
            lines.extend(['', '[' + '.'.join(_toml_key(p) for p in path) + ']'])
        lines.extend(f'{_toml_key(k)} = {_toml_value(v)}' for k, v in scalars)
        for k, v in tables:
            table(v, path + [k])

    table(d, [])
    return '\n'.join(lines).lstrip('\n') + '\n'


def _json_default(x):
    '''
    json.dump fallback: numpy scalars become Python numbers and arrays lists; anything else raises TypeError
    '''
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f'{type(x).__name__} is not JSON serializable')


def save_resolved(cfg, path):
    '''
    Writes the resolved cfg as JSON, e.g. a run's config.json
    Tuples become lists (load_resolved turns the asymmetric sigmas back into tuples); floats keep repr precision,
    so they read back exactly; inf and NaN are written as Infinity and NaN, which Python's json reads back. The
    file is written to path.tmp first and then renamed, so a crash never leaves a half-written one.

    Inputs:
        - cfg: the cfg dict [dict]
        - path: the JSON file [str]
    '''
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cfg, f, indent = 1, default = _json_default)
        f.write('\n')
    os.replace(tmp, path)


def load_resolved(path):
    '''
    Reads a cfg written by save_resolved, e.g. a run's config.json, restoring the asymmetric sigmas as tuples

    Inputs:
        - path: the JSON file [str]

    Output: the cfg dict [dict]
    '''
    with open(path) as f:
        cfg = json.load(f)
    for o in cfg['observables'].values():
        if isinstance(o['sigma'], list): # JSON has no tuples
            o['sigma'] = tuple(o['sigma'])
    return cfg
