'''
Tests of isofate.mcmc.config and isofate.mcmc.params. config: loading a TOML config, the defaults, validation
with its error messages and suggestions, --set overrides, the example configs, the solve-mode settings, and
writing TOML and config.json back. params: the log prior, the Latin-hypercube prior draws, jitter and the ball
start; and the nwalkers rule. Fast: no model runs.
'''

import copy
import math
import os
import tomllib

import numpy as np
import pytest
from scipy import stats

from isofate.constants import Ls, Ms, s2day
from isofate.orbit_params import EqTemp, Insolation, SemiMajor
from isofate.mcmc import config
from isofate.mcmc import params as prm
from isofate.mcmc.config import ConfigError


@pytest.fixture(autouse = True)
def outside_the_repo(tmp_path, monkeypatch):
    '''
    Runs each test from tmp_path. The default output_root is ./mcmc_results, which inside the repository adds a
    warning to the config.
    '''
    monkeypatch.chdir(tmp_path)


def minimal():
    '''
    A small valid raw config: LHS 1140 b-like (0.18 M_sun, 0.0038 L_sun, a 24.74 day orbit, Mp = 5.6 M_earth),
    log_f_atm free in [-5, -1.3], Rp observed (1.73 +- 0.025 R_earth)
    '''
    return {'name': 'test', 'star': {'mass': 0.18, 'luminosity': 0.0038}, 'planet': {'period': 24.74},
            'params': {'Mp': {'value': 5.6}, 'log_f_atm': {'vary': True, 'bounds': [-5.0, -1.3]}},
            'observables': {'Rp': {'use': True, 'value': 1.73, 'sigma': 0.025}}}


def load(overrides = None, raw = None):
    '''
    The resolved config of raw (default minimal()) with overrides (a dict or a list of strings)
    '''
    return config.load_config(raw or minimal(), overrides)


###_____config_____###

def test_defaults():
    '''
    Checks the resolved defaults of the minimal config: each section equals its defaults table, solve mode is off,
    a free parameter without a value starts at the middle of its bounds and fixed ones take the registry defaults.
    Rp defaults to a linear-space detection without floor or model error, and the star and planet are derived
    in SI (Fp = 587.6 W/m2 for LHS 1140 b).
    '''
    cfg = load()
    assert cfg['model'] == config.MODEL_DEFAULTS
    assert cfg['sampler'] == config.SAMPLER_DEFAULTS
    assert {k: cfg['solve'][k] for k in config.SOLVE_DEFAULTS} == config.SOLVE_DEFAULTS
    assert cfg['solve']['active'] is False and cfg['sampled'] == cfg['free'] and not config.solve_on(cfg)
    assert cfg['parallel'] == config.PARALLEL_DEFAULTS
    assert cfg['resolution'] == config.RESOLUTION_DEFAULTS
    assert list(cfg['params']) == list(config.PARAMS)
    assert cfg['free'] == ['log_f_atm']
    assert cfg['params']['log_f_atm']['value'] == pytest.approx(-3.15) # bounds midpoint
    assert cfg['params']['eps'] == {'vary': False, 'value': 0.15, 'prior': 'fixed', 'bounds': None, 'mu': None,
                                    'sigma': None, 'init_scale': None, 'unit': '', 'label': r'$\epsilon$'}
    for name, reg in config.PARAMS.items():
        if name not in ('log_f_atm', 'Mp'):
            assert cfg['params'][name]['value'] == reg['default']
    assert list(cfg['observables']) == ['Rp']
    rp = cfg['observables']['Rp']
    assert (rp['space'], rp['kind'], rp['floor'], rp['sigma_model'], rp['input']) == ('linear', 'detection', 0.0,
                                                                                   0.0, False)
    assert cfg['output_root'] == os.path.join(os.getcwd(), 'mcmc_results')
    assert cfg['warnings'] == []
    # derived star and planet, in SI
    P = 24.74/s2day # [s]
    assert cfg['star']['L'] == 0.0038*Ls and cfg['star']['M'] == 0.18*Ms
    assert cfg['planet']['P'] == P and cfg['planet']['d'] == SemiMajor(0.18*Ms, P)
    assert cfg['planet']['Fp'] == Insolation(0.0038*Ls, SemiMajor(0.18*Ms, P))
    assert cfg['planet']['Fp'] == pytest.approx(587.628, rel = 1e-5) # [W/m2]


def test_paths_and_name(tmp_path):
    '''
    Checks that a config without a name takes the name of its file, and that a relative output_root is relative
    to the config's folder, not to the working directory.
    '''
    raw = minimal()
    raw.pop('name')
    raw['output_root'] = '..'
    folder = tmp_path/'configs'
    folder.mkdir()
    path = folder/'my_planet.toml'
    path.write_text(config.dump_toml(raw))
    cfg = config.load_config(str(path))
    assert cfg['name'] == 'my_planet'
    assert cfg['output_root'] == str(tmp_path)
    assert cfg['config_path'] == str(path)


def test_example_configs():
    '''
    Checks the shipped configs: the template, and the LHS 1140 b example in full (free parameters, solve mode on
    the outflow H fraction, priors, data, sampler settings, walker counts), with the step-function XUV warning
    when t_sat falls inside the age prior. When the working copies in mcmc_results/configs (next to the
    repository) exist, it also checks that they hold the same physics.
    '''
    template = config.load_config(config.TEMPLATE)
    assert template['warnings'] == []
    assert template['free'] == ['log_f_atm', 'age', 'albedo']
    assert template['params']['albedo']['bounds'] == [0.0, 0.8] and template['params']['pms_factor']['value'] == 0.0
    cfg = config.load_config(os.path.join(config.CONFIG_DIR, 'lhs1140b.toml'))
    # the age prior starts at t_sat, so the step-function warning is silent
    assert cfg['warnings'] == [] and cfg['notes'] == []
    # with a lower bound of 3.1 Gyr, t_sat = 3.128 Gyr lies inside the age prior: ages below it would still see the
    # saturated XUV flux today (3.46 times F_final)
    sliver = config.load_config(os.path.join(config.CONFIG_DIR, 'lhs1140b.toml'), ['params.age.bounds=[3.1e9, 1e10]'])
    assert len(sliver['warnings']) == 1 and 'inside the age prior' in sliver['warnings'][0]
    assert '0.41% of the age prior' in sliver['warnings'][0] and '(3.46x)' in sliver['warnings'][0]
    assert config.load_config(os.path.join(config.CONFIG_DIR, 'lhs1140b.toml'), ['model.step_fn=false',
                              'params.age.bounds=[3.1e9, 1e10]'])['warnings'] == [] # no drop without step_fn
    assert cfg['free'] == ['log_f_atm', 'Mp', 'age', 'eps', 'albedo']
    sv = cfg['solve'] # solve mode on the outflow H fraction
    assert config.solve_on(cfg) and (sv['param'], sv['observable'], sv['index']) == ('log_f_atm',
                                                                                    'outflow_H_fraction', 0)
    assert cfg['sampled'] == ['u_outflow_H_fraction', 'Mp', 'age', 'eps', 'albedo']
    assert sv['name'] == 'u_outflow_H_fraction'
    v = np.log10(0.00115) # the datum in its log10 space
    assert sv['bounds'] == pytest.approx([v - 6*0.261, v + 6*0.240]) and (sv['tol'], sv['max_evals']) == (0.01, 40)
    assert list(cfg['observables']) == ['Mp', 'Rp', 'Mdot_HHe', 'outflow_H_fraction']
    assert cfg['model']['composition'] == 'legacy' and cfg['model']['step_fn'] and cfg['model']['dynamic_phi']
    assert cfg['model']['rad_evol'] and cfg['params']['Rp_fixed']['value'] is None # the radius evolves
    assert (cfg['model']['n_steps'], cfg['model']['n_atmodeller']) == (100000, 100)
    p = {name: q['value'] for name, q in cfg['params'].items()}
    assert (p['F_final'], p['t_sat'], p['log_F0_ratio'], p['eps']) == (0.17, 3.128e9, -3.0, 0.15)
    assert (p['t_pms'], p['pms_factor']) == (0.0, 0.0) # no pre-main-sequence XUV boost
    albedo = cfg['params']['albedo'] # free: Bond albedos of bare rock to Venus-like clouds
    assert (albedo['prior'], albedo['bounds'], albedo['value']) == ('uniform', [0.0, 0.8], 0.0)
    assert not cfg['params']['F_final']['vary']
    assert cfg['params']['log_f_atm']['bounds'] == [-5.0, -1.3]
    assert cfg['params']['age']['bounds'] == [3.128e9, 1e10] and cfg['params']['age']['bounds'][0] == p['t_sat']
    assert cfg['params']['Mp']['bounds'] == [4.5, 6.7]
    assert cfg['params']['eps']['bounds'] == [0.01, 1.0] and cfg['params']['eps']['prior'] == 'uniform'
    x, mdot, rp = (cfg['observables'][n] for n in ('outflow_H_fraction', 'Mdot_HHe', 'Rp'))
    assert (x['value'], x['space'], x['sigma'], x['floor']) == (0.00115, 'log10', (0.261, 0.240), 1e-15)
    assert (mdot['value'], mdot['space'], mdot['sigma'], mdot['ceiling']) == (2.03e5, 'log10', (0.137, 0.121), math.inf)
    assert (rp['value'], rp['sigma'], rp['sigma_model']) == (1.73, 0.025, 0.0)
    s = cfg['sampler']
    assert (s['init'], s['bisect_observable'], s['bisect_param'], s['reset_stuck_at']) == (
        'solve', 'outflow_H_fraction', 'log_f_atm', 0)
    assert config.resolution_source(cfg) == 'solve' and config.resolution_source(cfg, solve = False) == 'bisect'
    assert EqTemp(cfg['planet']['Fp']) == pytest.approx(225.61, abs = 0.01) # [K]
    assert config.choose_nwalkers(cfg, 6) == 24 and config.choose_nwalkers(cfg, 3) == 24
    # the working copies in the results folder: the same physics; static_phi differs in dynamic_phi only. Both
    # budget 6 GB for the workers without checking the available memory
    results = os.path.join(os.path.dirname(config.__file__), '..', '..', '..', 'mcmc_results', 'configs')
    if os.path.exists(os.path.join(results, 'lhs1140b.toml')):
        work = config.load_config(os.path.join(results, 'lhs1140b.toml'))
        static = config.load_config(os.path.join(results, 'lhs1140b_static_phi.toml'))
        for key in ('star', 'planet', 'params', 'free', 'observables', 'solve', 'sampled', 'sampler', 'resolution'):
            assert work[key] == cfg[key] == static[key], key
        assert work['parallel'] == static['parallel'] == dict(cfg['parallel'], max_memory_gb = 6.0,
                                                               check_available = False)
        assert work['model'] == cfg['model'] and static['model'] == dict(cfg['model'], dynamic_phi = False)
        assert (work['name'], static['name']) == ('lhs1140b', 'lhs1140b_static_phi')
        assert work['output_root'] == static['output_root'] == os.path.abspath(os.path.join(results, '..'))


@pytest.mark.parametrize('where, key, hint', [
    ('model', 'n_step', 'n_steps'),
    ('sampler', 'nwalker', 'nwalkers'),
    ('params.age', 'bound', 'bounds'),
    ('observables.Rp', 'sgima', 'sigma'),
    ('params', 'log_fatm', 'log_f_atm'),
    ('observables', 'rp', 'Rp'),
    ('', 'sampelr', 'sampler'), # a top-level key
])
def test_unknown_key_suggestions(where, key, hint):
    '''
    Checks that a misspelt key anywhere in a config (a section, an option, a parameter or observable name, or a key
    of their tables) is an error that suggests the right name.
    '''
    raw = minimal()
    node = raw
    for part in filter(None, where.split('.')):
        node = node.setdefault(part, {})
    node[key] = 1
    with pytest.raises(ConfigError, match = f"unknown key '{key}'.*did you mean '{hint}'"):
        load(raw = raw)


@pytest.mark.parametrize('overrides, match', [
    ({'model.n_steps': 2e6}, 'allow_large_n_steps'), # above 1e6 (about 1.1 GB per 1e6 steps per worker)
    ({'model.n_atmodeller': 0, 'observables.VMR_H2O': {'use': True, 'value': 1e-3, 'sigma': 1.0,
                                                       'space': 'log10'}}, 'needs Atmodeller'),
    ({'model.n_atmodeller': 0, 'observables.mmw': {'use': True, 'value': 3.0, 'sigma': 1.0}}, 'needs Atmodeller'),
    ({'observables.Rp.sigma': None}, 'needs sigma'),
    ({'observables.Rp.value': None}, 'needs value'),
    ({'observables.Rp.sigma': 0.0}, 'must be > 0'),
    ({'observables.Rp.sigma': [0.1, -0.1]}, 'must be > 0'),
    ({'observables.Rp.value': math.nan}, 'NaN'),
    ({'observables.Mdot': {'use': True, 'value': 0.0, 'sigma': 0.3, 'space': 'log10'}}, 'needs value > 0'),
    ({'params.Mp': {'vary': True, 'bounds': [0.5, 6.0]}}, r'outside the valid range \[1, 21\]'), # the melt grid
    ({'params.Mp': {'vary': True, 'bounds': [4.5, 22.0]}}, 'outside the valid range'),
    ({'params.Mp': {}}, 'Mp is required'),
    ({'params.log_f_atm.bounds': [-5.0, 0.0]}, 'outside the valid range'), # log_f_atm < 0
    ({'params.log_f_atm.value': -1.0}, 'outside its bounds'),
    ({'params.log_f_atm.bounds': None}, 'needs bounds'),
    ({'params.t_sat': {'vary': True, 'prior': 'loguniform', 'bounds': [0.0, 1e9]}}, 'bounds > 0'),
    ({'params.eps': {'vary': True, 'prior': 'normal', 'mu': 0.15}}, 'needs mu and sigma'),
    ({'params.eps': {'vary': True, 'bounds': [0.0, 0.5]}}, 'outside the valid range'), # eps in (0, 1]
    ({'params.age': {'vary': True, 'bounds': [1e5, 1e10]}}, 'outside the valid range'), # the age must exceed t0
    ({'params.albedo': 1.0}, 'outside the valid range'), # albedo in [0, 1)
    ({'params.log_f_atm.prior': 'flat'}, 'must be one of'),
    ({'model.composition': 'legacy', 'params.log_metallicity': 0.5}, 'needs model.composition = "ratios"'),
    ({'model.composition': 'legacy', 'params.log_CtoH_enhancement': {'vary': True, 'bounds': [-1, 1]}},
     'needs model.composition'),
    ({'model.n_atmodeller': 200000}, 'must be <= n_steps'),
    ({'model.n_steps': 1.5e4 + 0.5}, 'whole number'),
    ({'model.n_steps': math.inf}, 'whole number, got inf'), # not an OverflowError traceback
    ({'sampler.tau_checks': math.inf}, 'whole number'),
    ({'sampler.tau_checks': -math.inf}, 'whole number'),
    ({'sampler.tau_checks': 0}, 'must be >= 1'),
    ({'solve.refit_steps': [math.inf]}, 'whole number'),
    ({'solve.refit_steps': [25, 0]}, 'must be >= 1'),
    ({'solve.refit_steps': 25}, r'solve.refit_steps must be a list \(or \[\] for none\), got 25'),
    ({'resolution.n_steps': []}, 'must be a non-empty list'),
    ({'solve.derivative': 'backward'}, 'must be "forward" or "central"'),
    ({'model.mechanism': 'XUV+RR'}, 'must be one of'),
    ({'model.melt_fraction_override': 0.0}, 'treats 0.0 as'), # isocalc reads 0.0 as false: the melt grid
    ({'sampler.nwalkers': 7}, 'must be even'),
    ({'sampler.moves': 'walk'}, 'must be one of'),
    ({'star.luminosity': None}, 'needs luminosity'),
    ({'planet.period': None}, 'needs period'),
    ({'resolution.source': 'grid'}, 'resolution.source'),
    ({'sampler.init': 'bisect', 'sampler.bisect_param': 'age'}, "bisect_param = 'age' must be a free parameter"),
    ({'sampler.init': 'bisect', 'sampler.bisect_observable': 'fatm'}, 'must be an active observable'),
    ({'sampler.init': 'bisect', 'observables.Mp': {'use': True, 'value': 5.6, 'sigma': 0.2},
      'sampler.bisect_observable': 'Mp'}, 'computed by the model'), # Mp is an input, not a model output
    ({'sampler.bisect_observable': 'outflow_H_frac'}, "did you mean 'outflow_H_fraction'"),
    ({'sampler.bisect_param': 'log_fatm'}, "did you mean 'log_f_atm'"),
    ({'sampler.bisect_draws': 0}, 'must be >= 1'),
    ({'sampler.bisect_tol': 0.0}, 'must be > 0'),
    ({'resolution.shift_tol': -1e-3}, 'must be > 0'),
    ({'params.Rp_fixed': -1.0}, 'outside the valid range'),
])
def test_validation_errors(overrides, match):
    '''
    Checks that each invalid config is refused with a ConfigError whose message says what is wrong (match). An
    override value of None deletes that key from the minimal config instead.
    '''
    raw = minimal()
    for key in [k for k, v in overrides.items() if v is None]: # None deletes the key
        *path, last = key.split('.')
        node = raw
        for part in path:
            node = node[part]
        node.pop(last)
    with pytest.raises(ConfigError, match = match):
        load({k: v for k, v in overrides.items() if v is not None}, raw)


def test_allowed_edge_cases():
    '''
    Checks edge cases that must load: n_steps above 1e6 with allow_large_n_steps, refit_steps sorted without
    duplicates (or empty), refit_steps without warm starts (a note, not an error), any Mp without the melt grid
    (Atmodeller off, or a melt-fraction override), asymmetric sigma as a tuple, the warnings of T_surf, the
    Johnstone flux model and the legacy O/H enhancement, and an unused observable without value and sigma.
    '''
    cfg = load({'model.n_steps': 2e6, 'model.allow_large_n_steps': True})
    assert cfg['model']['n_steps'] == 2000000
    assert load({'model.n_steps': 10**30, 'model.allow_large_n_steps': True})['model']['n_steps'] == 10**30
    cfg = load({'solve.refit_steps': [400, 25, 100.0, 25]})
    assert cfg['solve']['refit_steps'] == [25, 100, 400]
    assert load({'solve.refit_steps': []})['solve']['refit_steps'] == []
    # refit_steps without warm starts: a note, not an error
    solving = {'params.Mp': {'vary': True, 'bounds': [5.0, 6.0]}, 'solve.param': 'log_f_atm'}
    cfg = load(dict(solving, **{'solve.warm_start': False, 'solve.refit_steps': [10, 20]}))
    assert any('solve.refit_steps is unused' in note for note in cfg['notes']) and not cfg['warnings']
    assert not any('refit_steps' in note for note in load(dict(solving, **{'solve.warm_start': False}))['notes'])
    cfg = load({'model.n_atmodeller': 0, 'params.Mp': 0.5}) # no melt grid without Atmodeller
    assert cfg['warnings'] == ['uncoupled: Atmodeller off (n_atmodeller = 0)']
    cfg = load({'model.melt_fraction_override': 0.3, 'params.Mp': 30.0}) # nor with a melt fraction override
    assert cfg['params']['Mp']['value'] == 30.0
    cfg = load({'observables.Rp.sigma': [0.02, 0.03]})
    assert cfg['observables']['Rp']['sigma'] == (0.02, 0.03)
    cfg = load({'observables.T_surf': {'use': True, 'value': 1800.0, 'sigma': 100.0},
                'model.flux_model': 'Johnstone'})
    assert any('not directly observable' in w for w in cfg['warnings'])
    assert any('Johnstone' in w for w in cfg['warnings'])
    cfg = load({'model.composition': 'legacy', 'params.log_OtoH_enhancement': 1.0})
    assert any('legacy composition' in w for w in cfg['warnings'])
    cfg = load({'observables.Mdot_He': {'use': False}}) # unused observables may omit value and sigma
    assert list(cfg['observables']) == ['Rp']


def test_generic_example():
    '''
    Checks the TOI-406 b example, a planet with only a radius and a mass and no outflow constraint: no warnings,
    log_f_atm, Mp, age and albedo free, init = "prescan" and the ratios composition. With solve.param = "log_f_atm" it
    samples u_Rp in place of log_f_atm, within 6 sigma of the radius datum (1.55 +- 0.06 R_earth).
    '''
    cfg = config.load_config(os.path.join(config.CONFIG_DIR, 'toi406b.toml'))
    assert cfg['warnings'] == [] and cfg['free'] == ['log_f_atm', 'Mp', 'age', 'albedo']
    assert list(cfg['observables']) == ['Mp', 'Rp'] and not config.solve_on(cfg)
    assert cfg['sampler']['init'] == 'prescan' and cfg['model']['composition'] == 'ratios'
    solved = config.load_config(os.path.join(config.CONFIG_DIR, 'toi406b.toml'), ['solve.param="log_f_atm"'])
    assert solved['sampled'] == ['u_Rp', 'Mp', 'age', 'albedo'] and solved['sampler']['init'] == 'solve'
    assert solved['solve']['bounds'] == pytest.approx([1.55 - 6*0.06, 1.55 + 6*0.06])


def test_solve_config():
    '''
    Checks the solve-mode settings: u_Rp sampled in place of log_f_atm, init = "solve" by default (and in place of
    "bisect", with a note), the label, unit and support of u (range_sigma = 6 sigmas each side of the datum; one
    side only for a limit, with a warning), the sampler's view of the params, the init draws of u from the split
    normal of the data, and the errors of a bad [solve] section.
    '''
    cfg = load({'solve.param': 'log_f_atm', 'params.age': {'vary': True, 'bounds': [1e9, 1e10]}})
    assert cfg['sampled'] == ['u_Rp', 'age'] and cfg['sampler']['init'] == 'solve' # the default in solve mode
    assert cfg['solve']['label'] == '$R_p$ (sampled)' and cfg['solve']['unit'] == 'R_earth'
    view = prm.sampled_cfg(cfg) # u as a uniform parameter on its support, for the params functions
    assert view['free'] == ['u_Rp', 'age'] and view['params']['u_Rp']['bounds'] == pytest.approx([1.58, 1.88])
    assert prm.centre_theta(view)[0] == 1.73 and prm.in_support('u_Rp', 1.6, view)
    assert not prm.in_support('u_Rp', 1.5, view) and cfg['params'] is not view['params']
    # init = "bisect" maps to "solve", with a note
    noted = load({'solve.param': 'log_f_atm', 'sampler.init': 'bisect'})
    assert noted['sampler']['init'] == 'solve' and 'redundant in solve mode' in noted['notes'][0]
    # a log10 observable: u in dex, its label from the registry
    x = load({'solve': {'param': 'log_f_atm', 'observable': 'outflow_H_fraction'},
              'observables.outflow_H_fraction': {'use': True, 'value': 1e-3, 'sigma': [0.3, 0.2], 'space': 'log10'}})
    assert x['solve']['bounds'] == pytest.approx([-3 - 1.8, -3 + 1.2]) and x['solve']['unit'] == 'dex'
    assert x['solve']['label'].startswith('log$_{10}$ H/(H+He)')
    # limits: u on the allowed side only, with a warning
    upper = load({'solve.param': 'log_f_atm', 'observables.Rp.kind': 'upper'})
    assert upper['solve']['bounds'] == pytest.approx([1.73 - 0.15, 1.73]) and any('meant for detections' in w
                                                                                 for w in upper['warnings'])
    # the draws of init = "solve": u from the split normal of the data within the support
    rng = np.random.default_rng(0)
    draws = prm.sample_solve(x, 4000, rng)
    u = draws[:, 0]
    assert np.all((u >= -4.8) & (u <= -1.8)) and abs(np.mean(u < -3) - 0.6) < 0.03 # P(below) = 0.3/(0.3 + 0.2)
    assert np.std(u[u < -3]) == pytest.approx(0.3*np.sqrt(1 - 2/np.pi), rel = 0.1) # half-normal of the lo side
    for overrides, match in ((['solve.param="eps"'], 'must be a free parameter'),
                             (['solve.param="log_f_atm"', 'solve.observable="Mp"'], 'computed by the model'),
                             (['solve.param="log_f_atm"', 'solve.observable="Mdot"'], 'computed by the model'),
                             (['solve.param="log_fatm"'], 'did you mean'),
                             (['solve.param="log_f_atm"', 'solve.tol=2.0'], 'solve.tol'),
                             (['sampler.init="solve"'], 'needs solve mode')):
        with pytest.raises(ConfigError, match = match):
            config.load_config(minimal(), overrides)


def test_bisect_and_resolution_source():
    '''
    Checks the bisection target (by default log_f_atm along which Rp is matched) and the default source of the
    resolution study's test points: "bisect" with init = "bisect", else "prescan"; "solve" needs solve mode. A
    limit as the bisection target warns that the walkers start on the boundary of the allowed region.
    '''
    cfg = load({'sampler.init': 'bisect'}) # the default observable Rp and parameter log_f_atm
    assert config.bisect_target(cfg) == ('log_f_atm', 'Rp') and cfg['warnings'] == []
    assert config.resolution_source(cfg) == 'bisect' and config.resolution_source(load()) == 'prescan'
    with pytest.raises(ConfigError, match = 'needs a study in solve mode'):
        config.resolution_source(cfg, 'solve')
    assert config.resolution_source(load(), 'prior') == 'prior' and config.resolution_source(load(), 'chain:x')
    with pytest.raises(ConfigError, match = 'must be one of'):
        load({'sampler.init': 'bisection'})
    limit = load({'sampler.init': 'bisect', 'observables.Rp.kind': 'upper'})
    assert any('start on the boundary' in w for w in limit['warnings'])


def test_rp_fixed():
    '''
    Checks params.Rp_fixed, a constant radius (isocalc's Rp_override): set, fixed or free, it switches
    model.rad_evol off with a warning, since isocalc applies the override only without radius evolution. Unset
    (the default) it is None and the radius evolves. The README table lists it as unset.
    '''
    cfg = load({'params.Rp_fixed': 1.7})
    assert cfg['params']['Rp_fixed']['value'] == 1.7 and cfg['model']['rad_evol'] is False
    assert any('rad_evol is switched to false' in w for w in cfg['warnings'])
    free = load({'params.Rp_fixed': {'vary': True, 'bounds': [1.5, 2.0]}})
    assert free['free'] == ['log_f_atm', 'Rp_fixed'] and free['model']['rad_evol'] is False
    assert prm.theta_to_params([-3.0, 1.8], free)['Rp_fixed'] == 1.8
    unset = load()
    assert unset['params']['Rp_fixed']['value'] is None and unset['model']['rad_evol'] is True
    assert prm.theta_to_params([-3.0], unset)['Rp_fixed'] is None
    params_table = config.markdown_tables()[0]
    assert '| `Rp_fixed` | R_earth | unset | > 0 |' in params_table


def test_overrides():
    '''
    Checks --set overrides: TOML values (numbers, lists, booleans, quoted strings) with a bare string as the
    fallback (phoenix), spaces around '=', a number as the shorthand of a fixed parameter, and parameters made
    free. The same overrides as a dict, of values or of their strings, give the same config. A malformed override
    is an error.
    '''
    raw = minimal()
    raw['params']['F_final'] = 0.17 # shorthand for a fixed parameter
    overrides = ['model.n_steps=2e4', 'params.log_f_atm.bounds=[-4, -2]', 'params.log_f_atm.value = -3',
                 'model.flux_model=phoenix', 'model.mechanism="XUV+CPML"', 'params.eps=0.2',
                 'params.age.vary=true', 'params.age.bounds=[1e9, 1e10]', 'params.F_final.vary=true',
                 'params.F_final.bounds=[0.0, 1.0]', 'sampler.nwalkers=32', 'parallel.cache=false']
    cfg = config.load_config(raw, overrides)
    assert cfg['model']['n_steps'] == 20000 and isinstance(cfg['model']['n_steps'], int)
    assert cfg['params']['log_f_atm']['bounds'] == [-4.0, -2.0]
    assert cfg['params']['log_f_atm']['value'] == -3.0
    assert cfg['model']['flux_model'] == 'phoenix' # unparseable TOML falls back to a string
    assert cfg['model']['mechanism'] == 'XUV+CPML'
    assert cfg['params']['eps']['value'] == 0.2
    assert cfg['free'] == ['log_f_atm', 'age', 'F_final']
    assert cfg['params']['F_final']['value'] == 0.17
    assert cfg['sampler']['nwalkers'] == 32 and cfg['parallel']['cache'] is False
    assert len(cfg['overrides']) == len(overrides)
    same = config.load_config(minimal() | {'params': raw['params']}, {
        'model.n_steps': '2e4', 'params.log_f_atm.bounds': [-4, -2], 'params.log_f_atm.value': -3.0,
        'model.flux_model': 'phoenix', 'model.mechanism': 'XUV+CPML', 'params.eps': 0.2, 'params.age.vary': True,
        'params.age.bounds': [1e9, 1e10], 'params.F_final.vary': 'true', 'params.F_final.bounds': [0.0, 1.0],
        'sampler.nwalkers': 32, 'parallel.cache': False})
    assert {k: v for k, v in same.items() if k != 'overrides'} == {k: v for k, v in cfg.items() if k != 'overrides'}
    assert config.parse_value('1e5') == 1e5 and config.parse_value('true') is True
    assert config.parse_value('[1, 2]') == [1, 2] and config.parse_value('power law') == 'power law'
    with pytest.raises(ConfigError, match = 'section.key=value'):
        config.load_config(minimal(), ['model.n_steps'])
    with pytest.raises(ConfigError, match = 'not a table'):
        config.load_config(minimal(), ['name.x=1'])


def test_dump_toml_round_trip():
    '''
    Checks that config.dump_toml writes TOML that tomllib reads back to the same dict: quotes in strings,
    booleans, small and large floats, +-inf, nested and empty lists, inline and empty tables, keys that need
    quotes and unicode; None values are left out. The shipped example and template round-trip too.
    '''
    raw = {'name': 'a "quoted" name', 'flag': True, 'off': False, 'n': 3, 'x': 1e-05, 'big': 1e16, 'neg': -0.5,
           'inf': math.inf, 'ninf': -math.inf, 'lst': [1.0, 2.5, -3e-8], 'nested': [[1, 2], [3]],
           'strs': ['XUV', 'power law'], 'empty_list': [],
           'model': {'flux_model': 'power law', 'mantle_iron': {'type': 'static', 'Fe_mass_fraction': 0.1}},
           'params': {'Mp': {'vary': True, 'bounds': [4.5, 6.7]}, 'F_final': 0.17},
           'empty': {}, 'odd keys': {'a.b': 1, 'c d': 'e'}, 'unicode': 'M⊕'}
    text = config.dump_toml(raw)
    assert tomllib.loads(text) == raw
    with_none = copy.deepcopy(raw)
    with_none['model']['skip'] = None # TOML has no null: the key is dropped
    assert tomllib.loads(config.dump_toml(with_none)) == raw
    for name in ('lhs1140b.toml', 'template.toml'):
        with open(os.path.join(config.CONFIG_DIR, name), 'rb') as f:
            example = tomllib.load(f)
        assert tomllib.loads(config.dump_toml(example)) == example


def test_resolved_json_round_trip(tmp_path):
    '''
    Checks that a resolved config survives config.json (save_resolved, load_resolved) exactly: an asymmetric sigma
    comes back as a tuple, floats at full (repr) precision, and the write goes through a temporary file that is
    gone afterwards.
    '''
    cfg = load({'observables.Rp.sigma': [0.02, 0.03], 'observables.Mdot': {
        'use': True, 'value': 2e5, 'sigma': 0.13, 'space': 'log10', 'kind': 'upper'},
        'params.age': {'vary': True, 'prior': 'normal', 'mu': 5e9, 'sigma': 1e9},
        'model.mantle_iron': {'type': 'dynamic', 'Fe_mass_fraction': 0.1}}, None)
    cfg['warnings'].append('a warning')
    path = str(tmp_path/'config.json')
    config.save_resolved(cfg, path)
    back = config.load_resolved(path)
    assert back == cfg
    assert isinstance(back['observables']['Rp']['sigma'], tuple)
    assert back['planet']['Fp'] == cfg['planet']['Fp'] # repr precision
    assert not os.path.exists(path + '.tmp')


###_____params_____###

def prior_cfg():
    '''
    A config with one free parameter of each kind of prior: log_f_atm uniform in [-5, -1], t_sat loguniform in
    [1e8, 5e9] yr, Mp normal 5.6 +- 0.19 M_earth without bounds (its support is then the valid range [1, 21] of
    the melt grid), and eps normal 0.15 +- 0.05 truncated to [0.05, 0.5]
    '''
    return load({'params.log_f_atm': {'vary': True, 'bounds': [-5.0, -1.0]},
                 'params.t_sat': {'vary': True, 'prior': 'loguniform', 'bounds': [1e8, 5e9]},
                 'params.Mp': {'vary': True, 'prior': 'normal', 'mu': 5.6, 'sigma': 0.19},
                 'params.eps': {'vary': True, 'prior': 'normal', 'mu': 0.15, 'sigma': 0.05, 'bounds': [0.05, 0.5]}})


def test_log_prior():
    '''
    Checks the log prior up to a constant (uniform 0, loguniform -ln x, normal -0.5 z**2): -inf outside the support
    and for NaN or inf, closed bounds, and a ValueError for a theta of the wrong length. Also theta_to_params,
    prior_box (mu +- 4 sigma for a normal prior without bounds), centre_theta (the params' values; the geometric
    middle of a loguniform prior) and unit_cube.
    '''
    cfg = prior_cfg()
    assert cfg['free'] == ['log_f_atm', 'Mp', 'eps', 't_sat'] # registry order
    theta = np.array([-3.0, 5.8, 0.2, 1e9])
    expected = -0.5*((5.8 - 5.6)/0.19)**2 - 0.5*((0.2 - 0.15)/0.05)**2 - np.log(1e9)
    assert prm.log_prior(theta, cfg) == pytest.approx(expected, rel = 1e-12)
    for i, bad in [(0, -0.5), (0, -5.1), (1, 22.0), (1, 0.9), (2, 0.6), (2, 0.04), (3, 6e9), (3, 9e7), (0, np.nan),
                   (1, np.inf)]:
        outside = theta.copy()
        outside[i] = bad
        assert prm.log_prior(outside, cfg) == -np.inf
    assert prm.log_prior(np.array([-1.0, 21.0, 0.5, 5e9]), cfg) > -np.inf # closed bounds
    with pytest.raises(ValueError):
        prm.log_prior(theta[:2], cfg)
    params = prm.theta_to_params(theta, cfg)
    assert (params['log_f_atm'], params['Mp'], params['eps'], params['t_sat']) == tuple(theta)
    assert params['age'] == 5e9 and set(params) == set(config.PARAMS)
    lo, hi = prm.prior_box(cfg)
    np.testing.assert_allclose(lo, [-5.0, 5.6 - 4*0.19, 0.05, 1e8])
    np.testing.assert_allclose(hi, [-1.0, 5.6 + 4*0.19, 0.5, 5e9])
    np.testing.assert_allclose(prm.centre_theta(cfg), [-3.0, 5.6, 0.15, np.sqrt(1e8*5e9)])
    assert np.all((prm.unit_cube(prm.centre_theta(cfg), cfg) >= 0) & (prm.unit_cube(prm.centre_theta(cfg), cfg) <= 1))


def test_lhs_marginals():
    '''
    Checks the prior draws: every Latin-hypercube draw lies in the support, each marginal follows its prior (a
    Kolmogorov-Smirnov test), and each falls exactly once into every 1/n stratum of its CDF. Plain random draws
    stay in the support too, and a seed always gives the same draws.
    '''
    cfg = prior_cfg()
    n = 2000
    draws = prm.sample_prior(cfg, n, np.random.default_rng(1))
    assert draws.shape == (n, 4)
    assert all(prm.log_prior(theta, cfg) > -np.inf for theta in draws)
    cdfs = [stats.uniform(-5, 4).cdf, stats.norm(5.6, 0.19).cdf,
            stats.truncnorm((0.05 - 0.15)/0.05, (0.5 - 0.15)/0.05, loc = 0.15, scale = 0.05).cdf,
            stats.loguniform(1e8, 5e9).cdf]
    for i, cdf in enumerate(cdfs):
        assert stats.kstest(draws[:, i], cdf).pvalue > 1e-3
        u = np.sort(cdf(draws[:, i])) # Latin hypercube: exactly one draw per 1/n stratum
        assert np.all(np.abs(u*n - (np.arange(n) + 0.5)) <= 0.5 + 1e-6)
    plain = prm.sample_prior(cfg, 500, np.random.default_rng(2), lhs = False)
    assert plain.shape == (500, 4) and all(prm.log_prior(theta, cfg) > -np.inf for theta in plain)
    np.testing.assert_array_equal(prm.sample_prior(cfg, 10, 3), prm.sample_prior(cfg, 10, 3))


def test_jitter_and_ball_stay_in_support():
    '''
    Checks that jitter keeps points inside the prior support, even points on or next to a bound, at 1e-3 and 0.3
    of the prior box, and that at the default scale (1e-3) a point moves by less than 1% of the box. ball starts
    the walkers around the params' values with a spread of about init_scale (default 1% of the box width).
    '''
    cfg = prior_cfg()
    rng = np.random.default_rng(4)
    edge = np.array([-1.0, 5.6, 0.0500001, 5e9]) # on or next to the bounds
    for scale in (1e-3, 0.3):
        jittered = prm.jitter(np.tile(edge, (200, 1)), cfg, rng, scale = scale)
        assert jittered.shape == (200, 4)
        assert all(prm.log_prior(theta, cfg) > -np.inf for theta in jittered)
        assert np.all(jittered != edge) or scale == 1e-3
    single = prm.jitter(edge, cfg, rng)
    assert single.shape == (4,) and prm.log_prior(single, cfg) > -np.inf
    lo, hi = prm.prior_box(cfg)
    assert np.all(np.abs(single - edge) < 1e-2*(hi - lo))
    start = prm.ball(cfg, 100, rng)
    assert start.shape == (100, 4) and all(prm.log_prior(theta, cfg) > -np.inf for theta in start)
    spread = np.std(start, axis = 0)/(hi - lo)
    assert np.all((spread > 2e-3) & (spread < 2e-2)) # init_scale defaults to 1% of the box width


def test_nwalkers_auto_rule():
    '''
    Checks the auto rule of sampler.nwalkers: max(4 ndim, 16), rounded up to a multiple of 4 x workers with the
    default "de" moves (DESnooker updates the walkers in 4 groups, one pool map each) or of 2 x workers otherwise,
    so that every worker is busy in each map; always the smallest such multiple. A given nwalkers is used as it is,
    but must be even and >= 2 ndim.
    '''
    assert [config.auto_nwalkers(d, 6) for d in (1, 2, 3, 4, 5, 6, 7, 12, 13)] == [24]*6 + [48, 48, 72]
    assert config.auto_nwalkers(3, 6, 'stretch') == 24 and config.auto_nwalkers(3, 6, 'de_only') == 24
    assert config.auto_nwalkers(5, 6, 'stretch') == 24 and config.auto_nwalkers(7, 6, 'stretch') == 36
    # at least 16 walkers: 8 split into DESnooker groups of 2 and mixed poorly
    assert config.auto_nwalkers(1, 1) == 16 and config.auto_nwalkers(5, 1) == 20 and config.auto_nwalkers(2, 4) == 16
    assert config.auto_nwalkers(2, 2) == 16 and config.auto_nwalkers(2, 1, 'stretch') == 16
    for ndim in range(1, 15):
        for workers in range(1, 9):
            for moves, step in (('de', 4*workers), ('stretch', 2*workers)):
                n = config.auto_nwalkers(ndim, workers, moves)
                least = max(4*ndim, config.MIN_AUTO_WALKERS)
                assert n % step == 0 and n % 2 == 0 and n >= least and n - step < least
    cfg = load({'params.age': {'vary': True, 'bounds': [1e9, 1e10]}, 'params.eps': {'vary': True,
                                                                                    'bounds': [0.05, 0.5]}})
    assert config.choose_nwalkers(cfg, 6) == 24 and config.choose_nwalkers(cfg, 2) == 16
    cfg['sampler']['nwalkers'] = 10
    assert config.choose_nwalkers(cfg, 6) == 10
    cfg['sampler']['nwalkers'] = 4
    with pytest.raises(ConfigError, match = '>= 2\\*ndim'):
        config.choose_nwalkers(cfg, 6)
    with pytest.raises(ConfigError, match = 'must be even'):
        load({'sampler.nwalkers': 2, 'params.age': {'vary': True, 'bounds': [1e9, 1e10]}})


def test_pms_check():
    '''
    Checks the pre-main-sequence XUV boost: off by default (t_pms = 0 and pms_factor = 0); on with t_pms > t0 it
    needs pms_factor > 0, since Fxuv takes log10(pms_factor) (0 makes the flux NaN), fixed or over its whole prior;
    a pms_factor set while the boost is off gets a note.
    '''
    cfg = load()
    assert (cfg['params']['t_pms']['value'], cfg['params']['pms_factor']['value']) == (0.0, 0.0)
    assert not any('pms_factor' in note for note in cfg['notes'])
    with pytest.raises(config.ConfigError, match = 'pms_factor can be 0'):
        load({'params.t_pms': 1e8})
    with pytest.raises(config.ConfigError, match = 'pms_factor can be 0'): # 0 inside its prior
        load({'params.t_pms': 1e8, 'params.pms_factor': {'vary': True, 'bounds': [0.0, 200.0]}})
    on = load({'params.t_pms': 1e8, 'params.pms_factor': 100.0}) # the setting of isofate_MCMC.py
    assert on['params']['pms_factor']['value'] == 100.0 and not any('pms_factor' in n for n in on['notes'])
    off = load({'params.pms_factor': 100.0}) # t_pms = 0: the boost is off
    assert any('pms_factor = 100 has no effect' in note for note in off['notes'])
    assert not any('pms_factor' in n for n in load({'params.t_pms': 5e5})['notes']) # t_pms < t0 = 1e6: off
