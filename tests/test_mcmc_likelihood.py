'''
Tests of isofate.mcmc.likelihood: each kind x space, asymmetric sigma, inf/NaN/0 model values, floors, and that
NaN never comes out.
'''

import itertools
import math

import pytest
from scipy import stats

from isofate.orbit_params import EqTemp
from isofate.mcmc import config
from isofate.mcmc import likelihood as lk

inf, nan = math.inf, math.nan


def obs(value = 1.0, sigma = 0.1, space = 'linear', kind = 'detection', floor = 1e-12, sigma_model = 0.0):
    return {'use': True, 'value': value, 'sigma': sigma, 'space': space, 'kind': kind, 'floor': floor,
            'sigma_model': sigma_model, 'unit': '', 'label': '', 'needs_atmodeller': False, 'input': False}


def test_detection_linear_and_log():
    assert lk.term(1.8, obs(1.73, 0.025)) == pytest.approx(-0.5*((1.8 - 1.73)/0.025)**2, rel = 1e-12)
    assert lk.term(1.73, obs(1.73, 0.025)) == 0.0
    o = obs(2e5, 0.13, 'log10')
    assert lk.term(1e5, o) == pytest.approx(-0.5*((5 - math.log10(2e5))/0.13)**2, rel = 1e-12)
    assert lk.observed(o) == math.log10(2e5) and lk.to_space(1e5, o) == 5.0


@pytest.mark.parametrize('space', ['linear', 'log10'])
def test_limits(space):
    value, sigma, model = (1e-3, 2e-4, 1.3e-3) if space == 'linear' else (1e-3, 0.3, 2e-3)
    v, m = (value, model) if space == 'linear' else (math.log10(value), math.log10(model))
    upper, lower = obs(value, sigma, space, 'upper'), obs(value, sigma, space, 'lower')
    assert lk.term(model, upper) == pytest.approx(stats.norm.logcdf((v - m)/sigma), rel = 1e-12)
    assert lk.term(model, lower) == pytest.approx(stats.norm.logcdf((m - v)/sigma), rel = 1e-12)
    assert lk.term(value, upper) == pytest.approx(math.log(0.5)) # at the limit
    far_below = 1e-30 if space == 'log10' else -1.0
    assert lk.term(far_below, upper) == pytest.approx(0.0, abs = 1e-12)
    assert lk.term(far_below, lower) < -100 # finite deep in the tail
    assert math.isfinite(lk.term(far_below, lower))


def test_asymmetric_sigma_and_model_error():
    o = obs(1.0, (0.1, 0.3))
    assert lk.term(1.6, o) == pytest.approx(-0.5*(0.6/0.3)**2) # above: hi side
    assert lk.term(0.4, o) == pytest.approx(-0.5*(0.6/0.1)**2) # below: lo side
    assert lk.sigma_eff(o) == pytest.approx(0.2) and lk.sigma_eff(o, above = True) == pytest.approx(0.3)
    o = obs(1.0, (0.1, 0.3), sigma_model = 0.4)
    assert lk.term(1.6, o) == pytest.approx(-0.5*0.6**2/(0.3**2 + 0.4**2))
    assert lk.term(0.4, o) == pytest.approx(-0.5*0.6**2/(0.1**2 + 0.4**2))
    o = obs(1.0, (0.1, 0.3), kind = 'upper')
    assert lk.term(1.6, o) == pytest.approx(stats.norm.logcdf(-0.6/0.3))
    o = obs(1e-3, (0.5, 0.2), space = 'log10')
    assert lk.term(1e-2, o) == pytest.approx(-0.5*(1/0.2)**2) # one dex above
    assert lk.term(1e-4, o) == pytest.approx(-0.5*(1/0.5)**2)


def test_non_finite_model_values():
    expected = { # (model value, space): (detection, upper, lower)
        (inf, 'linear'): (-inf, -inf, 0.0),
        (-inf, 'linear'): (-inf, 0.0, -inf),
        (nan, 'linear'): (-inf, 0.0, -inf),
        (None, 'linear'): (-inf, 0.0, -inf),
        (nan, 'log10'): (-inf, 0.0, -inf),
    }
    for (m, space), terms in expected.items():
        for kind, t in zip(('detection', 'upper', 'lower'), terms):
            assert lk.term(m, obs(1.0, 0.1, space, kind)) == t, (m, space, kind)


def test_floors_in_log_space():
    o = obs(1e-3, 1.0, 'log10', floor = 1e-12)
    for low in (0.0, -5.0, 1e-20, -inf):
        assert lk.to_space(low, o) == -12.0
        assert lk.term(low, o) == lk.term(1e-12, o) == pytest.approx(-0.5*9**2)
    for high in (inf, 1e30):
        assert lk.to_space(high, o) == 12.0
        assert lk.term(high, o) == lk.term(1e12, o)
    upper = obs(1e-3, 0.5, 'log10', 'upper', floor = 1e-12)
    assert lk.term(0.0, upper) == pytest.approx(stats.norm.logcdf(9/0.5)) # zero rate satisfies an upper limit
    assert lk.term(inf, upper) == pytest.approx(stats.norm.logcdf(-15/0.5))
    none = obs(1.7, 0.01, 'log10', floor = 0.0) # no floor: log10(0) = -inf
    assert lk.to_space(0.0, none) == -inf and lk.term(0.0, none) == -inf
    assert lk.term(0.0, obs(1.7, 0.01, 'log10', 'upper', floor = 0.0)) == 0.0
    assert lk.term(inf, obs(1.7, 0.01, 'log10', 'lower', floor = 0.0)) == 0.0


def test_nan_never_returned():
    values = [nan, inf, -inf, 0.0, -1.0, 1e-300, 1e-12, 1e-3, 1.0, 1e5, 1e300, None]
    for m, space, kind, floor, sigma in itertools.product(values, ('linear', 'log10'), ('detection', 'upper', 'lower'),
                                                          (0.0, 1e-12), (1e-200, 0.1, (0.05, 0.2), 1e200)):
        t = lk.term(m, obs(1e-3, sigma, space, kind, floor))
        assert not math.isnan(t) and t <= 0, (m, space, kind, floor, sigma, t)


def cfg_with(observables):
    raw = {'star': {'mass': 0.18, 'luminosity': 0.0038}, 'planet': {'period': 24.74},
           'params': {'Mp': {'value': 5.6}, 'albedo': 0.3, 'log_f_atm': {'vary': True, 'bounds': [-5.0, -1.0]}},
           'observables': observables}
    return config.load_config(raw)


def test_log_like_and_inputs():
    cfg = cfg_with({'Rp': {'use': True, 'value': 1.73, 'sigma': 0.025},
                    'Mp': {'use': True, 'value': 5.5, 'sigma': 0.2},
                    'Teq': {'use': True, 'value': 200.0, 'sigma': 10.0},
                    'C_O': {'use': True, 'value': 0.5, 'sigma': 0.1},
                    'D_H': {'use': True, 'value': 1e-4, 'sigma': 0.5, 'space': 'log10', 'kind': 'upper'},
                    'Mdot': {'use': False}})
    assert list(cfg['observables']) == ['Mp', 'Teq', 'Rp', 'C_O', 'D_H']
    params = {name: p['value'] for name, p in cfg['params'].items()}
    inputs = lk.input_values(params, cfg)
    assert inputs == {'Mp': 5.6, 'age': 5e9, 'Teq': EqTemp(cfg['planet']['Fp'], A = 0.3)}
    total, terms = lk.input_log_like(params, cfg)
    assert list(terms) == ['Mp', 'Teq']
    assert total == pytest.approx(-0.5*(0.1/0.2)**2 - 0.5*((inputs['Teq'] - 200)/10)**2)

    values = dict(inputs, Rp = 1.75, C_O = 0.6, D_H = 3e-5)
    total, terms = lk.log_like(values, cfg)
    assert total == pytest.approx(sum(terms.values())) and total < 0
    assert terms['Rp'] == pytest.approx(-0.5*(0.02/0.025)**2)
    # a stripped atmosphere: composition undefined, so the C/O detection rules it out; the D/H limit is fine
    total, terms = lk.log_like(dict(values, C_O = nan, D_H = nan), cfg)
    assert total == -inf and terms['C_O'] == -inf and terms['D_H'] == 0.0
    total, terms = lk.log_like({}, cfg) # missing values count as undefined
    assert total == -inf and not any(math.isnan(t) for t in terms.values())
    assert lk.log_like({}, cfg_with({})) == (0.0, {})
