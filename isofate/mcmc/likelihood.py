'''
Log-likelihood terms of the observables.

Each active observable is compared in its space: linear, or log10 with the model clipped to [floor, ceiling]
(so a zero escape rate becomes log10(floor) instead of -inf). The ceiling is 1/floor for dimensionless
observables (an infinite ratio x/0 becomes log10(1/floor)) and inf for the escape rates [kg/s], whose values
are far above 1/floor. Detections are Gaussian (split normal for asymmetric sigma); upper and lower limits use
log Phi, the log of the normal CDF. Model values can be 0, inf or NaN. NaN means undefined, e.g. the
composition of a stripped atmosphere: it rules out a detection or a lower limit and satisfies an upper limit
(the species is absent).
'''

import math

from scipy.special import log_ndtr

from isofate.orbit_params import EqTemp

inf = math.inf


def observed(ocfg):
    '''
    The data value in the likelihood space.
    '''
    return math.log10(ocfg['value']) if ocfg['space'] == 'log10' else ocfg['value']


def to_space(x, ocfg):
    '''
    A model value in the likelihood space: x, or log10 of x clipped to [floor, ceiling] (floor 0: no lower clip,
    and x <= 0 gives -inf; ceiling inf: +inf stays +inf). NaN stays NaN.
    '''
    x = math.nan if x is None else float(x)
    if ocfg['space'] == 'linear' or math.isnan(x):
        return x
    floor = ocfg['floor']
    ceiling = ocfg.get('ceiling') # absent in configs saved before it existed: 1/floor
    if ceiling is None:
        ceiling = 1/floor if floor > 0 else inf
    x = min(max(x, floor), ceiling)
    if x <= 0:
        return -inf
    return math.log10(x)


def sigma_eff(ocfg, above = None):
    '''
    Effective sigma in the likelihood space: the hi side of an asymmetric sigma when the model is above the data
    (above = True), the lo side otherwise, their mean for above = None; with sigma_model added in quadrature.
    '''
    s = ocfg['sigma']
    if isinstance(s, (tuple, list)):
        s = 0.5*(s[0] + s[1]) if above is None else (s[1] if above else s[0])
    return math.hypot(s, ocfg['sigma_model'])


def term(model_value, ocfg):
    '''
    Log-likelihood term of one observable for a model value in physical units. Never NaN.
    '''
    kind = ocfg['kind']
    m = to_space(model_value, ocfg)
    if math.isnan(m):
        return 0.0 if kind == 'upper' else -inf
    if math.isinf(m):
        if m > 0:
            return 0.0 if kind == 'lower' else -inf
        return 0.0 if kind == 'upper' else -inf
    v = observed(ocfg)
    s = sigma_eff(ocfg, above = m > v)
    if kind == 'detection':
        z = (m - v)/s
        return -0.5*z*z # not z**2, which raises OverflowError for |z| > 1e154
    if kind == 'upper':
        return float(log_ndtr((v - m)/s))
    return float(log_ndtr((m - v)/s))


def _sum_terms(values, observables):
    terms = {}
    total = 0.0
    for name, ocfg in observables.items():
        t = term(values.get(name, math.nan), ocfg)
        terms[name] = -inf if math.isnan(t) else t
        total += terms[name]
    if not math.isfinite(total):
        total = -inf
    return total, terms


def log_like(obs_values, cfg):
    '''
    Total log-likelihood of the active observables and the term of each: (total, {name: term}).
    -inf if any term is -inf; never NaN.
    '''
    return _sum_terms(obs_values, cfg['observables'])


def input_values(params, cfg):
    '''
    The observables computed from the inputs alone: Mp [M_earth], age [yr], Teq [K].
    '''
    return {'Mp': params['Mp'], 'age': params['age'], 'Teq': float(EqTemp(cfg['planet']['Fp'], A = params['albedo']))}


def input_log_like(params, cfg):
    '''
    log_like over the input observables only (Mp, age, Teq), to reject a point before running the model.
    '''
    inputs = {name: o for name, o in cfg['observables'].items() if o['input']}
    return _sum_terms(input_values(params, cfg), inputs)
