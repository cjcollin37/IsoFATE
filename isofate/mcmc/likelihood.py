'''
Log-likelihood terms of the observables.

For each active observable the model value m is compared with the data value v in the observable's likelihood
space, with the effective sigma s on the side of the data the model is on:
    detection     -0.5 ((m - v)/s)^2
    upper limit   log Phi((v - m)/s)
    lower limit   log Phi((m - v)/s)
Phi is the normal CDF. Its log comes from scipy.special.log_ndtr, which stays accurate far into the tails, where
log(ndtr(z)) would already be log(0) = -inf below about z = -38. An asymmetric sigma [lo, hi] gives a split
normal: the hi side when the model is above the data, the lo side below. sigma_model, the model's own error, is
added in quadrature. The total log-likelihood is the sum of the terms.

The space is linear, or log10 for observables that span many decades (escape rates, mixing ratios, the outflow
H fraction). There the errors are in dex, and a Gaussian in log10 is a lognormal in the value. In log10 space
the model is clipped to [floor, ceiling] first, so a zero escape rate becomes log10(floor) instead of -inf: far
below any detection, but consistent with an upper limit above the floor. The ceiling is 1/floor for
dimensionless observables (an infinite ratio x/0 becomes log10(1/floor)) and inf for the escape rates [kg/s],
whose values are far above 1/floor.

Model values can be 0, inf or NaN. NaN means undefined, e.g. the composition of a stripped atmosphere: it rules
out a detection or a lower limit and satisfies an upper limit (the species is absent). A value that is infinite
in the likelihood space (in linear space, or in log10 space past a floor of 0 or a ceiling of inf) satisfies only
the limit on its side: -inf an upper limit, +inf a lower limit. No term or total is ever NaN, so emcee never sees
one.
'''

import math

from scipy.special import log_ndtr

from isofate.orbit_params import EqTemp

inf = math.inf


###_____The likelihood space_____###

def observed(ocfg):
    '''
    Returns the data value of an observable in its likelihood space: log10(value) in log10 space, else value.
    '''
    return math.log10(ocfg['value']) if ocfg['space'] == 'log10' else ocfg['value']


def to_space(x, ocfg):
    '''
    Maps a model value into the likelihood space of an observable
    Linear space: x itself. log10 space: log10 of x clipped to [floor, ceiling]. A floor of 0 means no lower clip
    (x <= 0 gives -inf), a ceiling of inf no upper one (+inf stays +inf).

    Inputs:
        - x: model value in physical units; None counts as NaN
        - ocfg: resolved observable config, cfg['observables'][name] [dict]

    Output: the value in the likelihood space; NaN stays NaN
    '''
    x = math.nan if x is None else float(x)
    if ocfg['space'] == 'linear' or math.isnan(x):
        return x
    floor = ocfg['floor']
    ceiling = ocfg.get('ceiling') # absent in configs saved before it existed: 1/floor
    if ceiling is None:
        ceiling = 1/floor if floor > 0 else inf
    x = min(max(x, floor), ceiling)
    if x <= 0: # only possible with floor 0
        return -inf
    return math.log10(x)


def sigma_eff(ocfg, above = None):
    '''
    Returns the effective sigma of an observable in its likelihood space
    An asymmetric sigma [lo, hi] is a split normal: the hi side applies when the model is above the data, the lo
    side otherwise. sigma_model is added in quadrature.

    Inputs:
        - ocfg: resolved observable config [dict]
        - above: True if the model is above the data, False if below, None for the mean of the two sides [Bool]

    Output: effective sigma, in the unit of the space (dex in log10 space)
    '''
    s = ocfg['sigma']
    if isinstance(s, (tuple, list)): # [lo, hi]
        s = 0.5*(s[0] + s[1]) if above is None else (s[1] if above else s[0])
    return math.hypot(s, ocfg['sigma_model'])


###_____Terms_____###

def term(model_value, ocfg):
    '''
    Calculates the log-likelihood term of one observable
    Detection -0.5 z^2, upper limit log Phi(-z), lower limit log Phi(z), with z = (m - v)/s in the likelihood
    space (see the module docstring).

    Inputs:
        - model_value: model value in physical units; None or NaN means undefined
        - ocfg: resolved observable config [dict]

    Output: log-likelihood term, 0 or below; -inf when ruled out, never NaN
    '''
    kind = ocfg['kind']
    m = to_space(model_value, ocfg)
    if math.isnan(m): # undefined, e.g. the composition of a stripped atmosphere: the species is absent
        return 0.0 if kind == 'upper' else -inf
    if math.isinf(m): # beyond any finite datum: only the limit on its side holds
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
    '''
    Sums the terms of the given observables over values (a missing value counts as NaN, i.e. undefined).
    A NaN term counts as -inf, and a total that is not finite becomes -inf. Returns (total, {name: term}).
    '''
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
    Calculates the total log-likelihood of the active observables

    Inputs:
        - obs_values: model values by observable name, e.g. those of model.extract [dict]; a missing one counts
          as undefined
        - cfg: resolved config [dict]

    Output: tuple
        - total: sum of the terms; -inf if any term is -inf, never NaN
        - terms: {name: term} of every active observable [dict]
    '''
    return _sum_terms(obs_values, cfg['observables'])


###_____Input observables_____###

def input_values(params, cfg):
    '''
    Returns the observables computed from the inputs alone
    Mp and age are isocalc inputs, and Teq = EqTemp(Fp, A = albedo) assumes full redistribution, so comparing them
    with data acts as a prior on those parameters.

    Inputs:
        - params: parameter dict (params.theta_to_params) [dict]
        - cfg: resolved config [dict]

    Output: dict
        - Mp: planet mass [M_earth]
        - age: age [yr]
        - Teq: equilibrium temperature [K]
    '''
    return {'Mp': params['Mp'], 'age': params['age'], 'Teq': float(EqTemp(cfg['planet']['Fp'], A = params['albedo']))}


def input_log_like(params, cfg):
    '''
    Calculates log_like over the input observables only (Mp, age, Teq)
    A point these rule out is rejected before the model runs, so it costs nothing.

    Inputs:
        - params: parameter dict (params.theta_to_params) [dict]
        - cfg: resolved config [dict]

    Output: (total, {name: term}) as log_like, over the active input observables [tuple]
    '''
    inputs = {name: o for name, o in cfg['observables'].items() if o['input']} # the active input observables
    return _sum_terms(input_values(params, cfg), inputs)
