'''
Priors of the free parameters, prior draws (Latin hypercube) and the theta <-> parameter mapping.

theta holds the free parameters in cfg['free'] order, in the units of config.PARAMS (Mp in M_earth, age in yr).
The prior support of a parameter is its bounds, or for a normal prior without bounds its hard validity range
(config.hard_limits), so a prior never proposes inputs isocalc cannot run.

In solve mode the sampler's theta holds u_<observable> in place of the solved parameter (config.sampled_names).
sampled_cfg(cfg) is a view of cfg in which u is a uniform parameter on its support, so that the functions here
(prior_box, unit_cube, jitter, sample_prior, ball, in_support) work on the sampler's theta too. The density of
the sampler in solve mode is not log_prior of that view: it comes from solve.evaluate (prior of the solved
parameter at its root times the Jacobian).
'''

import numpy as np
from scipy import special, stats
from scipy.stats import qmc

from isofate.mcmc import config


def theta_to_params(theta, cfg):
    '''
    All registry parameters at their internal values: free ones from theta, fixed ones from their value (None
    for an unset optional parameter, e.g. Rp_fixed).
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    if theta.shape != (len(cfg['free']),):
        raise ValueError(f"theta has shape {theta.shape}, expected ({len(cfg['free'])},) for {cfg['free']}")
    params = {name: None if p['value'] is None else float(p['value']) for name, p in cfg['params'].items()}
    params.update(zip(cfg['free'], theta.tolist()))
    return params


def centre_theta(cfg):
    '''
    The free parameters' values: the centre of init = "ball" and the point of `check`.
    '''
    return np.array([cfg['params'][name]['value'] for name in cfg['free']], dtype = float)


def support(name, cfg):
    '''
    (lo, hi) of the prior support of a free parameter.
    '''
    bounds = cfg['params'][name]['bounds']
    if bounds is not None:
        return tuple(bounds)
    return config.hard_limits(name, cfg['model'])[:2]


def in_support(name, x, cfg):
    bounds = cfg['params'][name]['bounds']
    if bounds is not None and not bounds[0] <= x <= bounds[1]:
        return False
    return config.within_limits(name, x, cfg['model'])


def _support_mask(theta, cfg):
    '''
    Element-wise in_support for an (n, ndim) array.
    '''
    mask = np.empty(theta.shape, dtype = bool)
    for i, name in enumerate(cfg['free']):
        mask[:, i] = [in_support(name, x, cfg) for x in theta[:, i]]
    return mask


def log_prior(theta, cfg):
    '''
    Log prior up to a constant: uniform 0, loguniform -log x, normal -0.5((x - mu)/sigma)^2; -inf outside the
    support.
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    if theta.shape != (len(cfg['free']),):
        raise ValueError(f"theta has shape {theta.shape}, expected ({len(cfg['free'])},)")
    lp = 0.0
    for name, x in zip(cfg['free'], theta.tolist()):
        p = cfg['params'][name]
        if not in_support(name, x, cfg):
            return -np.inf
        if p['prior'] == 'loguniform':
            lp -= np.log(x)
        elif p['prior'] == 'normal':
            z = (x - p['mu'])/p['sigma']
            lp -= 0.5*z*z
        elif p['prior'] != 'uniform':
            return -np.inf
    return float(lp)


def prior_box(cfg):
    '''
    (lo, hi) arrays of the free parameters: the bounds, or mu -+ 4 sigma (within the hard limits) for a normal
    prior without bounds.
    '''
    lo, hi = [], []
    for name in cfg['free']:
        p = cfg['params'][name]
        if p['bounds'] is not None:
            a, b = p['bounds']
        else:
            a_lim, b_lim = config.hard_limits(name, cfg['model'])[:2]
            a, b = max(p['mu'] - 4*p['sigma'], a_lim), min(p['mu'] + 4*p['sigma'], b_lim)
        lo.append(a)
        hi.append(b)
    return np.array(lo, dtype = float), np.array(hi, dtype = float)


def _ppf(name, u, cfg):
    '''
    Inverse prior CDF of a free parameter at u in [0, 1].
    '''
    p = cfg['params'][name]
    lo, hi = support(name, cfg)
    if p['prior'] == 'uniform':
        x = lo + u*(hi - lo)
    elif p['prior'] == 'loguniform':
        x = np.exp(np.log(lo) + u*(np.log(hi) - np.log(lo)))
    else:
        a, b = (lo - p['mu'])/p['sigma'], (hi - p['mu'])/p['sigma']
        x = stats.truncnorm.ppf(u, a, b, loc = p['mu'], scale = p['sigma'])
    return np.clip(x, lo, hi) # round-off at the edges


def sample_prior(cfg, n, rng, lhs = True):
    '''
    n draws (n, ndim) distributed as the prior: a Latin hypercube (or plain uniform draws) on [0, 1]^ndim mapped
    through each marginal's inverse CDF. rng: numpy Generator or seed.
    '''
    rng = np.random.default_rng(rng)
    ndim = len(cfg['free'])
    if ndim == 0:
        return np.empty((n, 0))
    u = qmc.LatinHypercube(ndim, seed = rng).random(n) if lhs else rng.random((n, ndim))
    return np.column_stack([_ppf(name, u[:, i], cfg) for i, name in enumerate(cfg['free'])])


def unit_cube(theta, cfg):
    '''
    theta mapped to [0, 1] by the prior box (for distances and jitter).
    '''
    lo, hi = prior_box(cfg)
    return (np.asarray(theta, dtype = float) - lo)/(hi - lo)


def _perturb(centre, sigma, cfg, rng, tries = 100):
    '''
    centre + sigma*N(0, 1), redrawing each component that leaves the prior support (kept at centre after tries).
    '''
    out = centre + sigma*rng.standard_normal(centre.shape)
    bad = ~_support_mask(out, cfg)
    for _ in range(tries):
        if not bad.any():
            break
        redraw = centre + sigma*rng.standard_normal(centre.shape)
        out[bad] = redraw[bad]
        bad = ~_support_mask(out, cfg)
    out[bad] = centre[bad]
    return out


def jitter(theta, cfg, rng, scale = 1e-3):
    '''
    theta (ndim,) or (n, ndim) plus scale x box width of Gaussian noise, inside the prior support.
    '''
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta, dtype = float)
    lo, hi = prior_box(cfg)
    out = _perturb(np.atleast_2d(theta).copy(), scale*(hi - lo), cfg, rng)
    return out[0] if theta.ndim == 1 else out


def ball(cfg, n, rng):
    '''
    n starting points (n, ndim): value + init_scale*N(0, 1), with init_scale defaulting to 1e-2 of the box width.
    '''
    rng = np.random.default_rng(rng)
    lo, hi = prior_box(cfg)
    scale = np.array([cfg['params'][name]['init_scale'] or 1e-2*(b - a)
                      for name, a, b in zip(cfg['free'], lo, hi)])
    return _perturb(np.tile(centre_theta(cfg), (n, 1)), scale, cfg, rng)


###_____Solve mode: the sampler's coordinates_____###

def sampled_cfg(cfg):
    '''
    cfg itself without solve mode. In solve mode a shallow copy whose 'free' are the sampler's coordinates
    (u_<observable> in place of the solved parameter) and whose 'params' include u: uniform on its support
    (solve.bounds), value = the datum in the likelihood space (the centre of `check` and of init = "ball").
    '''
    if not config.solve_on(cfg):
        return cfg
    sv = cfg['solve']
    from isofate.mcmc import likelihood
    u = {'vary': True, 'value': float(likelihood.observed(cfg['observables'][sv['observable']])),
         'prior': 'uniform', 'bounds': list(sv['bounds']), 'mu': None, 'sigma': None, 'init_scale': None,
         'unit': sv['unit'], 'label': sv['label']}
    if not sv['bounds'][0] <= u['value'] <= sv['bounds'][1]: # a limit: its value is an end of the support
        u['value'] = 0.5*(sv['bounds'][0] + sv['bounds'][1])
    return dict(cfg, free = config.sampled_names(cfg), params = dict(cfg['params'], **{sv['name']: u}))


def u_ppf(q, cfg):
    '''
    Inverse CDF of the data distribution of the solve observable in its likelihood space at q in [0, 1]: a split
    normal (sigma_lo below the datum, sigma_hi above), truncated to the u support (solve.range_sigma sigmas). For an
    upper (lower) limit only the half below (above) the limit value.
    '''
    from isofate.mcmc import likelihood
    sv = cfg['solve']
    o = cfg['observables'][sv['observable']]
    v, k = likelihood.observed(o), sv['range_sigma']
    s_lo, s_hi = likelihood.sigma_eff(o, above = False), likelihood.sigma_eff(o, above = True)
    p_lo = {'upper': 1.0, 'lower': 0.0}.get(o['kind'], s_lo/(s_lo + s_hi)) # probability below the datum
    q = np.clip(np.asarray(q, dtype = float), 0.0, 1.0)
    edge = special.ndtr(-k) # normal CDF at -k
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        below = special.ndtri(edge + np.clip(q/p_lo, 0, 1)*(0.5 - edge)) if p_lo > 0 else np.zeros_like(q)
        above = special.ndtri(0.5 + np.clip((q - p_lo)/(1 - p_lo), 0, 1)*(0.5 - edge)) if p_lo < 1 else np.zeros_like(q)
    x = np.where(q < p_lo, v + s_lo*below, v + s_hi*above)
    return np.clip(x, *sv['bounds'])


def sample_solve(cfg, n, rng):
    '''
    n starting draws (n, ndim) of the sampler's coordinates in solve mode (init = "solve"): a Latin hypercube with
    the other free parameters from their priors and u from the data distribution of the observable (u_ppf).
    '''
    rng = np.random.default_rng(rng)
    names = config.sampled_names(cfg)
    j = cfg['solve']['index']
    q = qmc.LatinHypercube(len(names), seed = rng).random(n)
    cols = [u_ppf(q[:, i], cfg) if i == j else _ppf(name, q[:, i], cfg) for i, name in enumerate(names)]
    return np.column_stack(cols)
