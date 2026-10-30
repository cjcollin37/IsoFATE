'''
Priors of the free parameters, prior draws for the starting points (Latin hypercube), and the mapping between
theta and the parameters.

theta holds the free parameters in cfg['free'] order, in the units of config.PARAMS (Mp in M_earth, age in yr).
The prior support of a parameter is its bounds, or for a normal prior without bounds its hard validity range
(config.hard_limits), so a prior never proposes inputs isocalc cannot run (e.g. an age before t0, or a mass
outside the Atmodeller melt grid).

In solve mode the sampler's theta holds u_<observable> in place of the solved parameter (config.sampled_names).
sampled_cfg(cfg) is a view of cfg in which u is a uniform parameter on its support, so that the functions here
(prior_box, unit_cube, jitter, sample_prior, ball, in_support) work on the sampler's theta too. The density of
the sampler in solve mode is not log_prior of that view: it comes from solve.evaluate (the prior of the solved
parameter at its root times the Jacobian |dP*/du|).
'''

import numpy as np
from scipy import special, stats
from scipy.stats import qmc

from isofate.mcmc import config


###_____theta and the parameters_____###

def theta_to_params(theta, cfg):
    '''
    Maps theta to every registry parameter
    Free parameters come from theta, fixed ones from their value (None for an unset optional parameter, e.g.
    Rp_fixed), all in the units of config.PARAMS.

    Inputs:
        - theta: free parameters in cfg['free'] order (ndim,) [array]
        - cfg: resolved config [dict]

    Output: {name: value} of every parameter in cfg['params'] [dict]; raises ValueError when theta has the wrong
        shape
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    if theta.shape != (len(cfg['free']),):
        raise ValueError(f"theta has shape {theta.shape}, expected ({len(cfg['free'])},) for {cfg['free']}")
    params = {name: None if p['value'] is None else float(p['value']) for name, p in cfg['params'].items()}
    params.update(zip(cfg['free'], theta.tolist())) # the free parameters replace their values
    return params


def centre_theta(cfg):
    '''
    Returns the free parameters' values (ndim,): the centre of init = "ball", the point of `check` and of the
    workers' warm-up, and the fixed values of a scan.
    '''
    return np.array([cfg['params'][name]['value'] for name in cfg['free']], dtype = float)


###_____Prior support and density_____###

def support(name, cfg):
    '''
    Returns the prior support (lo, hi) of a free parameter: its bounds, or for a normal prior without bounds its
    hard validity range (config.hard_limits).
    '''
    bounds = cfg['params'][name]['bounds']
    if bounds is not None:
        return tuple(bounds)
    return config.hard_limits(name, cfg['model'])[:2]


def in_support(name, x, cfg):
    '''
    Returns True when x lies in the prior support of a free parameter: inside its bounds (if any) and inside its
    hard validity range, open ends excluded (e.g. log_f_atm < 0). False for NaN and inf.
    '''
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
    Calculates the log prior density of theta, up to a constant
    uniform: 0; loguniform: -log x (a density 1/x, flat in log x); normal: -0.5 ((x - mu)/sigma)^2, truncated to
    the support. The normalizations are left out: the sampler only uses differences of the log-probability.

    Inputs:
        - theta: free parameters in cfg['free'] order (ndim,) [array]
        - cfg: resolved config, or sampled_cfg(cfg), where it checks the support of a sampler theta [dict]

    Output: log prior; -inf outside the support (or for an unknown prior name); raises ValueError when theta has
        the wrong shape
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
    Returns the prior box of the free parameters
    The bounds, or mu -+ 4 sigma (within the hard limits) for a normal prior without bounds, which holds all but
    6e-5 of its mass. unit_cube, jitter, ball and the root finder of search.py work in this box.

    Inputs:
        - cfg: resolved config, or sampled_cfg(cfg) for the sampler's coordinates [dict]

    Output: tuple
        - lo: lower ends (ndim,) [array]
        - hi: upper ends (ndim,) [array]
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


###_____Draws of starting points_____###

def _ppf(name, u, cfg):
    '''
    Inverse prior CDF of a free parameter at u in [0, 1]: linear for a uniform prior, linear in log x for a
    loguniform one, and a normal truncated to the support (scipy truncnorm) for a normal one.
    '''
    p = cfg['params'][name]
    lo, hi = support(name, cfg)
    if p['prior'] == 'uniform':
        x = lo + u*(hi - lo)
    elif p['prior'] == 'loguniform':
        x = np.exp(np.log(lo) + u*(np.log(hi) - np.log(lo)))
    else:
        a, b = (lo - p['mu'])/p['sigma'], (hi - p['mu'])/p['sigma'] # the support in sigmas from mu
        x = stats.truncnorm.ppf(u, a, b, loc = p['mu'], scale = p['sigma'])
    return np.clip(x, lo, hi) # round-off at the edges


def sample_prior(cfg, n, rng, lhs = True):
    '''
    Draws n points distributed as the prior
    Latin hypercube sampling from McKay et al 1979 (scipy.stats.qmc)

    A Latin hypercube cuts each axis of [0, 1]^ndim into n equal slices and puts exactly one draw in each slice of
    every axis, so n draws cover every marginal evenly, with fewer clumps and gaps than plain uniform draws. Each
    column then goes through its marginal's inverse CDF, so the draws follow the prior.

    Inputs:
        - cfg: resolved config, or sampled_cfg(cfg) for the sampler's coordinates [dict]
        - n: number of draws [int]
        - rng: numpy Generator or seed
        - lhs: True for a Latin hypercube, False for plain uniform draws on [0, 1]^ndim [Bool]

    Output: draws (n, ndim) [array]; (n, 0) without free parameters
    '''
    rng = np.random.default_rng(rng)
    ndim = len(cfg['free'])
    if ndim == 0:
        return np.empty((n, 0))
    u = qmc.LatinHypercube(ndim, seed = rng).random(n) if lhs else rng.random((n, ndim))
    return np.column_stack([_ppf(name, u[:, i], cfg) for i, name in enumerate(cfg['free'])])


def unit_cube(theta, cfg):
    '''
    Maps theta, (ndim,) or (n, ndim), into [0, 1] by the prior box, so that every coordinate counts alike in
    distances between points (jitter is sized in the same units).
    '''
    lo, hi = prior_box(cfg)
    return (np.asarray(theta, dtype = float) - lo)/(hi - lo)


def _perturb(centre, sigma, cfg, rng, tries = 100):
    '''
    centre (n, ndim) + sigma*N(0, 1), redrawing each component that leaves the prior support (kept at centre
    after tries).
    '''
    out = centre + sigma*rng.standard_normal(centre.shape)
    bad = ~_support_mask(out, cfg)
    for _ in range(tries):
        if not bad.any():
            break
        redraw = centre + sigma*rng.standard_normal(centre.shape)
        out[bad] = redraw[bad] # only the bad components take the new draw
        bad = ~_support_mask(out, cfg)
    out[bad] = centre[bad]
    return out


def jitter(theta, cfg, rng, scale = 1e-3):
    '''
    Moves points by a little Gaussian noise, inside the prior support
    Used to spread copies of a starting point (sampler.safe_jitter), since an ensemble needs distinct walkers.

    Inputs:
        - theta: points (ndim,) or (n, ndim) [array]
        - cfg: resolved config, or sampled_cfg(cfg) for the sampler's coordinates [dict]
        - rng: numpy Generator or seed
        - scale: sigma of the noise, as a fraction of the prior box width [ndim]

    Output: jittered points, the shape of theta [array]; a component that keeps leaving the support stays put
    '''
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta, dtype = float)
    lo, hi = prior_box(cfg)
    out = _perturb(np.atleast_2d(theta).copy(), scale*(hi - lo), cfg, rng)
    return out[0] if theta.ndim == 1 else out


def ball(cfg, n, rng):
    '''
    Draws n starting points (n, ndim) around the params' values (init = "ball")
    value + init_scale*N(0, 1) for each parameter, with init_scale defaulting to 1e-2 of the box width; a
    component outside the prior support is drawn again.
    '''
    rng = np.random.default_rng(rng)
    lo, hi = prior_box(cfg)
    scale = np.array([cfg['params'][name]['init_scale'] or 1e-2*(b - a)
                      for name, a, b in zip(cfg['free'], lo, hi)])
    return _perturb(np.tile(centre_theta(cfg), (n, 1)), scale, cfg, rng)


###_____Solve mode: the sampler's coordinates_____###

def sampled_cfg(cfg):
    '''
    Returns the config in the sampler's coordinates
    cfg itself without solve mode. In solve mode a shallow copy whose 'free' are the sampler's coordinates
    (u_<observable> in place of the solved parameter) and whose 'params' include u: uniform on its support
    (solve.bounds), value = the datum in the likelihood space (the centre of `check` and of init = "ball"). A
    datum outside the support is replaced by the middle of the support; a limit's datum is an end of the support,
    which counts as inside, so it stays.
    '''
    if not config.solve_on(cfg):
        return cfg
    sv = cfg['solve']
    from isofate.mcmc import likelihood
    u = {'vary': True, 'value': float(likelihood.observed(cfg['observables'][sv['observable']])),
         'prior': 'uniform', 'bounds': list(sv['bounds']), 'mu': None, 'sigma': None, 'init_scale': None,
         'unit': sv['unit'], 'label': sv['label']}
    if not sv['bounds'][0] <= u['value'] <= sv['bounds'][1]: # outside (a limit's datum is an end: inside)
        u['value'] = 0.5*(sv['bounds'][0] + sv['bounds'][1])
    return dict(cfg, free = config.sampled_names(cfg), params = dict(cfg['params'], **{sv['name']: u}))


def u_ppf(q, cfg):
    '''
    Inverse CDF of the data distribution of the solve observable, in its likelihood space
    A split normal around the datum v (sigma_lo below, sigma_hi above, each side holding a share of the probability
    proportional to its sigma), truncated to the u support (solve.range_sigma sigmas on each side). For an upper
    (lower) limit only the half below (above) the limit value.

    Inputs:
        - q: quantiles in [0, 1] (clipped to it) [array]
        - cfg: resolved config in solve mode [dict]

    Output: values of u inside the u support, solve.bounds [array]
    '''
    from isofate.mcmc import likelihood
    sv = cfg['solve']
    o = cfg['observables'][sv['observable']]
    v, k = likelihood.observed(o), sv['range_sigma']
    s_lo, s_hi = likelihood.sigma_eff(o, above = False), likelihood.sigma_eff(o, above = True)
    p_lo = {'upper': 1.0, 'lower': 0.0}.get(o['kind'], s_lo/(s_lo + s_hi)) # probability below the datum
    q = np.clip(np.asarray(q, dtype = float), 0.0, 1.0)
    edge = special.ndtr(-k) # normal CDF at -k
    # q in [0, p_lo] maps to the standard normal between -k and 0 (below the datum), q in [p_lo, 1] to between 0
    # and k (above it)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        below = special.ndtri(edge + np.clip(q/p_lo, 0, 1)*(0.5 - edge)) if p_lo > 0 else np.zeros_like(q)
        above = special.ndtri(0.5 + np.clip((q - p_lo)/(1 - p_lo), 0, 1)*(0.5 - edge)) if p_lo < 1 else np.zeros_like(q)
    x = np.where(q < p_lo, v + s_lo*below, v + s_hi*above)
    return np.clip(x, *sv['bounds'])


def sample_solve(cfg, n, rng):
    '''
    Draws n starting points (n, ndim) of the sampler's coordinates in solve mode (init = "solve")
    A Latin hypercube, as in sample_prior: the other free parameters from their priors, u from the data
    distribution of the solve observable (u_ppf). The sampler then evaluates each draw, which solves for the
    solved parameter (solve.evaluate).

    Inputs:
        - cfg: resolved config in solve mode [dict]
        - n: number of draws [int]
        - rng: numpy Generator or seed

    Output: draws (n, ndim) in config.sampled_names order [array]
    '''
    rng = np.random.default_rng(rng)
    names = config.sampled_names(cfg)
    j = cfg['solve']['index'] # the column of u
    q = qmc.LatinHypercube(len(names), seed = rng).random(n)
    cols = [u_ppf(q[:, i], cfg) if i == j else _ppf(name, q[:, i], cfg) for i, name in enumerate(names)]
    return np.column_stack(cols)
