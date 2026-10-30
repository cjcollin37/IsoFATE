'''
Solve mode ([solve] param = P, observable = O; SPEC amendment 2): the sampler moves in u = T(O), the model value
of O in its likelihood space (log10 x for a log10-space observable, the value itself for a linear one), in place of
the free parameter P that pins O down. Example: log_f_atm and the outflow H fraction of LHS 1140 b, whose
posterior is a curved ridge 0.003 dex thin in log_f_atm, on which emcee's moves do not mix.

For a sampled point (u, phi), phi being the other free parameters, the root finder of search.py finds P* inside
P's prior support with T(O_model(P*, phi)) = u (to solve.tol sigma of O), and the model run at P* gives every
observable (the blob values belong to P*). The density the sampler sees is the exact change of variables

    p(u, phi) = prior_P(P*) prior_phi(phi) |dP*/du| L(model(P*, phi)),    dP*/du = 1/(dT/dP at P*)

so the posterior over (P, phi) is unchanged:
- dT/dP: a finite difference with h = 1e-5 of P's prior width. solve.derivative = "forward" (default): one more
  model run, at P* + h (P* - h at the upper prior bound; where T is undefined at P* + h, a second run at P* - h). On
  LHS 1140 b its ln|J| differs from the central one by 0.009 (median; at most 0.021), about the central one's own
  numerical noise. "central": 2 more model runs, and both one-sided slopes must have the same sign. A config saved
  before the option existed (2026-10-29) gets "central", the only scheme until then; a resumed run keeps the scheme
  of its last session (sampler.run_derivative). The slope must have the sign of the bracket (T rising from the side
  below u to the side above it); a derivative that is 0, undefined, or (central) changes sign within +-h gives lp =
  -inf (status 3) with the reason in the error string. log_jac = -ln|dT/dP| is stored in the blobs.
- The support of u is [v - k s_lo, v + k s_hi], v and s the datum and its effective sigma on each side, k =
  solve.range_sigma (6): the truncation removes only points whose O term is below -18 (e^-18 of the likelihood).
  It also keeps the sampler off the plateaus where T cannot be inverted that lie outside it, e.g. x = 1 (He not
  escaping) or the 1e-15 floor of the outflow H fraction. A plateau inside it (Rp = the bare core of a stripped
  planet) has no u of its own, so its posterior mass is left out (plateau() finds one in a search). A limit uses
  [v - k s, v] (upper) or [v, v + k s] (lower), with a warning: solve mode is meant for detections.
- No root in P's prior support for (u, phi): lp = -inf, status 1 ("outside prior: no P in its prior gives this
  u"). One root per bracket is assumed (the density would otherwise be a sum over roots); `scan` reports
  non-monotonic stretches of O(P) inside the u range so that this can be checked.
- The blobs' log_prior is the prior of the physical point (P*, phi); lp = log_prior + log_jac + log_like.

Warm start. After init = "solve" the parent fits a table of P* over the sampler's coordinates (local linear
regression on the k nearest solved points in the unit cube, the coordinates scaled by how strongly P* depends on
each, from a global linear fit) and writes it to <run_dir>/solve_table.npz before the first step. The
initialization's draws cover the whole prior, so after the steps of solve.refit_steps (default 25, 50, 100, 200,
400) the parent refits the table to those draws plus every distinct solved state of the chain so far
(build_table), between two steps, while no evaluation is running. Workers load the file lazily (model._RUN_DIR)
and reload it when it changes (worker_table); they never write it. On the final LHS 1140 b chain (1e5/100) a
catch-up refit after step 250, from 1161 solved points, cut the median error of the guess from 0.0067 to 0.00024
dex, and the model runs per solve (with the forward derivative) from a mean of about 10.5 to about 5.6. A solve
starts from the table's guess and expands a bracket around it (search.Root): the first step is 1.5 Newton steps
with the slope |dT/dP| of the neighbouring solves, within 1/16 and 16 times the 68th percentile of the table's
leave-one-out errors. The table changes only the path of a solve, not its root (to solve.tol). Between refits the
result depends only on theta and that file, and every checkpoint holds the active table (table_extras; or
table_version = -1 while the run has none), which a resume writes back (or, for -1, removes), and a resume redoes
a refit that the end of the session cut short, so a resumed run is bit-identical. Without the table (the
initialization itself, check, resolution) the search starts from both prior bounds.
'''

import math
import os

import numpy as np

from isofate.mcmc import likelihood, model, search
from isofate.mcmc import params as prm

nan, inf = math.nan, math.inf
TABLE = 'solve_table.npz'
DERIVATIVE_STEP = 1e-5 # of P's prior width: the finite-difference step of dT/dP
INPUT_PARAMS = ('Mp', 'age', 'albedo') # parameters the input observables (Mp, age, Teq) depend on
EVALS_GUESS = 15 # model runs per cold solve, for cost estimates before any was measured

MAX_TABLE_POINTS = 5000 # a refitted table keeps at most this many solved points (the most recent)
NO_TABLE = -1 # table_version of a checkpoint written while the run had no table yet (init other than "solve")
OLD_DERIVATIVE = 'central' # the scheme of configs saved before solve.derivative existed (2026-10-29)

_TABLES = {} # path -> (file signature, table), the worker's lazily loaded warm-start tables


def settings(cfg):
    '''
    search.bracket() of solve.param (lo, hi: its prior support; index; log) plus {observable, ocfg, name (u_<O>),
    bounds (the u support), tol, max_evals, warm_start, h (the derivative step), derivative ("forward" or
    "central")}. Configs saved before solve.derivative existed get "central", the scheme they were sampled with; a
    resumed run sets the key from its checkpoint first (sampler.run_derivative).
    '''
    sv = cfg['solve']
    st = search.bracket(cfg, sv['param'])
    st.update(observable = sv['observable'], ocfg = cfg['observables'][sv['observable']], name = sv['name'],
              bounds = tuple(sv['bounds']), tol = float(sv['tol']), max_evals = int(sv['max_evals']),
              warm_start = bool(sv['warm_start']), h = DERIVATIVE_STEP*(st['hi'] - st['lo']),
              derivative = sv.get('derivative', OLD_DERIVATIVE))
    return st


def refit_steps(cfg):
    '''
    The steps after which the sampler refits the warm-start table to the chain (solve.refit_steps; configs saved
    before it existed get the default), sorted; none without warm starts.
    '''
    sv = cfg['solve']
    if not sv.get('warm_start', True):
        return []
    from isofate.mcmc.config import SOLVE_DEFAULTS
    return sorted(int(k) for k in sv.get('refit_steps', SOLVE_DEFAULTS['refit_steps']))


def physical(theta, value, cfg):
    '''
    The physical theta (cfg['free'] order) of a sampler theta with value in place of u. theta: (ndim,) or (n, ndim);
    value: a number or (n,).
    '''
    out = np.array(theta, dtype = float)
    out[..., cfg['solve']['index']] = value
    return out


def to_sampled(theta, blobs, cfg):
    '''
    The sampler theta of physical points theta (n, ndim) with their blobs: u = T(O) from the blob values
    (NaN where O is undefined).
    '''
    st = settings(cfg)
    T = np.atleast_1d(search.g_of(blobs, st['observable'], st['ocfg'], 0.0))
    return physical(theta, T, cfg)


def _base(theta, cfg):
    res = {'status': nan, 'runtime': 0.0, 'lp': -inf, 'log_prior': -inf, 'log_like': nan, 'terms': {},
           'obs': model._nan_obs(), 'params': prm.theta_to_params(physical(theta, nan, cfg), cfg), 'error': None,
           'solve_value': nan, 'solve_evals': 0, 'log_jac': nan}
    return res


def _derivative(run, point, st, root, u):
    '''
    (dT/dP at the root, model runs, why it is unusable or None). point: the physical theta at the root.
    '''
    h, j, n = st['h'], st['index'], 0
    T0 = root.g + u
    T = {}
    forward = st.get('derivative', OLD_DERIVATIVE) == 'forward'
    for side in ((1, -1) if forward else (-1, 1)):
        x = root.x + side*h
        if not st['lo'] <= x <= st['hi']:
            continue
        p = point.copy()
        p[j] = x
        g, ran, _ = run(p)
        n += 1
        if ran and math.isfinite(g):
            T[side] = g + u
            if forward: # one usable side is enough
                break
    slopes = {side: (T[side] - T0)/(side*h) for side in T}
    if not slopes:
        return nan, n, f"{st['observable']} is undefined on both sides of the root (+-{h:.3g})"
    if len(slopes) == 2:
        if np.sign(slopes[-1]) != np.sign(slopes[1]) or slopes[-1] == 0 or slopes[1] == 0:
            return nan, n, (f'dT/d{st["param"]} changes sign or vanishes within +-{h:.3g} of the root (one-sided '
                            f'slopes {slopes[-1]:.4g} and {slopes[1]:.4g})')
        deriv = (T[1] - T[-1])/(2*h)
    else:
        deriv = next(iter(slopes.values()))
    if not math.isfinite(deriv) or deriv == 0:
        return nan, n, f'dT/d{st["param"]} = {deriv:.4g} at the root'
    if root.a is not None and np.sign(deriv) != -root.sign_a:
        return nan, n, (f'dT/d{st["param"]} = {deriv:.4g} at the root has the wrong sign for its bracket (T must rise '
                        'from below u to above it)')
    return deriv, n, None


def evaluate(theta, cfg, n_steps = None, n_atmodeller = None, table = None, history = False):
    '''
    The solve-mode evaluation of a sampler theta (u in place of solve.param; see the module docstring): the
    root, the Jacobian, and model.evaluate's result at the root plus solve_value (P*), solve_evals (model runs,
    the derivative included), log_jac (ln|dP*/du|); runtime is the isocalc time of all runs. table: the warm-start
    table (default: none, a cold start). history = True adds the search's evaluations (search.history_table).
    '''
    st = settings(cfg)
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    u = float(theta[st['index']])
    res = _base(theta, cfg)
    if history:
        res['history'] = []
    if not math.isfinite(prm.log_prior(theta, prm.sampled_cfg(cfg))): # u outside its support, phi outside the prior
        res['status'] = model.OUTSIDE_PRIOR
        return res
    if st['param'] not in INPUT_PARAMS: # inputs that do not depend on P rule the point out before any model run
        ll, terms = likelihood.input_log_like(res['params'], cfg)
        if ll == -inf:
            res.update(status = model.NO_LIKELIHOOD, log_like = -inf, terms = terms)
            return res
    runtime = [0.0]

    def run(point):
        r = model.evaluate(point, cfg, n_steps, n_atmodeller)
        runtime[0] += r['runtime']
        g = float(search.g_of(r['obs'], st['observable'], st['ocfg'], u)[0])
        return g, r['status'] in search.RAN, r

    guess, step, orient, slope = predict(table, theta) if table is not None else (None, None, 0.0, None)
    start = physical(theta, nan, cfg)
    root, hist = search.find_root(run, start, st, u, st['tol'], st['max_evals'], guess, step, orient, slope)
    n_evals = root.evals
    if history:
        res['history'] = hist
    P, name = st['param'], st['name']
    if root.outcome != search.CONVERGED:
        res.update(runtime = runtime[0], solve_evals = n_evals)
        if root.outcome in search.NO_ROOT:
            res.update(status = model.OUTSIDE_PRIOR,
                       error = f'outside prior: no {P} in its prior gives {name} = {u:.6g} '
                               f'({search.OUTCOMES[root.outcome]})')
        elif root.outcome == search.FAILED and hist:
            failed = hist[-1][3]
            res.update(status = failed['status'], error = f"the model failed in the solve of {P}: {failed['error']}")
        else:
            res.update(status = model.NO_LIKELIHOOD,
                       error = f'the solve of {P} did not converge ({search.OUTCOMES[root.outcome]}) in {n_evals} '
                               f'model runs (solve.max_evals = {st["max_evals"]})')
        return res
    at_root = root.payload
    point = start.copy()
    point[st['index']] = root.x
    deriv, n_deriv, why = _derivative(run, point, st, root, u)
    out = dict(at_root, runtime = runtime[0], solve_value = root.x, solve_evals = n_evals + n_deriv, log_jac = nan)
    if history:
        out['history'] = hist
    if why is not None:
        return dict(out, status = model.NO_LIKELIHOOD, lp = -inf, error = f'solve of {P}: {why}')
    out['log_jac'] = log_jac = -math.log(abs(deriv))
    lp = at_root['log_prior'] + log_jac + at_root['log_like']
    if at_root['status'] == model.OK and math.isfinite(lp):
        out.update(status = model.OK, lp = lp)
    else:
        out.update(status = model.NO_LIKELIHOOD, lp = -inf)
    return out


def plateau(history, st, u):
    '''
    A plateau of the solve observable seen by one search (history: search.find_root's (phase, x, g, result)
    entries, g = T - u): (T, P from, P to, its log-likelihood term) for the widest set of at least two evaluations
    at different P whose model ran and gave the same finite T (within 1e-9 sigma) inside the u support, else None.
    Example: Rp = the bare-core radius for every log_f_atm that strips the planet. T cannot be inverted there, so
    solve mode leaves out the posterior mass of a plateau (it is a single u value).
    '''
    s = likelihood.sigma_eff(st['ocfg'])
    ran = sorted(((g, x, res) for _, x, g, res in history if res['status'] in search.RAN and math.isfinite(g)),
                 key = lambda e: e[:2])
    best, i = None, 0
    while i < len(ran):
        j = i
        while j + 1 < len(ran) and ran[j + 1][0] - ran[i][0] <= 1e-9*s:
            j += 1
        xs = [x for _, x, _ in ran[i:j + 1]]
        T = ran[i][0] + u
        if j > i and max(xs) > min(xs) and st['bounds'][0] <= T <= st['bounds'][1] \
                and (best is None or max(xs) - min(xs) > best[2] - best[1]):
            best = (T, min(xs), max(xs), float(ran[i][2].get('terms', {}).get(st['observable'], nan)))
        i = j + 1
    return best


###_____The warm-start table_____###

def _local_fit(X, p, q, k, skip = None):
    '''
    (value, slope along each coordinate) at q of a weighted linear least-squares fit to the k points of X (n, d)
    nearest to q, with weights 1/(distance + 1e-6); the weighted mean (slopes 0) when the fit is degenerate.
    skip: an index left out (leave-one-out).
    '''
    d2 = np.sum((X - q)**2, axis = 1)
    if skip is not None:
        d2[skip] = inf
    order = np.argsort(d2, kind = 'stable')[:k]
    order = order[np.isfinite(d2[order])]
    w = 1/(np.sqrt(d2[order]) + 1e-6)
    A = np.column_stack([np.ones(order.size), X[order] - q])
    if order.size >= A.shape[1]:
        coef, _, rank, _ = np.linalg.lstsq(A*w[:, None], p[order]*w, rcond = None)
        if rank == A.shape[1] and np.all(np.isfinite(coef)):
            return float(coef[0]), coef[1:]
    return float(np.sum(w*p[order])/np.sum(w)), np.zeros(X.shape[1])


def fit_table(cfg, thetas, values, log_jac = None):
    '''
    The warm-start table from solved points: thetas (n, ndim) sampler coordinates, values (n,) their P* (NaN: not
    solved, left out) and log_jac (n,) their ln|dP*/du|. Holds the unit-cube coordinates x, multiplied by scale
    (|dP*/dx| of a global linear fit: nearness then means a similar P*), the values p (ln P* for a loguniform
    prior), lslope (ln|dT/dP| = -log_jac, NaN unknown), k (the neighbours of each local fit, 2(ndim + 1)), the unit
    box, the leave-one-out errors and step (the first step of the bracket expansion: their 68th percentile, within
    [1e-6, 0.05] of the prior width of P).
    '''
    st = settings(cfg)
    lo, hi = prm.prior_box(prm.sampled_cfg(cfg))
    thetas, values = np.atleast_2d(np.asarray(thetas, dtype = float)), np.asarray(values, dtype = float)
    ok = np.isfinite(values) & np.all(np.isfinite(thetas), axis = 1)
    X = (thetas[ok] - lo)/(hi - lo)
    p = np.log(values[ok]) if st['log'] else values[ok]
    lslope = -np.asarray(log_jac, dtype = float)[ok] if log_jac is not None else np.full(len(p), nan)
    ndim = X.shape[1]
    k = int(min(len(p), 2*(ndim + 1)))
    scale = np.ones(ndim)
    if len(p) > ndim + 1:
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(p)), X]), p, rcond = None)[0][1:]
        if np.all(np.isfinite(coef)) and np.abs(coef).max() > 0:
            scale = np.maximum(np.abs(coef), 1e-3*np.abs(coef).max())
    X = X*scale
    width = (math.log(st['hi']) - math.log(st['lo'])) if st['log'] else st['hi'] - st['lo']
    loo = np.array([_local_fit(X, p, X[i], k, skip = i)[0] - p[i] for i in range(len(p))]) if len(p) > 1 else \
        np.zeros(len(p))
    step = np.percentile(np.abs(loo), 68) if len(p) > 1 else 1e-3*width
    step = float(min(max(step, 1e-6*width), 0.05*width))
    return {'x': X, 'p': p, 'lslope': lslope, 'k': k, 'scale': scale, 'box_lo': lo, 'box_hi': hi, 'log': st['log'],
            'lo': st['lo'],
            'hi': st['hi'], 'step': step, 'loo': loo, 'names': np.array(prm.sampled_cfg(cfg)['free']),
            'param': st['param'], 'observable': st['observable'], 'index': st['index']}


def predict(table, theta):
    '''
    (guess of P*, first expansion step (in ln P for a loguniform prior), orientation: the sign of dP*/du of the
    local fit, i.e. of dT/dP, 0 when unknown; the expected |dT/dP| (per unit of ln P for a loguniform prior),
    the weighted mean of the neighbours' in log, None when unknown) at a sampler theta.
    '''
    q = (np.asarray(theta, dtype = float) - table['box_lo'])/(table['box_hi'] - table['box_lo'])*table.get('scale', 1.0)
    value, slopes = _local_fit(table['x'], table['p'], q, int(table['k']))
    guess = math.exp(value) if table['log'] else value
    guess = min(max(guess, float(table['lo'])), float(table['hi']))
    slope = None
    lslope = table.get('lslope')
    if lslope is not None and np.isfinite(lslope).sum() >= 2:
        known = np.isfinite(lslope)
        slope = math.exp(_mean_near(table['x'][known], lslope[known], q, int(table['k'])))
        if table['log']:
            slope *= guess # dT/dlnP = P dT/dP
    return guess, float(table['step']), float(np.sign(slopes[int(table['index'])])), slope


def _mean_near(X, y, q, k):
    '''
    The inverse-distance weighted mean of y over the k points of X nearest to q.
    '''
    d = np.sqrt(np.sum((X - q)**2, axis = 1))
    order = np.argsort(d, kind = 'stable')[:k]
    w = 1/(d[order] + 1e-6)
    return float(np.sum(w*y[order])/np.sum(w))


def save_table(table, path):
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **table)
    os.replace(tmp, path) # atomic: workers read it lazily
    return path


def _scalars(table):
    for key in ('k', 'log', 'lo', 'hi', 'step', 'index'):
        table[key] = np.asarray(table[key]).item()
    return table


def load_table(path):
    with np.load(path, allow_pickle = False) as d:
        table = {key: d[key] for key in d.files}
    return _scalars(table)


def table_extras(table, version):
    '''
    The table as chain.npz extras (keys table_<key>, and table_version: the step after which it was fitted, 0 for
    the initialization's), so that a resume continues with the table that was active at its checkpoint. Without a
    table (solve mode with init other than "solve", before the first refit) only table_version = NO_TABLE, so that
    a resume knows the workers ran cold and does not trust a solve_table.npz left by a later, lost refit.
    '''
    if table is None:
        return {'table_version': NO_TABLE}
    out = {f'table_{key}': np.asarray(value) for key, value in table.items()}
    out['table_version'] = int(version)
    return out


def table_from_extras(extra):
    '''
    (table, version) saved by table_extras in a checkpoint's extras: (None, NO_TABLE) when the run had no table at
    that checkpoint, (None, None) for a checkpoint from before table refits existed (no table_version at all).
    '''
    if 'table_version' not in extra:
        return None, None
    version = int(np.asarray(extra['table_version']).item())
    if version == NO_TABLE: # no table_<key> arrays either: _scalars would raise KeyError on the empty dict
        return None, NO_TABLE
    table = {key[6:]: np.asarray(value) for key, value in extra.items()
             if key.startswith('table_') and key != 'table_version'}
    return _scalars(table), version


def worker_table(cfg):
    '''
    The warm-start table of the worker's run directory (model._RUN_DIR), loaded when it first exists and again
    whenever the file changes (a refit replaces it atomically, so its inode changes), or None (no run directory,
    solve.warm_start = false, or not written yet: during the initialization).
    '''
    if not cfg['solve'].get('warm_start') or not model._RUN_DIR:
        return None
    path = os.path.join(model._RUN_DIR, TABLE)
    try:
        s = os.stat(path)
    except FileNotFoundError:
        return None
    signature = (s.st_ino, s.st_mtime_ns, s.st_size)
    cached = _TABLES.get(path)
    if cached is None or cached[0] != signature:
        cached = _TABLES[path] = (signature, load_table(path))
    return cached[1]


def build_table(cfg, run_dir, chain = None, blobs = None):
    '''
    The warm-start table fitted to the initialization's solves (<run_dir>/solve_init.npz) plus the distinct solved
    states of chain (steps, walkers, ndim) with their blobs (status 0, finite solve_value and log_jac), in the order
    init, then by step and walker, keeping the last MAX_TABLE_POINTS. Deterministic: the same inputs give the
    same table. Without chain: the table of the initialization (init_solve's).
    '''
    path = os.path.join(run_dir, 'solve_init.npz')
    ndim = len(prm.sampled_cfg(cfg)['free'])
    if os.path.exists(path):
        with np.load(path, allow_pickle = False) as d:
            thetas, b = d['theta'], d['blobs']
            values, log_jac = b['solve_value'], b['log_jac']
    else: # an initialization other than init = "solve": the chain alone
        thetas, values, log_jac = np.empty((0, ndim)), np.empty(0), np.empty(0)
    if chain is not None and len(chain):
        X = np.asarray(chain, dtype = float).reshape(-1, ndim)
        B = np.asarray(blobs).reshape(-1)
        ok = (B['status'] == model.OK) & np.isfinite(B['solve_value']) & np.isfinite(B['log_jac'])
        thetas = np.concatenate([thetas, X[ok]])
        values = np.concatenate([values, B['solve_value'][ok]])
        log_jac = np.concatenate([log_jac, B['log_jac'][ok]])
        _, first = np.unique(thetas, axis = 0, return_index = True)
        keep = np.sort(first)[-MAX_TABLE_POINTS:]
        thetas, values, log_jac = thetas[keep], values[keep], log_jac[keep]
    return fit_table(cfg, thetas, values, log_jac)


def evals_stats(counts):
    '''
    {mean, p10, p50, p90, max, n, none} of the model runs per evaluation from a histogram counts[k] = evaluations
    with k runs, over the evaluations that ran the model (none: those that did not, e.g. outside the prior).
    '''
    counts = np.asarray(counts, dtype = float)
    none = int(counts[0]) if counts.size else 0
    counts = counts.copy()
    if counts.size:
        counts[0] = 0
    n = counts.sum()
    if not n:
        return {'mean': nan, 'p10': nan, 'p50': nan, 'p90': nan, 'max': nan, 'n': 0, 'none': none}
    k = np.arange(counts.size)
    cum = np.cumsum(counts)/n
    q = {f'p{p}': float(k[np.searchsorted(cum, p/100)]) for p in (10, 50, 90)}
    return dict(mean = float(np.sum(k*counts)/n), max = float(k[counts > 0].max()), n = int(n), none = none, **q)
