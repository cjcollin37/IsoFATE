'''
Solve mode: the sampler moves in the model value of one observable, in place of the parameter that pins it down

With [solve] param = P and observable = O, the sampler moves in u = T(O), the model value of O in its likelihood
space (log10 x for a log10-space observable, the value itself for a linear one), in place of the free parameter
P. u is in T units: dex for a log10 observable, the unit of O for a linear one. Solve mode is for a posterior that
one precise datum pins to a thin, curved ridge, on which emcee's moves do not mix: they propose combinations of
walker positions, which stay on such a surface only when the walkers involved are very close together. Example:
log_f_atm and the outflow H fraction of LHS 1140 b, whose posterior is a curved ridge 0.003 dex thin in
log_f_atm. In (u, phi) the ridge is a smooth, wide direction.

For a sampled point (u, phi), phi being the other free parameters, the root finder of search.py finds P* inside
P's prior support with T(O_model(P*, phi)) = u (to solve.tol sigma of O), and the model run at P* gives every
observable (the blob values belong to P*). The density the sampler sees is the exact change of variables

    p(u, phi) = prior_P(P*) prior_phi(phi) |dP*/du| L(model(P*, phi)),    dP*/du = 1/(dT/dP at P*)

so the posterior over (P, phi) is unchanged. That is why solve mode needs dT/dP: where O is steep in P, a short
stretch of P covers a wide range of u, and without the Jacobian |dP*/du| the sampler would weight that stretch by
its length in u instead of in P.
- dT/dP: a finite difference with h = 1e-5 of P's prior width. solve.derivative = "forward" (default): one more
  model run, at P* + h (P* - h when P* + h is beyond the upper prior bound; where T is undefined at P* + h, a
  second run at P* - h). On LHS 1140 b its ln|J| differs from the central one by 0.009 (median; at most 0.021),
  about the central one's own numerical noise. "central": 2 more model runs, and both one-sided slopes must have
  the same sign. A config saved before solve.derivative existed gets "central", the scheme it was sampled with; a
  resumed run keeps the scheme it last ran with (sampler.run_derivative). The slope must have the sign of the
  bracket (T rising from the side below u to the side above it); a derivative that is 0, undefined, or (central)
  changes sign within +-h gives lp = -inf (status 3) with the reason in the error string. log_jac = -ln|dT/dP| =
  ln|dP*/du| is stored in the blobs.
- The support of u is [v - k s_lo, v + k s_hi], v and s the datum and its effective sigma on each side, k =
  solve.range_sigma (6): the truncation removes only points whose O term is below -k^2/2 = -18 (e^-18 of the
  likelihood). It also keeps the sampler off the plateaus where T cannot be inverted that lie outside it, e.g.
  x = 1 (He not escaping) or the 1e-15 floor of the outflow H fraction. A plateau inside it (Rp = the bare core of
  a stripped planet) has no u of its own; the plateau branch below gives it one (plateau() finds a plateau in a
  search). A limit uses [v - k s, v] (upper) or [v, v + k s] (lower), with a warning: solve mode is meant for
  detections.
- No root in P's prior support for (u, phi): lp = -inf, status 1 ("outside prior: no P in its prior gives this
  u"). One root per bracket is assumed (the density would otherwise be a sum over roots); `scan` reports
  non-monotonic stretches of O(P) inside the u range so that this can be checked.
- The blobs' log_prior is the prior of the physical point (P*, phi); lp = log_prior + log_jac + log_like.

The plateau branch (solve.plateau, default true). A planet stripped at a prior bound P_s of P has the same T = T_s
over the whole stretch of P that strips it, e.g. Rp = the bare-core radius for every log_f_atm below the stripping
edge. No u stands for that stretch, so without the branch its posterior mass is left out: on TOI-406 b, 82% of the
plain posterior. The u beyond the range of T have no root either. The branch maps them linearly onto P's prior
(plateau_map), with the constant Jacobian |dP/du| = |P_o - P_s|/L: first the u beyond T_s (Rp below the bare
core), across the jump of T at the stripping edge from the planets that keep an atmosphere, then the u beyond T_o,
the T at the other bound P_o (usually outside the u support). L is their total length, each starting the search's
convergence margin away from its T. The model runs at that P, and the point counts only when the planet is
stripped there with T = T_s, on the plateau: any other P has a root of its own, which counts it already. So each P
of the prior is reached once, and the posterior over (P, phi) stays unchanged. A plateau whose T lies outside the
u support gets nothing, like any model value there. It costs one model run more than a u without a root (two when
the search did not evaluate both bounds). The table leaves those points out (table_values), since their P is not
a root. A config saved before the option existed gets false: a resumed chain keeps the density it was sampled
with. Walkers cross between the stripped planets and the others only through moves that land in the other stretch
of u, which tau does not see: the summary counts the crossings (sampler.write_summary).

Warm start. After init = "solve" the parent fits a table of P* over the sampler's coordinates and writes it to
<run_dir>/solve_table.npz before the first step: a local linear regression on the k nearest solved points in the
unit cube, the coordinates scaled by how strongly P* depends on each (from a global linear fit). The
initialization's draws cover the whole prior, so after the steps of solve.refit_steps (default 25, 50, 100, 200,
400) the parent refits the table to those draws plus every distinct solved state of the chain so far
(build_table), between two steps, while no evaluation is running. Workers load the file lazily (model._RUN_DIR)
and reload it when it changes (worker_table); they never write it. On the LHS 1140 b chain at n_steps = 1e5 and
n_atmodeller = 100, a refit after step 250 (its first 250 steps ran on the initialization's table), from 1161
solved points, cut the median error of the guess from 0.0067 to 0.00024 dex, and the model runs per solve (with
the forward derivative) from a mean of about 10.5 to about 5.6.

A solve starts from the table's guess and expands a bracket around it (search.Root): the first step is 1.5 Newton
steps with the slope |dT/dP| of the neighbouring solves, within 1/16 and 16 times the 68th percentile of the
table's leave-one-out errors. The table changes only the path of a solve, not its root (to solve.tol). Between
refits the result depends only on theta and that file, and every checkpoint holds the active table (table_extras;
or table_version = -1 while the run has none), which a resume writes back (or, for -1, removes). A resume also
redoes a refit that an interruption cut short, so a resumed run is bit-identical. Without the table (the
initialization itself, check, resolution) the search starts from both prior bounds.

See "Solve mode" in the README.
'''

import math
import os

import numpy as np

from isofate.mcmc import likelihood, model, search
from isofate.mcmc import params as prm

nan, inf = math.nan, math.inf
TABLE = 'solve_table.npz' # the warm-start table, in the run directory
DERIVATIVE_STEP = 1e-5 # of P's prior width: the finite-difference step of dT/dP
INPUT_PARAMS = ('Mp', 'age', 'albedo') # parameters the input observables (Mp, age, Teq) depend on
EVALS_GUESS = 15 # model runs per cold solve, for cost estimates before any was measured

MAX_TABLE_POINTS = 5000 # a refitted table keeps at most this many solved points (the most recent)
NO_TABLE = -1 # table_version of a checkpoint written while the run had no table yet (init other than "solve")
OLD_DERIVATIVE = 'central' # the scheme of configs saved before solve.derivative existed
PLATEAU_EQUAL = 1e-9 # values of T within this many sigma are the same, on a plateau

_TABLES = {} # path -> (file signature, table), the worker's lazily loaded warm-start tables


###_____Settings and coordinates_____###

def settings(cfg):
    '''
    Returns the settings of solve mode: the solved parameter, the observable, the u support and the tolerances
    Configs saved before solve.derivative existed get "central", the scheme they were sampled with; a resumed run
    sets the key from its checkpoint first (sampler.run_derivative).

    Inputs:
        - cfg: resolved config in solve mode [dict]

    Output: dict, search.bracket() of solve.param plus
        - param, index: the solved parameter P and its slot in theta (the slot of u in the sampler's theta)
        - lo, hi: P's prior support [unit of P]
        - log: True for a loguniform prior of P [Bool]
        - observable: the solve observable O [str]
        - ocfg: its config [dict]
        - name: the name of u, u_<O> [str]
        - bounds: the support of u, (lo, hi) [T units]
        - tol: solve.tol, the convergence tolerance in units of sigma [ndim]
        - max_evals: solve.max_evals, the most model runs of the search [int]
        - warm_start: solve.warm_start [Bool]
        - h: the finite-difference step of dT/dP, 1e-5 of P's prior width [unit of P]
        - derivative: "forward" or "central" [str]
        - plateau: solve.plateau, the plateau branch; False for a config saved before the option existed, the
          density it was sampled with [Bool]
    '''
    sv = cfg['solve']
    st = search.bracket(cfg, sv['param'])
    st.update(observable = sv['observable'], ocfg = cfg['observables'][sv['observable']], name = sv['name'],
              bounds = tuple(sv['bounds']), tol = float(sv['tol']), max_evals = int(sv['max_evals']),
              warm_start = bool(sv['warm_start']), h = DERIVATIVE_STEP*(st['hi'] - st['lo']), # in P, not ln P
              derivative = sv.get('derivative', OLD_DERIVATIVE), plateau = bool(sv.get('plateau', False)))
    return st


def refit_steps(cfg):
    '''
    Returns the steps after which the sampler refits the warm-start table to the chain (solve.refit_steps)
    Configs saved before the option existed get the default (config.SOLVE_DEFAULTS); without warm starts there
    are none.

    Inputs:
        - cfg: resolved config in solve mode [dict]

    Output: the steps, sorted [list of int]
    '''
    sv = cfg['solve']
    if not sv.get('warm_start', True): # no table to refit
        return []
    from isofate.mcmc.config import SOLVE_DEFAULTS
    return sorted(int(k) for k in sv.get('refit_steps', SOLVE_DEFAULTS['refit_steps']))


def physical(theta, value, cfg):
    '''
    Puts value in the slot of the solved parameter
    Gives the physical theta (cfg['free'] order) of a sampler theta with value = P* in place of u, or the sampler
    theta of a physical one with value = u.

    Inputs:
        - theta: a sampler or physical theta, (ndim,) or (n, ndim) [array]
        - value: the value for the slot, a number or (n,) [unit of P, or T units for u]
        - cfg: resolved config in solve mode [dict]

    Output: a copy of theta with value in that slot [array]
    '''
    out = np.array(theta, dtype = float)
    out[..., cfg['solve']['index']] = value # the last axis: one theta or many
    return out


def to_sampled(theta, blobs, cfg):
    '''
    Converts physical points to the sampler's coordinates: u = T(O) from their blob values
    u is NaN where O is undefined (the H/He-wind rule of search.search_values included).

    Inputs:
        - theta: physical points (n, ndim) [array]
        - blobs: their blobs (n,) [structured array]
        - cfg: resolved config in solve mode [dict]

    Output: the sampler thetas (n, ndim) [array]
    '''
    st = settings(cfg)
    T = np.atleast_1d(search.g_of(blobs, st['observable'], st['ocfg'], 0.0)) # g with target 0: T itself
    return physical(theta, T, cfg)


###_____One solve-mode evaluation_____###

def _base(theta, cfg):
    '''
    The result of a solve-mode evaluation before any model run: lp = -inf, P and the observables NaN, no runs
    '''
    res = {'status': nan, 'runtime': 0.0, 'lp': -inf, 'log_prior': -inf, 'log_like': nan, 'terms': {},
           'obs': model._nan_obs(), 'params': prm.theta_to_params(physical(theta, nan, cfg), cfg), 'error': None,
           'solve_value': nan, 'solve_evals': 0, 'log_jac': nan}
    return res


def _derivative(run, point, st, root, u):
    '''
    Calculates dT/dP at the root by a finite difference, for the Jacobian |dP*/du| = 1/|dT/dP|
    Forward: one run at P* + h, or at P* - h when P* + h is beyond the upper prior bound; a second run, at P* - h,
    when T is undefined at P* + h (or the model failed there). Central: a run on each side; the two one-sided
    slopes must agree in sign (a single one where one side is beyond a prior bound or undefined). The slope must
    also have the sign of the bracket the root came from: T rises from the end below u to the end above it.

    Inputs:
        - run: point -> (g, ran, result), g = T - u [function]
        - point: the physical theta at the root [array]
        - st: solve settings (h, index, lo, hi, derivative, observable, param) [dict]
        - root: the converged search (x, g, a, sign_a) [search.Root]
        - u: the sampled value of T [T units]

    Output: (deriv, n, why)
        - deriv: dT/dP at the root; NaN when unusable [T units per unit of P]
        - n: model runs it took [int]
        - why: why it is unusable, for the error string; None when usable [str]
    '''
    h, j, n = st['h'], st['index'], 0
    T0 = root.g + u # T at the root (g = T - u)
    T = {} # T on each usable side: +1 for P* + h, -1 for P* - h
    forward = st.get('derivative', OLD_DERIVATIVE) == 'forward' # settings without the key: central
    for side in ((1, -1) if forward else (-1, 1)):
        x = root.x + side*h
        if not st['lo'] <= x <= st['hi']: # beyond a prior bound: the other side only
            continue
        p = point.copy()
        p[j] = x
        g, ran, _ = run(p)
        n += 1
        if ran and math.isfinite(g):
            T[side] = g + u
            if forward: # one usable side is enough
                break
    slopes = {side: (T[side] - T0)/(side*h) for side in T} # the one-sided slopes
    if not slopes:
        return nan, n, f"{st['observable']} is undefined on both sides of the root (+-{h:.3g})"
    if len(slopes) == 2: # central: both sides must agree
        if np.sign(slopes[-1]) != np.sign(slopes[1]) or slopes[-1] == 0 or slopes[1] == 0:
            return nan, n, (f'dT/d{st["param"]} changes sign or vanishes within +-{h:.3g} of the root (one-sided '
                            f'slopes {slopes[-1]:.4g} and {slopes[1]:.4g})')
        deriv = (T[1] - T[-1])/(2*h) # the central difference
    else: # forward, or central with one side beyond a bound or undefined
        deriv = next(iter(slopes.values()))
    if not math.isfinite(deriv) or deriv == 0:
        return nan, n, f'dT/d{st["param"]} = {deriv:.4g} at the root'
    if root.a is not None and np.sign(deriv) != -root.sign_a: # a root from a bracket: dT/dP has the sign -sign_a
        return nan, n, (f'dT/d{st["param"]} = {deriv:.4g} at the root has the wrong sign for its bracket (T must rise '
                        'from below u to above it)')
    return deriv, n, None


def plateau_map(u, bounds, lo, hi, T_lo, T_hi, stripped_lo, stripped_hi, margins = (0.0, 0.0)):
    '''
    Returns the point of P's prior that the plateau branch gives a u without a root, and |dP/du| there
    A planet stripped at a prior bound P_s of P has T = T_s there and over the whole stretch of P that strips it
    (e.g. Rp = the bare core for every log_f_atm below the stripping edge): a plateau, which no u stands for. The
    u beyond the range of T have no root either. Two stretches of them serve the plateau, in this order: the u
    beyond T_s (Rp below the bare core), then those beyond T_o, T at the other bound P_o (above it when T rises
    away from the plateau), each from its end next to T to the edge of the u support. Together they have length L,
    and the distance d of u into them, counted through the first and on into the second, maps linearly onto P's
    prior:

        P = P_s + (d/L) (P_o - P_s),    |dP/du| = |P_o - P_s|/L

    So the stripped planets lie next to T_s, usually the end nearest the datum (where the solve init draws u), and
    across the jump of T at the stripping edge from the planets that keep an atmosphere. The second stretch only
    adds room when the first is short; usually T_o lies beyond the u support and it is empty. Only the P that are
    stripped belong to the plateau (the others have roots of their own), and the caller keeps only those on the
    plateau (stripped, with T = T_s). With both bounds stripped, and the same T at both, the u below and above the
    plateau map onto the whole prior. Each stretch starts a margin away from its T: a u that close converges at
    that bound in the search instead (search.within: tol sigma, sigma on that side of u). So each P of the prior is
    reached once, as a root or here, and the posterior over (P, phi) is unchanged. A plateau whose T lies outside
    the u support gets no point, as any model value outside it: its likelihood term is below -range_sigma^2/2 for a
    detection.

    Inputs:
        - u: the sampled value of T [T units]
        - bounds: the u support (u_lo, u_hi) [T units]
        - lo, hi: P's prior bounds [unit of P]
        - T_lo, T_hi: T at them; NaN where undefined or where the model did not run [T units]
        - stripped_lo, stripped_hi: whether the planet loses its atmosphere there [Bool]
        - margins: (above, below), the search's convergence margins for a u below and above a bound's T [T units]

    Output: (P, jac, T_s, stretches), or None when u is in neither stretch
        - P: the point [unit of P]
        - jac: |dP/du| [unit of P per T unit]
        - T_s: T on the plateau [T units]
        - stretches: the two stretches, each as (end next to T, edge of the support), in the order of d [T units]
    '''
    u_lo, u_hi = bounds
    above, below = margins
    if not (stripped_lo or stripped_hi) or not (math.isfinite(T_lo) and math.isfinite(T_hi)):
        return None
    T_s = T_lo if stripped_lo else T_hi
    if not u_lo <= T_s <= u_hi or (stripped_lo and stripped_hi and T_lo != T_hi): # outside the support, or no plateau
        return None

    def clip(x):
        '''
        x clipped to the u support
        '''
        return min(max(x, u_lo), u_hi)

    if stripped_lo and stripped_hi: # stripped everywhere: the u below and above the plateau, onto the whole prior
        P_s, P_o = lo, hi
        stretches = ((clip(T_s - above), u_lo), (clip(T_s + below), u_hi))
    else:
        P_s, P_o, T_o = (lo, hi, T_hi) if stripped_lo else (hi, lo, T_lo)
        if T_o == T_s:
            return None
        if T_o > T_s: # T rises away from the plateau: the u below T_s, then those above T_o
            stretches = ((clip(T_s - above), u_lo), (clip(T_o + below), u_hi))
        else: # T falls away from it: the u above T_s, then those below T_o
            stretches = ((clip(T_s + below), u_hi), (clip(T_o - above), u_lo))
    lengths = [abs(edge - start) for start, edge in stretches]
    total = sum(lengths)
    if not total > 0:
        return None
    for (start, edge), offset in zip(stretches, (0.0, lengths[0])):
        sign = 1.0 if edge >= start else -1.0
        d = sign*(u - start) # into the stretch, from its end next to T; the edge of the support is part of it
        if 0 < d <= sign*(edge - start):
            return P_s + (offset + d)/total*(P_o - P_s), abs(P_o - P_s)/total, T_s, stretches
    return None


def _plateau_branch(run, start, st, u, hist):
    '''
    Runs the plateau branch of an evaluation whose search found no root (g of one sign at both bounds)
    T and whether the planet is stripped at each prior bound come from the search's history, or from a run there
    when the search did not evaluate it; plateau_map gives the point, and one more run evaluates it.

    Inputs:
        - run: point -> (g, ran, result), g = T - u [function]
        - start: the physical theta, P to be set [array]
        - st: solve settings (index, lo, hi, bounds) [dict]
        - u: the sampled value of T [T units]
        - hist: the search's (phase, x, g, result) entries [list]

    Output: (mapped, entries)
        - mapped: (P, jac, T_s, stretches, T at P, result at P), or None when u has no point on a plateau [tuple]
        - entries: the branch's runs as ('plateau', x, g, result) history entries [list]
    '''
    entries, ends = [], []
    for x in (st['lo'], st['hi']):
        seen = [(g, r) for _, xh, g, r in hist if xh == x] # the search evaluates the bounds exactly
        if seen:
            g, r = seen[-1]
        else:
            point = start.copy()
            point[st['index']] = x
            g, _, r = run(point)
            entries.append(('plateau', x, g, r))
        ran = r['status'] in search.RAN
        ends.append((g + u if ran else nan, ran and r['obs'].get('exhausted') == 1))
    (T_lo, stripped_lo), (T_hi, stripped_hi) = ends
    # a u this close to a bound's T converges there in the search, so the stretches start that far from it
    margins = tuple(st['tol']*likelihood.sigma_eff(st['ocfg'], above = side) for side in (True, False))
    mapped = plateau_map(u, st['bounds'], st['lo'], st['hi'], T_lo, T_hi, stripped_lo, stripped_hi, margins)
    if mapped is None:
        return None, entries
    P, jac, T_s, stretches = mapped
    point = start.copy()
    point[st['index']] = P
    g, _, r = run(point)
    entries.append(('plateau', P, g, r))
    return (P, jac, T_s, stretches, g + u, r), entries


def evaluate(theta, cfg, n_steps = None, n_atmodeller = None, table = None, history = False):
    '''
    Runs one solve-mode evaluation at a sampler theta: the root, the Jacobian and the model at the root
    The sampler theta holds u in place of solve.param (see the module docstring). A theta outside the prior (u
    outside its support included) is rejected before any model run, and so is one that the input observables
    rule out, when P is not an input parameter. Then P* is solved for, dT/dP is taken there, and
    lp = log_prior + log_jac + log_like. With solve.plateau, a u without a root (g of one sign at both prior
    bounds) gets the P of plateau_map instead, with log_jac = ln|dP/du| of the map, when a bound is stripped. The
    point counts when the planet there is stripped with T = T_s, on the plateau; any other P has a root of its own,
    so the point is outside the support (status 1).

    Inputs:
        - theta: a sampler theta (ndim,) [array]
        - cfg: resolved config in solve mode [dict]
        - n_steps, n_atmodeller: the model resolution; None: the config's (model.evaluate) [int]
        - table: the warm-start table (fit_table); None: a cold start from both prior bounds [dict]
        - history: True adds the search's evaluations (for search.history_table) [Bool]

    Output: dict with the keys of model.evaluate's result (its result at the root when the solve converged)
        - lp: log_prior + log_jac + log_like; -inf unless status 0
        - status: 0 ok; 1 outside the prior, or no P in its prior gives u; 3 likelihood -inf, the solve did not
          converge, or no usable derivative; after a model failure, the status of the last run of the search
          (normally the run that failed)
        - error: why the solve failed (no root, a model failure, not converged, no usable derivative); else None
          [str]
        - log_prior: the prior of the physical point (P*, phi); -inf without a root
        - solve_value: P*; NaN without a root [unit of P]
        - solve_evals: model runs, the derivative's included [int]
        - log_jac: ln|dP*/du|; NaN without a usable derivative
        - runtime: isocalc time of all the runs [s]
        - history: the search's (phase, x, g, result) entries, the plateau branch's last; only with
          history = True [list]
        - plateau: with the plateau branch only: its two stretches of u, T_s, the point P and |dP/du| [dict]
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
    runtime = [0.0] # isocalc time of every run [s], summed by run

    def run(point):
        '''
        One model run at a physical point: (g = T - u, whether the model ran, the model.evaluate result)
        '''
        r = model.evaluate(point, cfg, n_steps, n_atmodeller)
        runtime[0] += r['runtime']
        g = float(search.g_of(r['obs'], st['observable'], st['ocfg'], u)[0]) # the H/He-wind rule included
        return g, r['status'] in search.RAN, r

    guess, step, orient, slope = predict(table, theta) if table is not None else (None, None, 0.0, None)
    start = physical(theta, nan, cfg) # the physical theta, P to be found
    root, hist = search.find_root(run, start, st, u, st['tol'], st['max_evals'], guess, step, orient, slope)
    n_evals = root.evals
    if history:
        res['history'] = hist
    P, name = st['param'], st['name']
    if root.outcome in (search.ABOVE, search.BELOW) and st['plateau']: # a u beyond a plateau of T?
        mapped, entries = _plateau_branch(run, start, st, u, hist)
        hist, n_evals = hist + entries, n_evals + len(entries)
        if history:
            res['history'] = hist
        if mapped is not None:
            P_m, jac, T_s, stretches, T_m, r = mapped
            res.update(runtime = runtime[0], solve_evals = n_evals,
                       plateau = {'stretches': stretches, 'T': T_s, 'P': P_m, 'jac': jac})
            if r['status'] not in search.RAN:
                return dict(res, status = r['status'], error = f"the model failed in the plateau branch of {P} "
                                                                 f"(at {P} = {P_m:.6g}): {r['error']}")
            # on the plateau: stripped, with T = T_s (to PLATEAU_EQUAL sigma); any other P has a root of its own
            flat = abs(T_m - T_s) <= PLATEAU_EQUAL*likelihood.sigma_eff(st['ocfg'])
            if r['obs'].get('exhausted') != 1 or not flat:
                why = ('keeps its atmosphere' if r['obs'].get('exhausted') != 1 else
                       f'is stripped but not on the plateau (T = {T_m:.6g}, the plateau {T_s:.6g})')
                return dict(res, status = model.OUTSIDE_PRIOR,
                            error = f'outside prior: no {P} in its prior gives {name} = {u:.6g} '
                                    f"({search.OUTCOMES[root.outcome]}), and the plateau branch's {P} = "
                                    f'{P_m:.6g} {why}')
            # the stripped planet at P_m, with the Jacobian of the linear map
            out = dict(r, runtime = runtime[0], solve_value = P_m, solve_evals = n_evals, log_jac = math.log(jac),
                       plateau = res['plateau'])
            if history:
                out['history'] = hist
            lp = r['log_prior'] + out['log_jac'] + r['log_like']
            if r['status'] == model.OK and math.isfinite(lp):
                return dict(out, status = model.OK, lp = lp)
            return dict(out, status = model.NO_LIKELIHOOD, lp = -inf)
    if root.outcome != search.CONVERGED:
        res.update(runtime = runtime[0], solve_evals = n_evals)
        if root.outcome in search.NO_ROOT: # no P in its prior gives u: (u, phi) is outside the support
            res.update(status = model.OUTSIDE_PRIOR,
                       error = f'outside prior: no {P} in its prior gives {name} = {u:.6g} '
                               f'({search.OUTCOMES[root.outcome]})')
        elif root.outcome == search.FAILED and hist:
            failed = hist[-1][3] # the last run: the failed one, unless the lower bound failed and the upper ran
            res.update(status = failed['status'], error = f"the model failed in the solve of {P}: {failed['error']}")
        else: # not converged within solve.max_evals, or undefined inside the bracket
            res.update(status = model.NO_LIKELIHOOD,
                       error = f'the solve of {P} did not converge ({search.OUTCOMES[root.outcome]}) in {n_evals} '
                               f'model runs (solve.max_evals = {st["max_evals"]})')
        return res
    at_root = root.payload # model.evaluate's result at P*
    point = start.copy()
    point[st['index']] = root.x
    deriv, n_deriv, why = _derivative(run, point, st, root, u)
    out = dict(at_root, runtime = runtime[0], solve_value = root.x, solve_evals = n_evals + n_deriv, log_jac = nan)
    if history:
        out['history'] = hist
    if why is not None: # no usable derivative, so no density
        return dict(out, status = model.NO_LIKELIHOOD, lp = -inf, error = f'solve of {P}: {why}')
    out['log_jac'] = log_jac = -math.log(abs(deriv)) # ln|dP*/du| = -ln|dT/dP|
    lp = at_root['log_prior'] + log_jac + at_root['log_like'] # the change of variables (module docstring)
    if at_root['status'] == model.OK and math.isfinite(lp):
        out.update(status = model.OK, lp = lp)
    else:
        out.update(status = model.NO_LIKELIHOOD, lp = -inf)
    return out


def plateau(history, st, u):
    '''
    Finds a plateau of the solve observable among the evaluations of one search
    A plateau is the widest set of at least two evaluations at different P whose model ran and gave the same
    finite T (within PLATEAU_EQUAL sigma), inside the u support. Example: Rp = the bare-core radius for every
    log_f_atm that strips the planet. T cannot be inverted there: the plateau branch samples a plateau of planets
    stripped at a prior bound (plateau_covered), and any other plateau's posterior mass is left out (it is a single
    u value). check in solve mode prints it.

    Inputs:
        - history: search.find_root's (phase, x, g, result) entries, g = T - u [list]
        - st: solve settings (ocfg, bounds, observable) [dict]
        - u: the sampled value of T in that search [T units]

    Output: (T, P_from, P_to, term), or None without a plateau
        - T: the plateau's value of T [T units]
        - P_from, P_to: the stretch of P it was seen over [unit of P]
        - term: the log-likelihood term of O there [ndim]
    '''
    s = likelihood.sigma_eff(st['ocfg']) # the mean of the two sides' sigma
    ran = sorted(((g, x, res) for _, x, g, res in history if res['status'] in search.RAN and math.isfinite(g)),
                 key = lambda e: e[:2]) # the runs with a finite T, by T, then P
    best, i = None, 0
    while i < len(ran): # groups of equal T: each run within 1e-9 sigma of the group's first
        j = i
        while j + 1 < len(ran) and ran[j + 1][0] - ran[i][0] <= PLATEAU_EQUAL*s:
            j += 1
        xs = [x for _, x, _ in ran[i:j + 1]]
        T = ran[i][0] + u
        # at least two runs at different P, inside the u support; the widest stretch of P wins
        if j > i and max(xs) > min(xs) and st['bounds'][0] <= T <= st['bounds'][1] \
                and (best is None or max(xs) - min(xs) > best[2] - best[1]):
            best = (T, min(xs), max(xs), float(ran[i][2].get('terms', {}).get(st['observable'], nan)))
        i = j + 1
    return best


def plateau_covered(history, st, u, flat):
    '''
    Returns whether the plateau branch covers a plateau that plateau() found in a search: the branch is on, every
    run on the plateau lost its atmosphere, and the plateau reaches a prior bound of P

    Inputs:
        - history: search.find_root's (phase, x, g, result) entries, g = T - u [list]
        - st: solve settings (ocfg, lo, hi, plateau) [dict]
        - u: the sampled value of T in that search [T units]
        - flat: what plateau() returned for that history [tuple or None]

    Output: True when covered [Bool]
    '''
    if flat is None or not st.get('plateau', False):
        return False
    T, a, b, _ = flat
    s = likelihood.sigma_eff(st['ocfg'])
    on = [r for _, x, g, r in history if r['status'] in search.RAN and math.isfinite(g)
          and abs(g + u - T) <= PLATEAU_EQUAL*s] # the runs on the plateau
    return bool(on) and all(r['obs'].get('exhausted') == 1 for r in on) and (a == st['lo'] or b == st['hi'])


###_____The warm-start table_____###

def _local_fit(X, p, q, k, skip = None):
    '''
    Fits a weighted linear model to the k points of X nearest to q: the value and the slopes at q
    Weighted linear least squares in the coordinates relative to q, so that the intercept is the value at q, each
    row weighted by 1/(distance + 1e-6). When the fit is degenerate (too few points, rank deficient or not
    finite), the weighted mean, with slopes 0.

    Inputs:
        - X: coordinates of the points (n, d) [array]
        - p: their values (n,) [array]
        - q: where to fit (d,) [array]
        - k: the number of neighbours [int]
        - skip: the index of a point to leave out (leave-one-out); None: none [int]

    Output: (value, slopes)
        - value: the fit at q [float]
        - slopes: dp/dX along each coordinate (d,) [array]
    '''
    d2 = np.sum((X - q)**2, axis = 1) # squared distances to q
    if skip is not None:
        d2[skip] = inf
    order = np.argsort(d2, kind = 'stable')[:k] # the k nearest; stable, so ties break by index
    order = order[np.isfinite(d2[order])] # without the skipped point
    w = 1/(np.sqrt(d2[order]) + 1e-6) # inverse-distance weights, finite at q itself
    A = np.column_stack([np.ones(order.size), X[order] - q]) # centred on q: coefficient 0 is the value at q
    if order.size >= A.shape[1]: # enough points for the value and d slopes
        coef, _, rank, _ = np.linalg.lstsq(A*w[:, None], p[order]*w, rcond = None)
        if rank == A.shape[1] and np.all(np.isfinite(coef)):
            return float(coef[0]), coef[1:]
    return float(np.sum(w*p[order])/np.sum(w)), np.zeros(X.shape[1])


def fit_table(cfg, thetas, values, log_jac = None):
    '''
    Fits the warm-start table to solved points
    The table guesses P* (ln P* for a loguniform prior) at a sampler theta by a local linear fit on its nearest
    solved points (predict). Its coordinates are the unit cube of the sampler's prior box, multiplied by scale,
    the |dP*/dx| of a global linear fit (floored at 1e-3 of the largest), so that nearness means a similar P*.
    The leave-one-out errors measure how good its guesses are, and their 68th percentile sets the first step of
    the bracket expansion.

    Inputs:
        - cfg: resolved config in solve mode [dict]
        - thetas: sampler coordinates of the solved points (n, ndim) [array]
        - values: their P* (n,); NaN: not solved, left out [unit of P]
        - log_jac: their ln|dP*/du| (n,); None: unknown [array]

    Output: dict
        - x: the scaled unit-cube coordinates of the solved points (n, ndim) [array]
        - p: their P*, or ln P* for a loguniform prior (n,) [array]
        - lslope: ln|dT/dP| = -log_jac; NaN where unknown (n,) [array]
        - k: the neighbours of each local fit, 2(ndim + 1), or fewer with fewer points [int]
        - scale: the coordinate scales (ndim,); ones with ndim + 1 points or fewer [array]
        - box_lo, box_hi: the unit box, the sampler's prior box (ndim,) [array]
        - log: True when P's prior is loguniform [Bool]
        - lo, hi: P's prior support [unit of P]
        - step: the first step of the bracket expansion: the 68th percentile of |loo|, within 1e-6 and 0.05 of
          P's prior width (1e-3 of it with fewer than 2 points); in ln P for a loguniform prior
        - loo: the leave-one-out errors of the guess of p (n,) [array]
        - names: the names of the sampler's coordinates [array of str]
        - param, observable: P and O [str]
        - index: the slot of P (and of u) in theta [int]
    '''
    st = settings(cfg)
    lo, hi = prm.prior_box(prm.sampled_cfg(cfg)) # the sampler's box: u's support in the slot of P
    thetas, values = np.atleast_2d(np.asarray(thetas, dtype = float)), np.asarray(values, dtype = float)
    ok = np.isfinite(values) & np.all(np.isfinite(thetas), axis = 1) # the solved points
    X = (thetas[ok] - lo)/(hi - lo) # unit cube
    p = np.log(values[ok]) if st['log'] else values[ok] # ln P* for a loguniform prior, the search's coordinate
    lslope = -np.asarray(log_jac, dtype = float)[ok] if log_jac is not None else np.full(len(p), nan) # ln|dT/dP|
    ndim = X.shape[1]
    k = int(min(len(p), 2*(ndim + 1))) # twice the coefficients of a local linear fit
    scale = np.ones(ndim)
    if len(p) > ndim + 1: # enough points for a global linear fit
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(p)), X]), p, rcond = None)[0][1:]
        if np.all(np.isfinite(coef)) and np.abs(coef).max() > 0:
            # |dp/dx| of each coordinate; the floor keeps every coordinate in the distances
            scale = np.maximum(np.abs(coef), 1e-3*np.abs(coef).max())
    X = X*scale
    width = (math.log(st['hi']) - math.log(st['lo'])) if st['log'] else st['hi'] - st['lo'] # P's prior width, in p
    # the leave-one-out errors of the guess: n local fits over the n points
    loo = np.array([_local_fit(X, p, X[i], k, skip = i)[0] - p[i] for i in range(len(p))]) if len(p) > 1 else \
        np.zeros(len(p))
    step = np.percentile(np.abs(loo), 68) if len(p) > 1 else 1e-3*width # about the 1-sigma error of a guess
    step = float(min(max(step, 1e-6*width), 0.05*width))
    return {'x': X, 'p': p, 'lslope': lslope, 'k': k, 'scale': scale, 'box_lo': lo, 'box_hi': hi, 'log': st['log'],
            'lo': st['lo'],
            'hi': st['hi'], 'step': step, 'loo': loo, 'names': np.array(prm.sampled_cfg(cfg)['free']),
            'param': st['param'], 'observable': st['observable'], 'index': st['index']}


def predict(table, theta):
    '''
    Predicts the start of a warm solve at a sampler theta from the warm-start table

    Inputs:
        - table: the warm-start table (fit_table) [dict]
        - theta: a sampler theta (ndim,) [array]

    Output: (guess, step, orient, slope), as search.Root takes them
        - guess: P* from the local fit, within P's prior support [unit of P]
        - step: the first step of the expansion, the table's step; in ln P for a loguniform prior
        - orient: the sign of dP*/du of the local fit, i.e. of dT/dP; 0 when unknown [ndim]
        - slope: the expected |dT/dP| (per unit of ln P for a loguniform prior), the inverse-distance weighted mean
          of the neighbours' in log (a geometric mean); None when fewer than 2 solved points have one
    '''
    # theta in the table's coordinates: the unit cube, scaled (scale 1 for a table without one)
    q = (np.asarray(theta, dtype = float) - table['box_lo'])/(table['box_hi'] - table['box_lo'])*table.get('scale', 1.0)
    value, slopes = _local_fit(table['x'], table['p'], q, int(table['k']))
    guess = math.exp(value) if table['log'] else value
    guess = min(max(guess, float(table['lo'])), float(table['hi'])) # inside P's prior support
    slope = None
    lslope = table.get('lslope') # None for a table without slopes
    if lslope is not None and np.isfinite(lslope).sum() >= 2:
        known = np.isfinite(lslope)
        slope = math.exp(_mean_near(table['x'][known], lslope[known], q, int(table['k']))) # a geometric mean
        if table['log']:
            slope *= guess # dT/dlnP = P dT/dP
    # orient: the sign of the local slope of P* along u, i.e. of dT/dP
    return guess, float(table['step']), float(np.sign(slopes[int(table['index'])])), slope


def _mean_near(X, y, q, k):
    '''
    The inverse-distance weighted mean of y over the k points of X nearest to q (weights 1/(distance + 1e-6))
    '''
    d = np.sqrt(np.sum((X - q)**2, axis = 1))
    order = np.argsort(d, kind = 'stable')[:k]
    w = 1/(d[order] + 1e-6)
    return float(np.sum(w*y[order])/np.sum(w))


###_____Table files and checkpoints_____###

def save_table(table, path):
    '''
    Saves the warm-start table, an npz of its entries, atomically: to a temporary file, then moved into place
    Workers read the file lazily, so they must never see a partial one.

    Inputs:
        - table: the warm-start table [dict]
        - path: the output file, <run_dir>/solve_table.npz [str]

    Output: path [str]
    '''
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **table)
    os.replace(tmp, path) # atomic: workers read it lazily
    return path


def _scalars(table):
    '''
    Turns the table's scalar entries back into numbers (np.load and the checkpoint give 0-d arrays)
    '''
    for key in ('k', 'log', 'lo', 'hi', 'step', 'index'):
        table[key] = np.asarray(table[key]).item()
    return table


def load_table(path):
    '''
    Loads a warm-start table saved by save_table

    Inputs:
        - path: the npz file [str]

    Output: the table, its scalars as numbers [dict]
    '''
    with np.load(path, allow_pickle = False) as d:
        table = {key: d[key] for key in d.files}
    return _scalars(table)


def table_extras(table, version):
    '''
    Packs the active warm-start table into the extras of a checkpoint (chain.npz)
    A resume then continues with the table that was active at its checkpoint. Without a table (solve mode with
    init other than "solve", before the first refit) only table_version = NO_TABLE, so that a resume knows the
    workers ran cold and does not trust a solve_table.npz left by a later, lost refit.

    Inputs:
        - table: the active table; None: none [dict]
        - version: the step after which it was fitted; 0 for the initialization's [int]

    Output: the extras [dict]
        - table_<key>: each entry of the table [array]
        - table_version: version; NO_TABLE without a table [int]
    '''
    if table is None:
        return {'table_version': NO_TABLE}
    out = {f'table_{key}': np.asarray(value) for key, value in table.items()}
    out['table_version'] = int(version)
    return out


def table_from_extras(extra):
    '''
    Unpacks the table that table_extras saved in a checkpoint's extras

    Inputs:
        - extra: the checkpoint's extras [dict]

    Output: (table, version)
        - (table, version) when the run had a table at that checkpoint
        - (None, NO_TABLE) when it had none
        - (None, None) for a checkpoint from before table refits existed (no table_version at all)
    '''
    if 'table_version' not in extra:
        return None, None
    version = int(np.asarray(extra['table_version']).item())
    if version == NO_TABLE: # no table_<key> arrays either: _scalars would raise KeyError on the empty dict
        return None, NO_TABLE
    table = {key[6:]: np.asarray(value) for key, value in extra.items() # key[6:]: the key without 'table_'
             if key.startswith('table_') and key != 'table_version'}
    return _scalars(table), version


###_____The workers' table and the refits_____###

def worker_table(cfg):
    '''
    Returns the warm-start table of this worker's run directory (model._RUN_DIR)
    The file is loaded when it first exists, and again whenever it changes: a refit replaces it atomically, so its
    inode changes. Each worker keeps its own copy (_TABLES) and never writes the file.

    Inputs:
        - cfg: the worker's config [dict]

    Output: the table; None without a run directory, with solve.warm_start = false, or while there is no file
        (during the initialization, or before the first refit of a run with another init) [dict]
    '''
    if not cfg['solve'].get('warm_start') or not model._RUN_DIR:
        return None
    path = os.path.join(model._RUN_DIR, TABLE)
    try:
        s = os.stat(path)
    except FileNotFoundError: # no table (yet): cold solves
        return None
    signature = (s.st_ino, s.st_mtime_ns, s.st_size) # a replaced file has a new inode
    cached = _TABLES.get(path)
    if cached is None or cached[0] != signature: # first use, or replaced by a refit
        cached = _TABLES[path] = (signature, load_table(path))
    return cached[1]


def table_values(cfg, thetas, blobs):
    '''
    Returns the solved values that the warm-start table may learn from: solve_value, NaN at the points of the
    plateau branch, whose P is not a root of T = u and would spoil the local fits next to the roots
    Those are stripped and have T farther from u than the solve's tolerance; a root, stripped or not, never does,
    so a run without the branch keeps every point it had.

    Inputs:
        - cfg: resolved config in solve mode [dict]
        - thetas: the sampler coordinates of the points (n, ndim) [array]
        - blobs: their blobs (n,) [structured array]

    Output: their P*, NaN where not solved or from the plateau branch [array]
    '''
    st = settings(cfg)
    values = np.array(blobs['solve_value'], dtype = float)
    u = np.asarray(thetas, dtype = float).reshape(len(values), -1)[:, st['index']]
    above, below = (likelihood.sigma_eff(st['ocfg'], above = side) for side in (True, False))
    with np.errstate(invalid = 'ignore'): # NaN where undefined: not the branch's
        g = search.g_of(blobs, st['observable'], st['ocfg'], 0.0) - u # T - u
        branch = (np.asarray(blobs['exhausted']) == 1) & (np.abs(g) > st['tol']*np.where(g > 0, above, below))
    return np.where(branch, nan, values)


def build_table(cfg, run_dir, chain = None, blobs = None):
    '''
    Fits the warm-start table to the initialization's solves plus the chain's solved states (a refit)
    Takes the draws of <run_dir>/solve_init.npz and the distinct solved states of chain (status 0, finite
    solve_value and log_jac, not from the plateau branch: table_values), in the order init, then by step and
    walker, and keeps the last MAX_TABLE_POINTS. A walker that stays put repeats its state, and only its first
    appearance counts. Deterministic: the same inputs give the same table. Without chain: the table of the
    initialization (init_solve's). Without solve_init.npz (an initialization other than init = "solve"): the chain
    alone.

    Inputs:
        - cfg: resolved config in solve mode [dict]
        - run_dir: the run directory [str]
        - chain: the chain so far, sampler coordinates (steps, walkers, ndim); None: none [array]
        - blobs: its blobs (steps, walkers) [structured array]

    Output: the table (fit_table) [dict]
    '''
    path = os.path.join(run_dir, 'solve_init.npz')
    ndim = len(prm.sampled_cfg(cfg)['free'])
    if os.path.exists(path): # init = "solve": every draw, solved or not (fit_table leaves out the unsolved)
        with np.load(path, allow_pickle = False) as d:
            thetas, b = d['theta'], d['blobs']
            values, log_jac = table_values(cfg, thetas, b), b['log_jac']
    else: # an initialization other than init = "solve": the chain alone
        thetas, values, log_jac = np.empty((0, ndim)), np.empty(0), np.empty(0)
    if chain is not None and len(chain):
        X = np.asarray(chain, dtype = float).reshape(-1, ndim) # by step, then walker
        B = np.asarray(blobs).reshape(-1)
        ok = (B['status'] == model.OK) & np.isfinite(table_values(cfg, X, B)) & np.isfinite(B['log_jac']) # solved
        thetas = np.concatenate([thetas, X[ok]])
        values = np.concatenate([values, B['solve_value'][ok]])
        log_jac = np.concatenate([log_jac, B['log_jac'][ok]])
        # the first appearance of each distinct state, in order of appearance; the last MAX_TABLE_POINTS of them
        _, first = np.unique(thetas, axis = 0, return_index = True)
        keep = np.sort(first)[-MAX_TABLE_POINTS:]
        thetas, values, log_jac = thetas[keep], values[keep], log_jac[keep]
    return fit_table(cfg, thetas, values, log_jac)


###_____Model runs per evaluation_____###

def evals_stats(counts):
    '''
    Summarizes the model runs per evaluation from a histogram
    The statistics are over the evaluations that ran the model; those that did not (e.g. outside the prior) are
    counted apart.

    Inputs:
        - counts: the histogram, counts[k] = evaluations with k model runs [array]

    Output: dict
        - mean, p10, p50, p90, max: model runs per evaluation; NaN without any [float]
        - n: evaluations that ran the model [int]
        - none: evaluations without a model run [int]
    '''
    counts = np.asarray(counts, dtype = float)
    none = int(counts[0]) if counts.size else 0 # evaluations without a model run
    counts = counts.copy()
    if counts.size:
        counts[0] = 0 # the statistics are over the evaluations that ran the model
    n = counts.sum()
    if not n:
        return {'mean': nan, 'p10': nan, 'p50': nan, 'p90': nan, 'max': nan, 'n': 0, 'none': none}
    k = np.arange(counts.size) # model runs
    cum = np.cumsum(counts)/n # the cumulative fraction of evaluations
    q = {f'p{p}': float(k[np.searchsorted(cum, p/100)]) for p in (10, 50, 90)} # the first k that reaches p %
    return dict(mean = float(np.sum(k*counts)/n), max = float(k[counts > 0].max()), n = int(n), none = none, **q)
