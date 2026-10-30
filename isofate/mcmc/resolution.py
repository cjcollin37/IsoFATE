'''
Resolution study: the cheapest n_steps and n_atmodeller whose observables are converged, run before an MCMC.

A few test points theta are run at every cell (n_steps, n_atmodeller) of a grid and compared with the reference
cell, the finest one computed (the most steps; among those, the most frequent Atmodeller calls). The error of a
cell is measured in units of the measurement error, in the space where the likelihood compares each active,
non-input observable k:

    d_k = |T_k(model at the cell) - T_k(model at the reference)| / sigma_k

T_k is linear, or log10 with the observable's floor; sigma_k is the effective sigma (the mean of the two sides
when asymmetric). d_k is 0 when the observable is undefined (NaN) at both resolutions and inf when it is defined
at only one of them, e.g. when the atmosphere is lost at one resolution only.

    PASS    max d_k <= tol and |delta log L| <= dlogl_tol at every test point
    STABLE  passes, and so does every computed finer cell (as many or more steps, and Atmodeller called at
            least as often)

The recommendation is the STABLE cell with the smallest measured time per evaluation. The noise floor repeats
every evaluation at theta + perturb*(prior width) in the parameter p_j of the equivalent shift (below) and
subtracts the change a smooth model has there, dT_k/dp_j x the step (from the derivative runs): what is left over
is numerical noise, and a model whose noise exceeds tol gives the MCMC a rough likelihood at any resolution.

Solve mode (--solve, the default when [solve] is active). The test points are points of the sampler's
coordinates (u, phi) and every cell solves for the solved parameter P again (solve.py, from both prior bounds),
so d of the solve observable is about 0 by construction and the other observables show the error that matters
for the posterior: at fixed (u, phi), as the sampler sees them. |delta log L| then compares the sampler's density
(log_prior + log_jac + log_like). The noise floor moves u by perturb of its range (raw differences: no
observable is steep in u). The test points come from init-style draws (source = "solve": the other parameters
from the prior, u from the data distribution, the better half by lp). A solve costs several model runs, which the
cost estimate counts (measured on the test-point selection). The equivalent-shift block stays as a cross-check:
the derivative runs are made at (P* + h, phi) of the reference cell, and the shift of P* itself between a cell and
the reference (solve_shift) is reported.

Equivalent parameter shift. Each test point is also run at the reference cell at theta + h e_j, h = 1e-4 of the
prior width W_j of p_j (sampler.bisect_param if free, else the first free parameter), which gives dT_k/dp_j.
A cell's error is then also expressed as the shift of p_j that would cause it:

    shift_k = (T_k(cell) - T_k(reference)) / (dT_k/dp_j)

On a thin ridge (a steep observable, e.g. the outflow H fraction near the LHS 1140 b island, where
d log10 x/d log_f_atm is about 190) a cell can fail by more than a sigma at fixed theta although its error equals
a shift of p_j far below the posterior width. The second recommendation is the cheapest cell that is stable when
an observable passes at a point with d_k <= tol OR |shift_k| <= shift_tol W_j, and the ridge weights stay put: at
each point the cell's values are moved back along p_j until the steepest observable kb matches the reference
again (linearized), and the other terms' change there, ridge_dlogl, may differ between the test points by at most
dlogl_tol (that spread is what reweights the posterior along the ridge). For an observable that is not steep
(|dT_k/dp_j| shift_tol W_j < tol sigma_k) the shift condition implies d_k <= tol, so those still pass at fixed
theta. The worst shift is reported over the (point, observable) pairs that need it (d_k > tol). The first
recommendation stays the conservative one.

Files in <output_root>/<name>/resolution_<stamp>/:
    config.json          the resolved config, with [resolution] as used
    points.npz           theta and theta_perturbed (points, ndim), the free names, the selection candidates
                         (with source = "bisect" also the bisection arrays, bisect_*)
    results.npz          blobs (points, cells, 2) as model.blob_dtype() ([..., 1] is the perturbed run), lp, the
                         cells, d and noise (points, cells, observables), dlogl, rel_diff, and per cell worst_d,
                         worst_dlogl, worst_noise, mean_runtime [s], mean_runs (model runs per evaluation),
                         passes, stable, verdict; the equivalent shift: shift_blobs (points,), deriv (points,
                         observables), shift and shift_frac (points, cells, observables), worst_shift,
                         ridge_dlogl (points, cells), ridge_spread, passes_shift, stable_shift, recommended_shift;
                         solve mode: solve (true), solve_shift (points, cells), worst_solve_shift
    table.txt            the summary for humans (also printed)
    recommendation.json  the recommended cell, the command line options to use it, and every cell's verdict
    resolution_heatmap.png, resolution_convergence.png

The parent process never imports the coupled model; the evaluations run in WorkerPools (model.eval_task).
'''

import copy
import json
import math
import os
import shlex
import time

import numpy as np

from isofate.mcmc import config, likelihood, model, search, solve
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool, choose_workers

inf, nan = math.inf, math.nan

QUICK_GRID = {'n_steps': [10000, 30000, 100000, 300000], 'n_atmodeller': [1000, 100, 10]}
SHIFT_STEP = 1e-4 # of the prior width: the step of the derivative for the equivalent shift
FIDUCIAL = (100000, 100) # (n_steps, n_atmodeller) of the previous runs
SMALL_TIER = 300000 # cells up to this n_steps share the first pool; larger ones may get fewer workers (memory)
RAN = (model.OK, model.NO_LIKELIHOOD) # statuses of evaluations whose model ran to the end
VERDICT_MARK = {'reference': 'R', 'stable': 'S', 'pass': 'p', 'fail': 'x', 'skipped': '-'}
WIDTH = 110 # of table.txt


###_____Grid and cost_____###

def cost_model(n_steps, n_atmodeller):
    '''
    Measured isocalc time per evaluation [s] (report_parallel.md section 6, within about 10%): 120 us per
    Atmodeller call plus 0.2 us per step. Runs that lose their atmosphere early cost less.
    '''
    if n_atmodeller == 0:
        return 2e-7*n_steps
    return 1.2e-4*(n_steps/n_atmodeller + 1) + 2e-7*n_steps


def est_eval_seconds(n_steps, n_atmodeller):
    '''
    The cost model times 2 for safety [s]; cells above resolution.max_eval_seconds are skipped. The toy backend
    uses the same estimate.
    '''
    return 2*cost_model(n_steps, n_atmodeller)


def fmt_n(n):
    '''
    Compact grid value: 10000 -> 1e4, 300000 -> 3e5, 2000 -> 2000.
    '''
    n = int(n)
    if n >= 10000:
        e = int(math.floor(math.log10(n)))
        k = n/10**e
        if k == int(k):
            return f'{int(k)}e{e}'
    return str(n)


def grid_values(cfg, grid = 'default'):
    '''
    (n_steps ascending, n_atmodeller descending) of the study. grid: 'default' (cfg['resolution']), 'quick', or a
    dict with n_steps and/or n_atmodeller lists (the missing one from cfg['resolution']). An uncoupled config
    (n_atmodeller = 0) uses n_atmodeller = [0], unless the dict gives a list.
    '''
    res = cfg['resolution']
    explicit_atm = isinstance(grid, dict) and 'n_atmodeller' in grid
    if isinstance(grid, dict):
        config._check_keys(grid, ('n_steps', 'n_atmodeller'), 'the resolution grid')
        n_steps, n_atm = grid.get('n_steps', res['n_steps']), grid.get('n_atmodeller', res['n_atmodeller'])
    elif grid == 'quick':
        n_steps, n_atm = QUICK_GRID['n_steps'], QUICK_GRID['n_atmodeller']
    elif grid == 'default':
        n_steps, n_atm = res['n_steps'], res['n_atmodeller']
    else:
        raise config.ConfigError(f"grid must be 'default', 'quick' or a dict of lists, got {grid!r}")
    if cfg['model']['n_atmodeller'] == 0 and not explicit_atm:
        n_atm = [0]
    if not len(n_steps) or not len(n_atm):
        raise config.ConfigError('the resolution grid needs at least one n_steps and one n_atmodeller value')
    n_steps = sorted({config._int(n, 'resolution n_steps', minimum = 1) for n in n_steps})
    n_atm = sorted({config._int(n, 'resolution n_atmodeller', minimum = 0) for n in n_atm}, reverse = True)
    if n_steps[-1] > config.MAX_N_STEPS and not cfg['model']['allow_large_n_steps']:
        raise config.ConfigError(f'resolution n_steps = {n_steps[-1]:.3g} is above {config.MAX_N_STEPS:.0e}; set '
                                 'model.allow_large_n_steps = true if you really mean it')
    return n_steps, n_atm


def plan_cells(cfg, n_steps, n_atmodeller):
    '''
    The grid cells, n_steps by n_steps: dicts {n_steps, n_atmodeller, est_s (estimated seconds per evaluation),
    skip (None, or why the cell is not computed)}.
    '''
    max_s = cfg['resolution']['max_eval_seconds']
    cells = []
    for ns in n_steps:
        for na in n_atmodeller:
            est = est_eval_seconds(ns, na)
            skip = None
            if na > ns:
                skip = 'n_atmodeller > n_steps'
            elif est > max_s:
                skip = f'estimated {est:.0f} s per evaluation > max_eval_seconds = {max_s:g} s'
            cells.append({'n_steps': ns, 'n_atmodeller': na, 'est_s': est, 'skip': skip})
    return cells


def reference_index(cells):
    '''
    Index of the reference cell: the computed cell with the most steps and, among those, the smallest
    n_atmodeller; uncoupled cells (n_atmodeller = 0) only when they are all there is. None if nothing is computed.
    '''
    computed = [i for i, cell in enumerate(cells) if cell['skip'] is None]
    coupled = [i for i in computed if cells[i]['n_atmodeller'] > 0] or computed
    if not coupled:
        return None
    top = max(cells[i]['n_steps'] for i in coupled)
    return min((i for i in coupled if cells[i]['n_steps'] == top), key = lambda i: cells[i]['n_atmodeller'])


def at_least_as_fine(a, b):
    '''
    True when cell b resolves at least as finely as cell a: as many or more steps and Atmodeller called at least
    as often. Uncoupled cells (n_atmodeller = 0) only compare with each other.
    '''
    if (a['n_atmodeller'] == 0) != (b['n_atmodeller'] == 0):
        return False
    return b['n_steps'] >= a['n_steps'] and b['n_atmodeller'] <= a['n_atmodeller']


def cell_name(cell):
    return f"n_steps = {fmt_n(cell['n_steps'])}, n_atmodeller = {cell['n_atmodeller']}"


def _cell_index(cells, n_steps, n_atmodeller):
    for c, cell in enumerate(cells):
        if cell['n_steps'] == n_steps and cell['n_atmodeller'] == n_atmodeller:
            return c
    return None


###_____Test points_____###

def farthest_points(u, n):
    '''
    Greedy farthest-point sampling: indices of n rows of u (points in the unit cube), starting from row 0.
    '''
    u = np.asarray(u, dtype = float)
    chosen = [0]
    dist = np.linalg.norm(u - u[0], axis = 1)
    dist[0] = -1.0
    while len(chosen) < min(n, len(u)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(u - u[i], axis = 1))
        dist[chosen] = -1.0
    return chosen


def _evaluate_at_config(pool, thetas, solving = False):
    '''
    eval_task results at the config resolution, in order (solving: sampler thetas, solved).
    '''
    return pool.map(model.eval_task, [{'theta': theta, 'id': i, 'solve': solving} for i, theta in enumerate(thetas)])


def _status_counts(status):
    values, counts = np.unique(np.asarray(status)[np.isfinite(status)], return_counts = True)
    return ', '.join(f'{n} {model.STATUS_NAMES[int(v)]}' for v, n in zip(values, counts))


def _prescan_points(cfg, pool, n, rng, warnings, solving = False):
    '''
    20n Latin-hypercube prior draws evaluated at the config resolution; of the best quarter by log-probability
    (at least n), n spread out by farthest-point sampling in the unit cube, starting from the best. solving: in the
    sampler's coordinates (u uniform on its support), solved.
    '''
    space = prm.sampled_cfg(cfg) if solving else cfg
    candidates = prm.sample_prior(space, 20*n, rng)
    results = _evaluate_at_config(pool, candidates, solving)
    lp = np.array([r['lp'] for r in results])
    status = np.array([r['status'] for r in results])
    finite = np.flatnonzero(np.isfinite(lp))
    if finite.size == 0:
        raise RuntimeError(f'none of the {len(candidates)} prescan points has a finite log-probability at the '
                           f'config resolution ({_status_counts(status)}). Check the observables with `python -m '
                           'isofate.mcmc check CONFIG`, or use source = "prior"')
    if finite.size < n:
        warnings.append(f'only {finite.size} of {len(candidates)} prescan points have a finite log-probability; '
                        f'using {finite.size} test points instead of {n}')
    order = finite[np.argsort(-lp[finite], kind = 'stable')]
    top = order[:max(n, math.ceil(0.25*finite.size))]
    chosen = top[farthest_points(prm.unit_cube(candidates[top], space), n)]
    return candidates[chosen], {'candidates': candidates, 'candidates_lp': lp, 'candidates_status': status,
                                'candidates_runs': np.array([r['solve_evals'] for r in results], dtype = float)}


def _prior_points(cfg, pool, n, rng, warnings, solving = False):
    '''
    Random prior draws whose atmosphere survives at the config resolution, from 4n candidates (stripped ones
    fill up if there are not enough). solving: in the sampler's coordinates, solved.
    '''
    candidates = prm.sample_prior(prm.sampled_cfg(cfg) if solving else cfg, 4*n, rng, lhs = False)
    results = _evaluate_at_config(pool, candidates, solving)
    lp = np.array([r['lp'] for r in results])
    status = np.array([r['status'] for r in results])
    alive = np.array([r['obs']['exhausted'] == 0 for r in results])
    ran = np.isin(status, RAN)
    ranked = np.concatenate([np.flatnonzero(ran & alive & (status == model.OK)),
                             np.flatnonzero(ran & alive & (status != model.OK)), np.flatnonzero(ran & ~alive)])
    if ranked.size == 0:
        raise RuntimeError(f'the model failed at all {len(candidates)} prior draws ({_status_counts(status)})')
    chosen = ranked[:n]
    if chosen.size < n:
        warnings.append(f'only {chosen.size} of {len(candidates)} prior draws ran; using {chosen.size} test points')
    n_stripped = int((~alive[chosen]).sum())
    if n_stripped:
        warnings.append(f'{n_stripped} test point(s) lose their atmosphere at the config resolution (too few prior '
                        'draws keep one)')
    return candidates[chosen], {'candidates': candidates, 'candidates_lp': lp, 'candidates_status': status,
                                'candidates_runs': np.array([r['solve_evals'] for r in results], dtype = float)}


def _solve_points(cfg, pool, n, rng, warnings, solving = True):
    '''
    Solve mode: init-style draws (params.sample_solve: the other parameters from the prior, u from the data
    distribution), solved at the config resolution, in batches of bisect_draws(n) until n have a finite lp (at
    most search.BATCHES); of the better half by lp (at least n), n spread out by farthest-point sampling in the
    unit cube of the sampler's coordinates, starting from the best.
    '''
    thetas, results = [], []
    for _ in range(search.BATCHES):
        draws = prm.sample_solve(cfg, bisect_draws(n), rng)
        thetas.append(draws)
        results += _evaluate_at_config(pool, draws, True)
        if sum(math.isfinite(r['lp']) for r in results) >= n:
            break
    candidates = np.concatenate(thetas)
    lp = np.array([r['lp'] for r in results])
    status = np.array([r['status'] for r in results])
    runs = np.array([r['solve_evals'] for r in results], dtype = float)
    finite = np.flatnonzero(np.isfinite(lp))
    print(f'[resolution] solve-mode draws: {len(candidates)}, {finite.size} with a finite lp '
          f'({_status_counts(status)}); {np.mean(runs):.3g} model runs per solve', flush = True)
    if finite.size == 0:
        raise RuntimeError(f'none of the {len(candidates)} solve-mode draws has a finite log-probability at the config '
                           f'resolution ({_status_counts(status)}). Check the solve with `python -m isofate.mcmc check '
                           'CONFIG`')
    if finite.size < n:
        warnings.append(f'only {finite.size} of {len(candidates)} solve-mode draws have a finite log-probability; '
                        f'using {finite.size} test points instead of {n}')
    order = finite[np.argsort(-lp[finite], kind = 'stable')]
    top = order[:max(n, math.ceil(0.5*finite.size))]
    chosen = top[farthest_points(prm.unit_cube(candidates[top], prm.sampled_cfg(cfg)), n)]
    return candidates[chosen], {'candidates': candidates, 'candidates_lp': lp, 'candidates_status': status,
                                'candidates_runs': runs}


def bisect_draws(points):
    '''
    Draws per batch of the bisection source: more than the points, so farthest-point sampling has a choice.
    '''
    return max(20, 5*points)


def _bisect_points(cfg, pool, n, rng, warnings, solving = False):
    '''
    Points where sampler.bisect_observable fits its data (search.bisect_points at the config resolution): of the
    better half of the converged ones by log-probability (at least n), n spread out by farthest-point sampling in
    the unit cube, starting from the best lp. The half: a converged draw can still lie far off the posterior (on
    LHS 1140 b one matched the outflow H fraction at the stripping edge, Mdot_HHe 5e-8 kg/s and lp -1859, where
    the outflow is numerically ill-defined), and the walkers start from the best points too. Too few converged:
    a warning; none: prescan points instead when sampler.bisect_fallback, else an error.
    '''
    result = search.bisect_points(cfg, pool, n, rng, log = lambda text: print(f'[resolution] {text}', flush = True),
                                  draws = bisect_draws(n))
    ok = search.good(result)
    if ok.size == 0:
        text = search.failure_text(result, n)
        if not cfg['sampler']['bisect_fallback']:
            raise RuntimeError(text)
        warnings.append(f'the bisection found no test point, so these are prescan points instead: {text}')
        thetas, selection = _prescan_points(cfg, pool, n, rng, warnings)
        return thetas, dict(selection, fallback = True)
    if ok.size < n:
        warnings.append(f'only {ok.size} bisection draws converged; using {ok.size} test points instead of {n}')
    order = ok[np.argsort(-result['lp'][ok], kind = 'stable')]
    top = order[:max(n, math.ceil(0.5*ok.size))]
    chosen = top[farthest_points(prm.unit_cube(result['point'][top], cfg), n)]
    selection = {'candidates': result['point'][ok], 'candidates_lp': result['lp'][ok],
                 'candidates_status': result['blobs']['status'][ok], 'bisect_draws': len(result['outcome']),
                 'bisect_converged': ok.size, 'bisect_evaluations': result['evaluations'],
                 'param': result['settings']['param'], 'observable': result['settings']['observable']}
    selection.update({f'bisect_{key}': result[key] for key in ('theta', 'point', 'x', 'g', 'outcome', 'lp')})
    return result['point'][chosen], selection


def shift_param_index(cfg, solving = False):
    '''
    Index in the physical theta of the parameter of the equivalent shift: solve.param in solve mode, else
    sampler.bisect_param if free, else the first.
    '''
    if solving:
        return cfg['solve']['index']
    param = cfg['sampler']['bisect_param']
    return cfg['free'].index(param) if param in cfg['free'] else 0


def shifted(theta, cfg, j, step = SHIFT_STEP):
    '''
    (theta + h e_j, h) with h = step*(prior width of p_j), or -h at the upper edge of the prior.
    '''
    lo, hi = prm.prior_box(cfg)
    h = step*(hi[j] - lo[j])
    out = np.array(theta, dtype = float)
    out[j] += h
    if not prm.in_support(cfg['free'][j], out[j], cfg):
        out[j], h = theta[j] - h, -h
    return out, h


def read_discard(run_dir, n_steps):
    '''
    The burn-in of an MCMC run [steps]: summary.json's discard, else half the chain.
    '''
    path = os.path.join(run_dir, 'summary.json')
    if os.path.exists(path):
        with open(path) as f:
            discard = json.load(f).get('discard')
        if discard is not None:
            return min(int(discard), n_steps - 1)
    return n_steps//2


def _chain_points(run_dir, cfg, n, rng, warnings, solving = False):
    '''
    n distinct random posterior samples (after the burn-in) of an MCMC run directory, as physical thetas, or
    (solving) in the sampler's coordinates. A solve-mode run's physical point takes P* from its blobs; a run
    without solve mode gives u = T(observable) of its blobs (samples outside the u support are left out).
    '''
    path = os.path.join(run_dir, 'chain.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'source = "chain:{run_dir}": there is no chain.npz in it')
    run_cfg = os.path.join(run_dir, 'config.json')
    run_solved = False
    if os.path.exists(run_cfg):
        rcfg = config.load_resolved(run_cfg)
        if list(rcfg['free']) != list(cfg['free']):
            raise config.ConfigError(f'the run in {run_dir} has free parameters {rcfg["free"]}, this config '
                                     f'{cfg["free"]}')
        run_solved = config.solve_on(rcfg)
    with np.load(path) as data:
        chain, log_prob = data['chain'], data['log_prob']
        blobs = data['blobs'] if 'blobs' in data.files else None
    discard = read_discard(run_dir, chain.shape[0])
    keep = np.isfinite(log_prob[discard:].reshape(-1))
    samples = chain[discard:].reshape(-1, chain.shape[-1])[keep]
    sample_blobs = blobs[discard:].reshape(-1)[keep] if blobs is not None else None
    if run_solved != solving:
        if sample_blobs is None:
            raise RuntimeError(f'{path} has no blobs to convert its samples between solve mode and the physical '
                               'parameters')
        if run_solved: # to physical
            samples = solve.physical(samples, sample_blobs['solve_value'], cfg)
        else: # to the sampler's coordinates
            samples = solve.to_sampled(samples, sample_blobs, cfg)
            lo, hi = cfg['solve']['bounds']
            u = samples[:, cfg['solve']['index']]
            samples = samples[np.isfinite(u) & (u >= lo) & (u <= hi)]
    samples = np.unique(samples[np.all(np.isfinite(samples), axis = 1)], axis = 0) # walkers repeat positions
    if len(samples) == 0:
        raise RuntimeError(f'no finite posterior samples in {path}' + (' inside the u support' if solving else ''))
    if len(samples) < n:
        warnings.append(f'the chain has only {len(samples)} distinct posterior samples; using them all')
    return samples[rng.choice(len(samples), size = min(n, len(samples)), replace = False)], {}


def perturbed(theta, cfg, perturb, j = 0):
    '''
    theta moved by perturb*(prior width) in coordinate j (backwards at the upper edge of the prior). cfg: the
    config of theta's coordinates (params.sampled_cfg for the sampler's).
    '''
    lo, hi = prm.prior_box(cfg)
    step = perturb*(hi[j] - lo[j])
    out = np.array(theta, dtype = float)
    out[j] += step
    if not prm.in_support(cfg['free'][j], out[j], cfg):
        out[j] = theta[j] - step
    return out


def _plural(n, word):
    return f"{n} {word}{'s' if n != 1 else ''}"


def _source_text(source, n, info):
    if source == 'prescan':
        return (f"{_plural(n, 'test point')} from a prescan of {len(info['candidates'])} Latin-hypercube prior "
                'draws at the config resolution (the best quarter by log-probability, spread out by '
                'farthest-point sampling)')
    if source == 'prior':
        return (f"{_plural(n, 'random prior draw')} that {'keeps' if n == 1 else 'keep'} an atmosphere at the "
                'config resolution')
    if source == 'thetas':
        return _plural(n, 'given test point')
    if source == 'solve':
        return (f"{_plural(n, 'test point')} of the sampler's coordinates from init-style draws "
                f"({len(info['candidates'])} draws, the other parameters from the prior and u from the data "
                'distribution, solved at the config resolution; the better half by log-probability, spread out by '
                'farthest-point sampling)')
    if source == 'bisect' and not info.get('fallback'):
        return (f"{_plural(n, 'test point')} where the bisection on {info.get('param', '?')} brought "
                f"{info.get('observable', '?')} to its data ({info['bisect_converged']} of {info['bisect_draws']} "
                'draws converged; the better half by log-probability, spread out by farthest-point sampling)')
    if source == 'bisect':
        return f"{_plural(n, 'test point')} from a prescan (the bisection found none; see the warnings)"
    return f"{_plural(n, 'random posterior sample')} of {source[len('chain:'):]}"


###_____Metrics_____###

def _space(values, ocfg):
    '''
    likelihood.to_space over an array.
    '''
    flat = [likelihood.to_space(x, ocfg) for x in np.ravel(values)]
    return np.array(flat, dtype = float).reshape(np.shape(values))


def distance(a, b, s = 1.0):
    '''
    |a - b|/s element-wise, with 0 where a and b are both NaN (or equal infinities) and inf where only one of
    them is finite.
    '''
    a, b = np.broadcast_arrays(np.asarray(a, dtype = float), np.asarray(b, dtype = float))
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid = 'ignore'):
        d = np.abs(a - b)/s
    d = np.where(same, 0.0, d)
    return np.where(~same & ~(np.isfinite(a) & np.isfinite(b)), inf, d)


def relative_change(x, ref):
    '''
    |x - ref|/|ref| element-wise: NaN where both are NaN (undefined), 0 where equal, inf where only one is
    finite or ref = 0.
    '''
    x, ref = np.broadcast_arrays(np.asarray(x, dtype = float), np.asarray(ref, dtype = float))
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        r = np.abs(x - ref)/np.abs(ref)
    r = np.where(np.isfinite(x) != np.isfinite(ref), inf, r)
    r = np.where(x == ref, 0.0, r)
    return np.where(np.isnan(x) & np.isnan(ref), nan, r)


def _stable(passes, cells, computed):
    '''
    Cells that pass and whose every computed finer cell passes too.
    '''
    return np.array([bool(passes[c]) and all(passes[f] for f in np.flatnonzero(computed)
                                             if at_least_as_fine(cells[c], cells[f])) for c in range(len(cells))])


def equivalent_shift(T, T_shift, ref, h):
    '''
    (deriv (P, K), shift (P, C, K)): dT/dp_j = (T_shift - T_ref)/h at the reference cell, and the shift of p_j
    equivalent to each cell's error, (T(cell) - T(ref))/deriv. T: (P, C, K) in the likelihood space, T_shift:
    (P, K), h: (P,). shift is 0 where the cell equals the reference (both undefined included) and inf where the
    error has no equivalent shift (defined at one resolution only, or a zero or undefined derivative).
    '''
    T_ref = T[:, ref, :]
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        deriv = (T_shift - T_ref)/np.asarray(h, dtype = float)[:, None]
        dT = T - T_ref[:, None, :]
        shift = dT/deriv[:, None, :]
    same = (T == T_ref[:, None, :]) | (np.isnan(T) & np.isnan(T_ref[:, None, :]))
    shift = np.where(same, 0.0, shift)
    shift = np.where(~same & ~np.isfinite(shift), inf, shift)
    return deriv, shift


def _term_in_space(T, ocfg):
    '''
    likelihood.term of a model value given in the likelihood space T (log10 for a log10-space observable).
    '''
    if ocfg['space'] == 'log10':
        with np.errstate(over = 'ignore'):
            return likelihood.term(10.0**T if math.isfinite(T) else (0.0 if T < 0 else T), ocfg)
    return likelihood.term(T, ocfg)


def analyse(cfg, cells, blobs, ref, tol, dlogl_tol, shift_blobs = None, shift_h = None, shift_j = 0,
            shift_tol = None, solving = False, noise_step = None):
    '''
    Metrics and verdicts of a finished study (no model runs). blobs: (points, cells, 1 or 2) array of
    model.blob_dtype(), NaN where not computed; [..., 1] is the perturbed run. Test points whose reference run
    failed are left out (used = False). Returns a dict of arrays (see the module docstring) plus 'verdict' per
    cell ('reference', 'stable', 'pass', 'fail', 'skipped') and 'recommended' (a cell index, or None).
    shift_blobs (points,): the runs at theta + shift_h e_shift_j at the reference cell (solve mode: at (P* + h, phi)
    of the reference cell); with them also the equivalent shift (module docstring): deriv, shift, shift_frac, steep,
    worst_shift (per cell, in units of p_j), ridge_dlogl, ridge_spread, passes_shift, stable_shift, verdict_shift
    and recommended_shift. noise_step (points,): the step of the perturbed runs along p_j; with the derivative the
    noise is what is left after the smooth change deriv*noise_step (solving: no subtraction). solving: the blobs
    are solve-mode evaluations; |delta log L| then compares log_prior + log_jac + log_like, and solve_shift is the
    change of the solved parameter against the reference.
    '''
    n_points, n_cells, n_k = blobs.shape
    names = [name for name, o in cfg['observables'].items() if not o['input']]
    computed = np.array([cell['skip'] is None for cell in cells])
    used = np.isin(blobs['status'][:, ref, 0], RAN)

    T = np.stack([_space(blobs[name], cfg['observables'][name]) for name in names], axis = -1) # (P, C, K, J)
    s = np.array([likelihood.sigma_eff(cfg['observables'][name]) for name in names])
    d = distance(T[:, :, 0], T[:, ref:ref + 1, 0], s)
    ll = blobs['log_like'].copy()
    if solving and 'log_jac' in blobs.dtype.names: # the sampler's density at fixed (u, phi)
        ll = ll + blobs['log_jac'] + blobs['log_prior']
    ll = np.where(np.isnan(ll), -inf, ll) # no likelihood: -inf
    dlogl = distance(ll[:, :, 0], ll[:, ref:ref + 1, 0])

    has_shift = shift_blobs is not None and shift_tol is not None
    deriv = deriv_ll = None
    if has_shift:
        T_shift = np.stack([_space(shift_blobs[name], cfg['observables'][name]) for name in names], axis = -1)
        deriv = equivalent_shift(T[:, :, 0], T_shift, ref, shift_h)[0]
        ll_shift = np.where(np.isnan(shift_blobs['log_like']), -inf, shift_blobs['log_like'])
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            deriv_ll = (ll_shift - blobs['log_like'][:, ref, 0])/np.asarray(shift_h, dtype = float)
    if n_k > 1:
        T1, ll1 = T[:, :, 1], ll[:, :, 1]
        if deriv is not None and noise_step is not None and not solving:
            # remove the change a smooth model has over the step (finite derivative only)
            step = np.asarray(noise_step, dtype = float)
            smooth = np.where(np.isfinite(deriv), deriv*step[:, None], 0.0)
            smooth_ll = np.where(np.isfinite(deriv_ll), deriv_ll*step, 0.0)
            with np.errstate(invalid = 'ignore'):
                T1 = np.where(np.isfinite(T1), T1 - smooth[:, None, :], T1)
                ll1 = np.where(np.isfinite(ll1), ll1 - smooth_ll[:, None], ll1)
        noise = distance(T1, T[:, :, 0], s)
        dlogl_noise = distance(ll1, ll[:, :, 0])
    else:
        noise, dlogl_noise = np.full(d.shape, nan), np.full(dlogl.shape, nan)
    rel_fields = [f for f in model.OBS_FIELDS if not config.OBSERVABLES.get(f, {}).get('input')]
    rel_diff = np.stack([relative_change(blobs[f][:, :, 0], blobs[f][:, ref:ref + 1, 0]) for f in rel_fields],
                        axis = -1)
    for a in (d, noise, dlogl, dlogl_noise, rel_diff):
        a[:, ~computed] = nan

    out = {'obs_names': names, 'computed': computed, 'used': used, 'd': d, 'noise': noise, 'dlogl': dlogl,
           'dlogl_noise': dlogl_noise, 'changed': np.isinf(d), 'rel_fields': rel_fields, 'rel_diff': rel_diff,
           'solve': bool(solving)}
    for key in ('worst_d', 'worst_dlogl', 'worst_noise', 'worst_dlogl_noise', 'mean_runtime', 'n_failed', 'mean_runs'):
        out[key] = np.full(n_cells, nan)
    for key in ('worst_obs', 'worst_point', 'noise_obs', 'noise_point'):
        out[key] = np.full(n_cells, -1)
    runs = blobs['solve_evals'] if 'solve_evals' in blobs.dtype.names else np.ones(blobs.shape)
    for c in np.flatnonzero(computed):
        out['mean_runtime'][c] = np.nanmean(blobs['runtime'][:, c])
        if np.isfinite(runs[:, c]).any():
            out['mean_runs'][c] = np.nanmean(runs[:, c])
        out['n_failed'][c] = np.sum(~np.isin(blobs['status'][:, c], RAN))
        if not used.any():
            continue
        p, j = np.unravel_index(np.argmax(np.where(used[:, None], d[:, c], -1.0)), d[:, c].shape)
        out['worst_d'][c], out['worst_obs'][c], out['worst_point'][c] = d[p, c, j], j, p
        out['worst_dlogl'][c] = np.max(dlogl[used, c])
        if n_k > 1:
            p, j = np.unravel_index(np.argmax(np.where(used[:, None], noise[:, c], -1.0)), noise[:, c].shape)
            out['worst_noise'][c], out['noise_obs'][c], out['noise_point'][c] = noise[p, c, j], j, p
            out['worst_dlogl_noise'][c] = np.max(dlogl_noise[used, c])
    if solving and 'solve_value' in blobs.dtype.names:
        sv = blobs['solve_value'][:, :, 0]
        out['solve_shift'] = sv - sv[:, ref:ref + 1]
        out['solve_shift'][:, ~computed] = nan
        worst = np.full(n_cells, nan)
        for c in np.flatnonzero(computed):
            values = np.abs(out['solve_shift'][used, c])
            if np.isfinite(values).any():
                worst[c] = np.nanmax(values)
        out['worst_solve_shift'] = worst

    passes = computed & (out['worst_d'] <= tol) & (out['worst_dlogl'] <= dlogl_tol)
    stable = _stable(passes, cells, computed)

    def verdicts(passes, stable):
        verdict = []
        for c in range(n_cells):
            if not computed[c]:
                verdict.append('skipped')
            elif c == ref:
                verdict.append('reference')
            else:
                verdict.append('stable' if stable[c] else 'pass' if passes[c] else 'fail')
        candidates = list(np.flatnonzero(stable))
        best = min(candidates, key = lambda c: (out['mean_runtime'][c], c)) if candidates else None
        return verdict, None if best is None else int(best)

    verdict, recommended = verdicts(passes, stable)
    out.update(passes = passes, stable = stable, verdict = verdict, recommended = recommended)

    out['has_shift'] = has_shift
    if has_shift:
        lo, hi = prm.prior_box(cfg)
        width = hi[shift_j] - lo[shift_j]
        deriv, shift = equivalent_shift(T[:, :, 0], T_shift, ref, shift_h)
        shift[:, ~computed] = nan
        shift_frac = np.abs(shift)/width
        # steep: a shift of shift_tol W moves T by more than tol sigma, so the shift condition is the weaker one
        steep = np.abs(deriv)*shift_tol*width/s > tol
        ok = (d <= tol) | (shift_frac <= shift_tol)
        # the ridge weights: the cell's values moved back along p_j until the steepest observable kb matches the
        # reference (linearized; solve mode has done that by solving), and the change of the other terms there
        with np.errstate(invalid = 'ignore'):
            steepness = np.nanmedian(np.abs(deriv[used])*width/s, axis = 0) if used.any() else np.zeros(len(names))
        kb = int(np.nanargmax(np.where(np.isfinite(steepness), steepness, -1.0))) if len(names) else 0
        ridge = np.full((n_points, n_cells), nan)
        for p in range(n_points):
            for c in np.flatnonzero(computed):
                if solving:
                    ridge[p, c] = ll[p, c, 0] - ll[p, ref, 0]
                    continue
                total = 0.0
                for k, name in enumerate(names):
                    if k == kb:
                        continue
                    o = cfg['observables'][name]
                    with np.errstate(invalid = 'ignore'):
                        T_root = T[p, c, 0, k] - deriv[p, k]*shift[p, c, kb] if shift[p, c, kb] != 0 else T[p, c, 0, k]
                    total += _term_in_space(T_root, o) - _term_in_space(T[p, ref, 0, k], o)
                ridge[p, c] = total
        spread = np.full(n_cells, nan)
        for c in np.flatnonzero(computed):
            values = ridge[used, c]
            spread[c] = 0.0 if c == ref else (np.ptp(values) if values.size and np.isfinite(values).all() else inf)
        passes_shift = computed & np.array([bool(np.all(ok[used, c])) if used.any() else False
                                            for c in range(n_cells)]) & (spread <= dlogl_tol)
        stable_shift = _stable(passes_shift, cells, computed)
        worst = np.full(n_cells, nan)
        worst_obs, worst_point = np.full(n_cells, -1), np.full(n_cells, -1)
        for c in np.flatnonzero(computed):
            if used.any(): # over the (point, observable) pairs that need the shift criterion (d > tol)
                a = np.where(used[:, None] & (d[:, c] > tol), np.abs(shift[:, c]), -1.0)
                p, k = np.unravel_index(np.argmax(np.nan_to_num(a, nan = -1.0)), a.shape)
                worst[c], worst_obs[c], worst_point[c] = (a[p, k], k, p) if a[p, k] >= 0 else (0.0, -1, -1)
        verdict_shift, recommended_shift = verdicts(passes_shift, stable_shift)
        out.update(deriv = deriv, shift = shift, shift_frac = shift_frac, steep = steep, worst_shift = worst,
                   worst_shift_obs = worst_obs, worst_shift_point = worst_point, ridge_dlogl = ridge,
                   ridge_spread = spread, ridge_obs = kb, passes_shift = passes_shift,
                   stable_shift = stable_shift, verdict_shift = verdict_shift, recommended_shift = recommended_shift,
                   shift_width = width, shift_h = np.asarray(shift_h, dtype = float), shift_j = shift_j,
                   shift_tol = shift_tol)
    return out


def _warnings(study):
    '''
    What the user should know about the result: roughness, unconverged grids, failures, too few points.
    '''
    cfg, cells, m = study['cfg'], study['cells'], study['metrics']
    ref, rec, tol, dlogl_tol = study['reference'], m['recommended'], study['tol'], study['dlogl_tol']
    space = study.get('space', cfg)
    first, perturb = space['free'][study.get('noise_j', 0)], study['perturb']
    out = []
    if perturb > 0 and rec is not None and m['worst_noise'][rec] >= tol:
        out.append(f'likelihood is rough at the {perturb:g} level; MCMC will see numerical noise: at the recommended '
                   f'cell, moving {first} by {perturb:g} of its prior width changes {_worst_text(study, rec, "noise")} '
                   f"by {_fmt_d(m['worst_noise'][rec])} sigma (tol = {tol:g})")
    if perturb > 0:
        rough = [p for p in np.flatnonzero(m['used']) if np.max(m['noise'][p, ref]) >= tol
                 or m['dlogl_noise'][p, ref] >= dlogl_tol]
        if rough:
            out.append(f"convergence is ill-defined at point(s) {', '.join(map(str, rough))}: even at the reference "
                       f'cell a {perturb:g} perturbation of {first} changes an observable by >= tol (or log L by >= '
                       'dlogl_tol), so d there measures numerical noise rather than resolution error ('
                       + '; '.join(f"point {p}: {_fmt_theta(study['thetas'][p], space)}" for p in rough) + ')')
    if rec is not None and rec == ref:
        out.append('only the reference cell is stable, so this grid does not show convergence and the reference '
                   'may itself be unconverged: extend the grid (n_steps up to 1e6, n_atmodeller down to 1), use '
                   'more test points, or loosen tol')
    elif rec is not None:
        coupled = [cell for cell in cells if cell['n_atmodeller'] > 0] or cells
        if (cells[rec]['n_steps'] == min(cell['n_steps'] for cell in cells)
                or cells[rec]['n_atmodeller'] == max(cell['n_atmodeller'] for cell in coupled)):
            out.append('the recommended cell is at the coarse edge of the grid: a cheaper cell outside it may pass '
                       'too')
    if rec is not None and rec != ref:
        loose = []
        for i, field in enumerate(m['rel_fields']):
            change = m['rel_diff'][m['used'], rec, i]
            change = change[~np.isnan(change)]
            if field in config.OBSERVABLES and field not in cfg['observables'] and change.size and change.max() > 0.1:
                loose.append(f'{field} ({_fmt_d(change.max())})')
        if loose:
            out.append('outputs that are not fitted change by more than 10% between the recommended cell and the '
                       f"reference: {', '.join(loose)}. Fitting any of them later needs a new study with it switched "
                       'on')
    fid = _cell_index(cells, *FIDUCIAL)
    if fid is not None and m['verdict'][fid] in ('fail', 'pass'):
        out.append(f"the fiducial n_steps = 1e5, n_atmodeller = 100 is not stable (worst d = "
                   f"{_fmt_d(m['worst_d'][fid])}): runs at it carry numerical errors above tol")
    for p in np.flatnonzero(~m['used']):
        out.append(f"point {p} is left out: its run at the reference cell failed ({study['ref_errors'].get(p)})")
    for c, error in study['errors'].items():
        if m['n_failed'][c] > 0:
            out.append(f"{int(m['n_failed'][c])} run(s) failed at {cell_name(cells[c])}; first error: {error}")
    if study['n_unexpected']:
        out.append(f"{study['n_unexpected']} unexpected error(s) (status 5): tracebacks in "
                   f"{os.path.join(study['out_dir'], 'worker_errors')}")
    if len(study['thetas']) < 3:
        out.append(f"only {_plural(len(study['thetas']), 'test point')}: other parts of the parameter space are not "
                   'tested; use points >= 6 before trusting the recommendation')
    return study['selection_warnings'] + out + [f'config: {w}' for w in cfg['warnings']]


###_____Text output_____###

def _fmt_d(x):
    if np.isnan(x):
        return '?'
    if np.isinf(x):
        return 'inf'
    return '0' if x == 0 else f'{x:.2g}'


def fmt_seconds(t):
    '''
    Seconds per evaluation, compactly: 0.0043 s, 0.15 s, 12.2 s, 340 s.
    '''
    if not np.isfinite(t):
        return '?'
    if t < 1:
        return f'{t:.2g} s'
    return f'{t:.3g} s' if t < 100 else f'{t:.0f} s'


def fmt_duration(t):
    if t < 90:
        return f'{t:.0f} s'
    if t < 5400:
        return f'{t/60:.1f} min'
    return f'{t/3600:.1f} h'


def _fmt_theta(theta, cfg):
    parts = []
    for name, x in zip(cfg['free'], theta):
        unit = cfg['params'][name]['unit']
        parts.append(f"{name} = {x:.4g}{'' if unit in ('', 'dex') else ' ' + unit}")
    return ', '.join(parts)


def _wrap(text, first, indent = None):
    '''
    text word-wrapped to WIDTH: the first line starts with first, the others with indent (default: as wide).
    '''
    indent = ' '*len(first) if indent is None else indent
    lines, line = [], first
    for word in text.split():
        if len(line) + len(word) + 1 > WIDTH and line.strip():
            lines.append(line.rstrip())
            line = indent
        line += word + ' '
    lines.append(line.rstrip())
    return lines


def _worst_text(study, c, kind = 'd'):
    m = study['metrics']
    j, p = (m['worst_obs'][c], m['worst_point'][c]) if kind == 'd' else (m['noise_obs'][c], m['noise_point'][c])
    return f"{m['obs_names'][j]} at point {p}" if j >= 0 else 'nothing'


def options_text(cell):
    return f"--set model.n_steps={cell['n_steps']} --set model.n_atmodeller={cell['n_atmodeller']}"


def run_command(cfg, cell):
    '''
    The copy-paste MCMC command at a cell, keeping this study's physics overrides (cfg['overrides'] hold TOML
    values, which --set parses back; not the parallel or resolution ones, nor the old n_steps and n_atmodeller).
    '''
    keep = [o for o in cfg.get('overrides', [])
            if not o.startswith(('model.n_steps=', 'model.n_atmodeller=', 'resolution.', 'parallel.'))]
    path = shlex.quote(cfg['config_path']) if cfg.get('config_path') else 'CONFIG'
    return f"python -m isofate.mcmc run {path}{''.join(f' --set {shlex.quote(o)}' for o in keep)} {options_text(cell)}"


def _grid_lines(study):
    '''
    The n_steps x n_atmodeller grid: two lines per n_steps, the verdict mark and worst d, then the mean time per
    evaluation (so that the 7 columns of the default grid fit in WIDTH).
    '''
    cells, m = study['cells'], study['metrics']
    top, bottom = {}, {}
    for c, cell in enumerate(cells):
        key, v = (cell['n_steps'], cell['n_atmodeller']), m['verdict'][c]
        if v == 'skipped':
            top[key], bottom[key] = '-', 'skipped'
            continue
        dtext = 'ref' if v == 'reference' else _fmt_d(m['worst_d'][c])
        top[key] = f'[{VERDICT_MARK[v]} {dtext}]' if c == m['recommended'] else f'{VERDICT_MARK[v]} {dtext}'
        bottom[key] = fmt_seconds(m['mean_runtime'][c])
    header = [f'n_atm = {na}' for na in study['n_atmodeller']]
    width = max(len(t) for t in list(top.values()) + list(bottom.values()) + header) + 2
    lines = ['  n_steps  ' + ''.join(f'{h:>{width}}' for h in header)]
    for ns in study['n_steps']:
        lines.append(f'  {fmt_n(ns):>7}  ' + ''.join(f'{top[ns, na]:>{width}}' for na in study['n_atmodeller']))
        lines.append(' '*11 + ''.join(f'{bottom[ns, na]:>{width}}' for na in study['n_atmodeller']))
    return lines


def _shift_grid_lines(study):
    '''
    The grid of the worst |equivalent shift| per cell, two lines per n_steps like _grid_lines.
    '''
    cells, m = study['cells'], study['metrics']
    top, bottom = {}, {}
    for c, cell in enumerate(cells):
        key, v = (cell['n_steps'], cell['n_atmodeller']), m['verdict_shift'][c]
        if v == 'skipped':
            top[key], bottom[key] = '-', 'skipped'
            continue
        text = 'ref' if v == 'reference' else _fmt_d(m['worst_shift'][c])
        top[key] = f'[{VERDICT_MARK[v]} {text}]' if c == m['recommended_shift'] else f'{VERDICT_MARK[v]} {text}'
        k = m['worst_shift_obs'][c]
        bottom[key] = m['obs_names'][k] if k >= 0 and v != 'reference' and m['worst_shift'][c] > 0 else ''
    header = [f'n_atm = {na}' for na in study['n_atmodeller']]
    width = max(len(t) for t in list(top.values()) + list(bottom.values()) + header) + 2
    lines = ['  n_steps  ' + ''.join(f'{h:>{width}}' for h in header)]
    for ns in study['n_steps']:
        lines.append(f'  {fmt_n(ns):>7}  ' + ''.join(f'{top[ns, na]:>{width}}' for na in study['n_atmodeller']))
        lines.append(' '*11 + ''.join(f'{bottom[ns, na]:>{width}}' for na in study['n_atmodeller']))
    return lines


def _shift_lines(study):
    '''
    table.txt block of the equivalent parameter shift and the second recommendation.
    '''
    cfg, cells, m = study['cfg'], study['cells'], study['metrics']
    j, shift_tol, width = m['shift_j'], m['shift_tol'], m['shift_width']
    param = cfg['free'][j]
    unit = cfg['params'][param]['unit']
    unit_text = f' ({unit})' if unit else ''
    used, names = m['used'], m['obs_names']
    out = _wrap(f'Equivalent shift of {param}{unit_text}: |T(cell) - T(reference)| / |dT/d{param}|, worst over the '
                'test points and the observables that need it (d > tol; below it: which observable)', '')
    if m.get('solve'):
        out += _wrap('(a cross-check in solve mode: the cells were solved, so the solve observable has d of about 0, '
                     f'and its error is the shift of {param}* itself, listed below the grid)', '')
    out += _shift_grid_lines(study)
    out.append('  S stable by the shift criterion   p passes, a finer cell fails   x fails   R reference   '
               '[ ] recommended')
    out += _wrap(f"An observable passes the shift criterion at a test point when d <= tol = {study['tol']:g} OR its "
                 f'equivalent shift is <= shift_tol = {shift_tol:g} of the prior width of {param} '
                 f'(= {shift_tol*width:.3g}{" " + unit if unit else ""}). A steep observable (a thin ridge: moving '
                 f'{param} by shift_tol of its width changes it by more than tol sigma) can fail the fixed-theta '
                 'criterion by a sigma or more while its error is equivalent to a shift of the parameter that is '
                 "negligible next to the posterior's extent along the other parameters. For an observable that is "
                 'not steep the shift condition implies d <= tol, so it must still pass at fixed theta. Instead of '
                 f"|delta log L| the ridge weights must stay put: the cell's values moved back along {param} until "
                 f"{names[m['ridge_obs']]} (the steepest) matches the reference again, the other terms may change by "
                 f"amounts that differ between the test points by at most dlogl_tol = {study['dlogl_tol']:g} (the "
                 'spread reweights the posterior along the ridge).', '  ')
    out.append(f"  derivatives at the reference cell (h = {SHIFT_STEP:g} of the width, per unit of {param}); "
               '* steep:')
    for p in np.flatnonzero(used):
        items = []
        for k, name in enumerate(names):
            flag = '*' if m['steep'][p, k] else ''
            items.append(f'{name} {m["deriv"][p, k]:.3g}{flag}')
        out += _wrap(', '.join(items), f'    point {p}: ')
    rows = [(c, cell) for c, cell in enumerate(cells) if m['computed'][c] and c != study['reference']]
    out.append(f'  ridge-weight spread per cell (must be <= {study["dlogl_tol"]:g})'
               + (f' and the worst shift of {param}* of the re-solve [{unit or "-"}]' if m.get('solve') else '') + ':')
    items = [f"{fmt_n(cell['n_steps'])}/{cell['n_atmodeller']} {_fmt_d(m['ridge_spread'][c])}"
             + (f" ({_fmt_d(m['worst_solve_shift'][c])})" if m.get('solve') else '') for c, cell in rows]
    out += _wrap(', '.join(items), '    ')
    rec = m['recommended_shift']
    if rec is None:
        out.append('SHIFT-BASED RECOMMENDATION: none (no cell is stable by the shift criterion)')
    else:
        cell = cells[rec]
        out.append(f"SHIFT-BASED RECOMMENDATION:  n_steps = {cell['n_steps']}, n_atmodeller = {cell['n_atmodeller']}")
        if rec == m['recommended']:
            why = 'the same cell as the fixed-theta recommendation.'
        else:
            worst = m['worst_shift'][rec]
            k = m['worst_shift_obs'][rec]
            what = f' ({names[k]})' if k >= 0 and worst > 0 else ''
            why = (f"the fastest cell that is stable by the shift criterion: its largest equivalent shift is "
                   f"{_fmt_d(worst)}{' ' + unit if unit else ''}{what}, {_fmt_d(worst/width)} of the prior width, "
                   f"every observable that is not steep passes at fixed theta, and the ridge weights differ between "
                   f"the test points by {_fmt_d(m['ridge_spread'][rec])}; "
                   f"{fmt_seconds(m['mean_runtime'][rec])} per evaluation.")
        out += _wrap(why, '  why:  ')
        out += _wrap('The fixed-theta recommendation above stays the conservative choice. Use this one when the '
                     f'posterior is much wider than {shift_tol*width:.3g}{" " + unit if unit else ""} in {param} '
                     'along the ridge, or after checking that both give the same posterior.', '  note: ')
        out += [f'      {options_text(cell)}']
    return out


def _cell_verdict_text(study, c):
    if c is None:
        return 'not in this grid'
    m, cell = study['metrics'], study['cells'][c]
    v = m['verdict'][c]
    if v == 'skipped':
        return f"skipped ({cell['skip']})"
    if v == 'reference':
        return f"the reference; {fmt_seconds(m['mean_runtime'][c])} per evaluation"
    words = {'stable': 'STABLE', 'pass': 'passes, but a finer cell fails (not stable)', 'fail': 'FAILS'}[v]
    return (f"{words}: worst d = {_fmt_d(m['worst_d'][c])} ({_worst_text(study, c)}), |delta log L| = "
            f"{_fmt_d(m['worst_dlogl'][c])}; {fmt_seconds(m['mean_runtime'][c])} per evaluation")


def format_table(study):
    '''
    table.txt: how to read it, the grid, the recommendation and why, the fiducial and config cells, the noise
    floor, warnings, the test points, and per-cell and per-output details.
    '''
    cfg, cells, m = study['cfg'], study['cells'], study['metrics']
    ref, rec, tol, dlogl_tol = study['reference'], m['recommended'], study['tol'], study['dlogl_tol']
    names, used = m['obs_names'], m['used']
    out = [f"RESOLUTION STUDY: {cfg['name']}   ({study['stamp']}, backend {cfg['model']['backend']}, took "
           f"{fmt_duration(study['wall_s'])})", '='*WIDTH, '']

    out.append('How to read this')
    out += _wrap(f"{study['source_text']} are run at every (n_steps, n_atmodeller) cell and compared with the "
                 f"reference cell, the finest one computed: {cell_name(cells[ref])}.", '  ')
    if study.get('solve'):
        sv = cfg['solve']
        out += _wrap(f"Solve mode: every cell solves for {sv['param']} again at fixed ({sv['name']}, the other "
                     f"parameters), as the sampler does, so d of {sv['observable']} is about 0 by construction (within "
                     f"solve.tol = {sv['tol']:g} sigma) and the other observables show the error that matters for "
                     'the posterior. |delta log L| compares the sampler\'s density, log_prior + log_jac + log_like. '
                     f"A solve took {np.nanmean(m['mean_runs'][m['computed']]):.3g} model runs on average; the times "
                     'below are per evaluation (all its runs).', '  ')
    out += _wrap(f"d = |model(cell) - model(reference)| / sigma for each active observable ({', '.join(names)}), "
                 'in the space where the likelihood compares it: the numerical error of a cell in units of the '
                 'measurement error. The grid shows the worst d over the test points and observables, and below '
                 'it the mean isocalc time per evaluation.', '  ')
    out += _wrap(f'A cell PASSES when d <= tol = {tol:g} and |delta log L| <= {dlogl_tol:g} at every test point. '
                 'It is STABLE when it passes and so does every finer cell (as many or more steps, and n_atmodeller '
                 'as small or smaller, i.e. Atmodeller called at least as often). The recommendation is the '
                 'fastest stable cell.', '  ')
    out.append('')

    out.append('Worst d and mean time per evaluation (n_atmodeller = steps between Atmodeller calls)')
    out += _grid_lines(study)
    out.append('  S stable   p passes, but a finer cell fails   x fails   R reference   - skipped   [ ] recommended')
    if np.isinf(m['worst_d']).any():
        out += _wrap('inf: an observable is defined at one resolution and undefined at the other (e.g. the '
                     'atmosphere is lost at only one of them), or the run failed.', '  ')
    for cell in cells:
        if cell['skip']:
            out.append(f"  skipped {cell_name(cell)}: {cell['skip']}")
    out.append('')

    if rec is None:
        out.append('NO RECOMMENDATION: no cell is stable (every test point failed at the reference cell).')
    else:
        cell = cells[rec]
        out.append(f"RECOMMENDATION:  n_steps = {cell['n_steps']}, n_atmodeller = {cell['n_atmodeller']}")
        if rec == ref:
            why = ('this is the reference itself: no cheaper cell is stable, so the grid does not show convergence '
                   '(see the warnings).')
        else:
            where = 'At the test point' if used.sum() == 1 else f'At all {int(used.sum())} test points'
            why = (f"the fastest stable cell. {where} every observable is within {_fmt_d(m['worst_d'][rec])} sigma "
                   f"of the reference (worst: {_worst_text(study, rec)}) and |delta log L| <= "
                   f"{_fmt_d(m['worst_dlogl'][rec])}; every finer cell passes too.")
        out += _wrap(why, '  why:  ')
        cost = f"{fmt_seconds(m['mean_runtime'][rec])} per evaluation"
        if rec != ref and m['mean_runtime'][rec] > 0:
            cost += (f", {m['mean_runtime'][ref]/m['mean_runtime'][rec]:.1f}x faster than the reference "
                     f"({fmt_seconds(m['mean_runtime'][ref])})")
        out += _wrap(cost + '.', '  cost: ')
        out += ['  To use it, add to the run command:', f'      {options_text(cell)}', '  for example:',
                f'      {run_command(cfg, cell)}',
                f"  or set n_steps = {cell['n_steps']} and n_atmodeller = {cell['n_atmodeller']} in the [model] table "
                'of the config.']
        if not study['source'].startswith('chain:'):
            out += _wrap('after a first MCMC run, re-check on the posterior itself, with enough points to reach '
                         'its tails: python -m isofate.mcmc resolution CONFIG --source chain:RUN_DIR --points 30 '
                         '(and only the few cells in question)', '  next: ')
    out.append('')

    fid = _cell_index(cells, *FIDUCIAL)
    out += _wrap(_cell_verdict_text(study, fid), '  fiducial (n_steps = 1e5, n_atmodeller = 100): ', '    ')
    current = (cfg['model']['n_steps'], cfg['model']['n_atmodeller'])
    if current != FIDUCIAL:
        out += _wrap(_cell_verdict_text(study, _cell_index(cells, *current)),
                     f'  this config (n_steps = {fmt_n(current[0])}, n_atmodeller = {current[1]}): ', '    ')
    out.append('')

    if m.get('has_shift') and used.any():
        out += _shift_lines(study)
        out.append('')

    if study['perturb'] > 0:
        moved = study.get('space', cfg)['free'][study.get('noise_j', 0)]
        how = ('raw differences' if study.get('solve') or not m.get('has_shift') else
               f'after subtracting the smooth change d{moved} x dT/d{moved} of the derivative runs')
        out.append(f"Noise floor: every run repeated with {moved} moved by {study['perturb']:g} of its prior width "
                   f'(same units as d; {how})')
        rows = ([('recommended', rec)] if rec is not None and rec != ref else []) + [('reference', ref)]
        for label, c in rows:
            noise = m['worst_noise'][c]
            out.append(f"  {label + ' cell:':<18}worst {_fmt_d(noise)} ({_worst_text(study, c, 'noise')}), "
                       f"|delta log L| {_fmt_d(m['worst_dlogl_noise'][c])}: "
                       f"{'ROUGH (>= tol)' if noise >= tol else 'smooth'}")
        out.append('')

    out.append('Warnings')
    for warning in study['warnings']: # each line of a warning on its own (e.g. the bisection's failure counts)
        for k, part in enumerate(str(warning).splitlines()):
            out += _wrap(part.strip(), '  - ' if k == 0 else '      ')
    if not study['warnings']:
        out.append('  none')
    out.append('')

    out.append('Test points' + (" (the sampler's coordinates; the solved value at the reference cell after it)"
                                if study.get('solve') else ''))
    space = study.get('space', cfg)
    for p, theta in enumerate(study['thetas']):
        solved = ''
        if study.get('solve'):
            sv = cfg['solve']
            solved = f"; {sv['param']}* = {study['blobs']['solve_value'][p, ref, 0]:.6g}"
        out.append(f"  point {p}: {_fmt_theta(theta, space)}{solved}{'' if used[p] else '   (left out)'}")
    out.append('')

    out.append('Per cell: worst d of each observable, |delta log L|, the noise floor, time per evaluation, failed '
               'runs')
    widths = [max(9, len(n) + 2) for n in names]
    out.append(f"  {'n_steps':>7} {'n_atm':>6}  {'verdict':<10}" + ''.join(f'{n:>{w}}' for n, w in zip(names, widths))
               + f"{'|dlogL|':>9}{'noise':>9}{'s/eval':>11}{'failed':>8}")
    for c, cell in enumerate(cells):
        if not m['computed'][c]:
            continue
        worst = [np.max(m['d'][used, c, j]) if used.any() else nan for j in range(len(names))]
        out.append(f"  {fmt_n(cell['n_steps']):>7} {cell['n_atmodeller']:>6}  {m['verdict'][c]:<10}"
                   + ''.join(f'{_fmt_d(x):>{w}}' for x, w in zip(worst, widths))
                   + f"{_fmt_d(m['worst_dlogl'][c]):>9}{_fmt_d(m['worst_noise'][c]):>9}"
                   + f"{fmt_seconds(m['mean_runtime'][c]):>11}{int(m['n_failed'][c]):>8}")
    out.append('')

    columns = []
    for label, c in (('recommended', rec), ('fiducial', fid), ('this config', _cell_index(cells, *current))):
        if c is not None and m['computed'][c] and c != ref and c not in [k for _, k in columns]:
            columns.append((label, c))
    if columns:
        out.append('Relative change of every output against the reference, max over test points: |x - x_ref|/|x_ref|')
        out.append('  (undef: undefined at both resolutions; inf: defined at only one of them, or x_ref = 0)')
        name_width = max(14, max(len(f) for f in m['rel_fields']) + 1)
        out.append(f"  {'output':<{name_width}}" + ''.join(f'{label:>16}' for label, _ in columns))
        out.append(f"  {'':<{name_width}}" + ''.join(
            f"{fmt_n(cells[c]['n_steps']) + '/' + str(cells[c]['n_atmodeller']):>16}" for _, c in columns))
        for i, field in enumerate(m['rel_fields']):
            row = []
            for _, c in columns:
                values = m['rel_diff'][used, c, i]
                values = values[~np.isnan(values)]
                row.append('undef' if values.size == 0 else _fmt_d(np.max(values)))
            out.append(f'  {field:<{name_width}}' + ''.join(f'{x:>16}' for x in row))
        out.append('')
    return '\n'.join(out) + '\n'


###_____Running_____###

class _Progress:
    '''
    Progress lines: evaluations done/total, elapsed time, and an ETA from the cost model scaled to the measured
    speed. Printed at every 10% that is at least 5 s after the previous line, every 60 s, and at the end.
    '''

    def __init__(self, total, total_cost):
        self.total, self.total_cost = total, total_cost
        self.done, self.done_cost = 0, 0.0
        self.busy = 0.0 # [s] wall time spent in earlier evaluation loops
        self.start = self._last = time.perf_counter()

    def update(self, cost, loop_start):
        self.done += 1
        self.done_cost += cost
        now = time.perf_counter()
        milestone = self.done % max(1, self.total//10) == 0 and now - self._last >= 5
        if self.done != self.total and not milestone and now - self._last < 60:
            return
        self._last = now
        busy = self.busy + now - loop_start
        eta = busy/self.done_cost*(self.total_cost - self.done_cost) if self.done_cost > 0 else nan
        left = f', about {fmt_duration(eta)} left' if self.done < self.total and np.isfinite(eta) else ''
        print(f'[resolution] {self.done}/{self.total} evaluations, {fmt_duration(now - self.start)} elapsed{left}',
              flush = True)


def _tiers(cells, computed, cfg_n_steps, select):
    '''
    The worker pools in run order: cells with n_steps <= SMALL_TIER in one, larger ones in another (fewer workers
    fit in memory). The test-point selection runs at the config resolution, first, in the pool of its n_steps.
    '''
    small = [c for c in computed if cells[c]['n_steps'] <= SMALL_TIER]
    large = [c for c in computed if cells[c]['n_steps'] > SMALL_TIER]
    select_small = cfg_n_steps <= SMALL_TIER
    tiers = [{'cells': small, 'n_steps_max': SMALL_TIER, 'select': select and select_small},
             {'cells': large, 'select': select and not select_small,
              'n_steps_max': max([cells[c]['n_steps'] for c in large] + [cfg_n_steps])}]
    return sorted([t for t in tiers if t['cells'] or t['select']], key = lambda t: not t['select'])


def _new_dir(cfg, out_dir):
    '''
    The study directory: out_dir (must be new or empty), else <output_root>/<name>/resolution_<stamp> (claimed
    atomically).
    '''
    if out_dir is None:
        base = os.path.join(cfg['output_root'], cfg['name'], f"resolution_{time.strftime('%Y%m%d-%H%M%S')}")
        return config.new_output_dir(base, first = 2)
    return config.empty_output_dir(out_dir, 'a study')


def _json_cell(cell, c, m):
    return {'n_steps': cell['n_steps'], 'n_atmodeller': cell['n_atmodeller'], 'verdict': m['verdict'][c],
            'skip': cell['skip'], 'est_seconds': cell['est_s'], 'seconds_per_eval': float(m['mean_runtime'][c]),
            'worst_d': float(m['worst_d'][c]),
            'worst_observable': m['obs_names'][m['worst_obs'][c]] if m['worst_obs'][c] >= 0 else None,
            'worst_point': int(m['worst_point'][c]), 'worst_dlogl': float(m['worst_dlogl'][c]),
            'worst_noise': float(m['worst_noise'][c]), 'failed_runs': float(m['n_failed'][c]),
            'model_runs_per_eval': float(m['mean_runs'][c])}


def _write_outputs(study, out_dir):
    cfg, cells, m = study['cfg'], study['cells'], study['metrics']
    ref, rec = study['reference'], m['recommended']
    cfg_names = cfg['observables']
    shift = {}
    if m.get('has_shift'):
        rec_shift = m['recommended_shift']
        shift = {key: m[key] for key in ('deriv', 'shift', 'shift_frac', 'steep', 'worst_shift', 'worst_shift_obs',
                                         'worst_shift_point', 'passes_shift', 'stable_shift', 'shift_width',
                                         'shift_h', 'shift_tol', 'ridge_dlogl', 'ridge_spread', 'ridge_obs')}
        shift.update(shift_blobs = study['shift_blobs'], verdict_shift = np.array(m['verdict_shift']),
                     recommended_shift = -1 if rec_shift is None else rec_shift,
                     shift_param = cfg['free'][m['shift_j']])
    solved = {'solve': bool(m.get('solve'))}
    if m.get('solve'):
        solved.update(solve_shift = m['solve_shift'], worst_solve_shift = m['worst_solve_shift'],
                      solve_param = cfg['solve']['param'], solve_observable = cfg['solve']['observable'])
    np.savez_compressed(os.path.join(out_dir, 'points.npz'), theta = study['thetas'],
                        theta_perturbed = study['thetas_perturbed'], theta_shifted = study['thetas_shifted'],
                        free = np.array(study.get('space', cfg)['free']), physical_free = np.array(cfg['free']),
                        source = study['source'], solve = bool(m.get('solve')), **study['selection'])
    np.savez_compressed(
        os.path.join(out_dir, 'results.npz'), blobs = study['blobs'], lp = study['lp'],
        n_steps = np.array([cell['n_steps'] for cell in cells]),
        n_atmodeller = np.array([cell['n_atmodeller'] for cell in cells]),
        grid_n_steps = np.array(study['n_steps']), grid_n_atmodeller = np.array(study['n_atmodeller']),
        est_seconds = np.array([cell['est_s'] for cell in cells]),
        skip = np.array([cell['skip'] or '' for cell in cells]),
        theta = np.stack([study['thetas'], study['thetas_perturbed']], axis = 1),
        free = np.array(study.get('space', cfg)['free']),
        obs_names = np.array(m['obs_names']), obs_labels = np.array([cfg_names[n]['label'] for n in m['obs_names']]),
        obs_units = np.array([cfg_names[n]['unit'] for n in m['obs_names']]),
        obs_spaces = np.array([cfg_names[n]['space'] for n in m['obs_names']]),
        rel_fields = np.array(m['rel_fields']), verdict = np.array(m['verdict']), reference = ref,
        recommended = -1 if rec is None else rec, tol = study['tol'], dlogl_tol = study['dlogl_tol'],
        perturb = study['perturb'], name = cfg['name'], **shift, **solved,
        **{key: m[key] for key in ('computed', 'used', 'd', 'noise', 'dlogl', 'dlogl_noise', 'changed',
                                   'rel_diff', 'worst_d', 'worst_dlogl', 'worst_noise', 'worst_dlogl_noise',
                                   'mean_runtime', 'mean_runs', 'n_failed', 'worst_obs', 'worst_point', 'passes',
                                   'stable')})

    def summary(c):
        return None if c is None else _json_cell(cells[c], c, m)

    rec_json = None
    if rec is not None:
        rec_json = summary(rec)
        rec_json.update(options = options_text(cells[rec]), command = run_command(cfg, cells[rec]),
                        speedup_vs_reference = float(m['mean_runtime'][ref]/m['mean_runtime'][rec])
                        if m['mean_runtime'][rec] > 0 else None)
    shift_json = None
    if m.get('has_shift'):
        c = m['recommended_shift']
        shift_json = {'param': cfg['free'][m['shift_j']], 'shift_tol': m['shift_tol'],
                      'prior_width': float(m['shift_width']), 'recommended': None}
        if c is not None:
            shift_json['recommended'] = dict(summary(c), options = options_text(cells[c]),
                                             command = run_command(cfg, cells[c]),
                                             worst_shift = float(m['worst_shift'][c]))
        shift_json['cells'] = [{'n_steps': cell['n_steps'], 'n_atmodeller': cell['n_atmodeller'],
                                'verdict': m['verdict_shift'][k], 'worst_shift': float(m['worst_shift'][k]),
                                'ridge_spread': float(m['ridge_spread'][k])}
                               for k, cell in enumerate(cells)]
    record = {'name': cfg['name'], 'created': study['stamp'], 'out_dir': out_dir, 'backend': cfg['model']['backend'],
              'recommended': rec_json, 'recommended_shift': shift_json['recommended'] if shift_json else None,
              'equivalent_shift': shift_json, 'reference': summary(ref),
              'fiducial': summary(_cell_index(cells, *FIDUCIAL)),
              'config_cell': summary(_cell_index(cells, cfg['model']['n_steps'], cfg['model']['n_atmodeller'])),
              'tol': study['tol'], 'dlogl_tol': study['dlogl_tol'], 'perturb': study['perturb'],
              'observables': m['obs_names'], 'points': len(study['thetas']), 'source': study['source'],
              'solve': bool(m.get('solve')),
              'wall_seconds': study['wall_s'], 'warnings': study['warnings'],
              'cells': [_json_cell(cell, c, m) for c, cell in enumerate(cells)]}
    with open(os.path.join(out_dir, 'recommendation.json'), 'w') as f:
        json.dump(record, f, indent = 1)
        f.write('\n')
    with open(os.path.join(out_dir, 'table.txt'), 'w') as f:
        f.write(study['table'])


def run_resolution(cfg, grid = 'default', points = None, source = None, workers = None, tol = None,
                   out_dir = None, dry_run = False, thetas = None, solve_mode = None):
    '''
    Runs the study, writes its files (see the module docstring) and prints table.txt. Returns out_dir, or None
    for dry_run, which only prints the plan (cells, points, estimated time).

    grid: 'default' (cfg['resolution']), 'quick' (n_steps 1e4, 3e4, 1e5, 3e5 x n_atmodeller 1000, 100, 10) or a
          dict of n_steps and/or n_atmodeller lists
    points, source ('auto', 'prescan', 'prior', 'bisect', 'solve', 'chain:RUN_DIR'; 'auto' is 'solve' in solve
          mode, else 'bisect' when sampler.init is 'bisect' or 'solve', else 'prescan'), tol: default from
          cfg['resolution']
    thetas: explicit test points (n, ndim) instead of a source (in the sampler's coordinates in solve mode)
    workers: default parallel.workers, capped by memory per pool
    out_dir: default <output_root>/<name>/resolution_<stamp>; never overwritten
    solve_mode: evaluate the cells in solve mode (default: when [solve] is active; False: at fixed physical theta)
    '''
    t_start = time.perf_counter()
    res = cfg['resolution']
    n_steps, n_atm = grid_values(cfg, grid)
    tol = res['tol'] if tol is None else float(tol)
    dlogl_tol, perturb = res['dlogl_tol'], res['perturb']
    solving = config.solve_on(cfg) if solve_mode is None else bool(solve_mode)
    if solving and not config.solve_on(cfg):
        raise config.ConfigError('--solve needs solve mode in the config: set [solve] param (e.g. param = "log_f_atm", '
                                 'observable = "Rp")')
    space = prm.sampled_cfg(cfg) if solving else cfg # the coordinates of the test points
    if not cfg['free']:
        raise config.ConfigError('the resolution study needs a free parameter (vary = true) to draw test points')
    if all(o['input'] for o in cfg['observables'].values()):
        raise config.ConfigError('the resolution study compares model observables, but none is active (only inputs '
                                 'like Mp): switch one on, e.g. --set observables.Rp.use=true with value and sigma')
    if thetas is not None:
        thetas = np.atleast_2d(np.asarray(thetas, dtype = float))
        if thetas.shape[1] != len(space['free']):
            raise config.ConfigError(f"thetas need {len(space['free'])} columns ({space['free']}), got {thetas.shape}")
        outside = [i for i, theta in enumerate(thetas) if not np.isfinite(prm.log_prior(theta, space))]
        if outside:
            raise config.ConfigError(f'test points {outside} lie outside the prior')
        source, points = 'thetas', len(thetas)
    else:
        source = config.resolution_source(cfg, source, solving)
        points = res['points'] if points is None else int(points)
        if points < 1:
            raise config.ConfigError(f'points must be >= 1, got {points}')

    cells = plan_cells(cfg, n_steps, n_atm)
    ref = reference_index(cells)
    if ref is None:
        raise config.ConfigError('every cell of the grid is skipped: raise resolution.max_eval_seconds or use '
                                 'cheaper cells')
    n_k = 2 if perturb > 0 else 1
    computed = [c for c, cell in enumerate(cells) if cell['skip'] is None]
    cfg_cell = (cfg['model']['n_steps'], cfg['model']['n_atmodeller'])
    shift_j, shift_tol = shift_param_index(cfg, solving), res['shift_tol'] # index in the physical theta
    noise_j = cfg['solve']['index'] if solving else shift_j # the perturbed coordinate (u in solve mode)
    runs = solve.EVALS_GUESS if solving else 1 # model runs per evaluation of a cell (measured later)
    # evaluations of the selection: the bisection's upper bound is a full batch of iters rounds
    n_select = {'prescan': 20*points, 'prior': 4*points, 'solve': bisect_draws(points),
                'bisect': bisect_draws(points)*(2 + cfg['sampler']['bisect_iters'])}.get(source, 0)
    select_runs = runs if source in ('prescan', 'prior', 'solve') else 1
    tiers = _tiers(cells, computed, cfg_cell[0], n_select > 0)
    requested = cfg['parallel']['workers'] if workers is None else int(workers)

    def plan_seconds(runs):
        wall = 0.0 # [s]
        for tier in tiers:
            n_shift = points if ref in tier['cells'] else 0 # the derivative runs at the reference cell
            n_tasks = len(tier['cells'])*points*n_k + n_shift + (n_select if tier['select'] else 0)
            tier['workers'] = choose_workers(max(1, min(requested, n_tasks)), tier['n_steps_max'], cfg)[0]
            costs = [runs*cost_model(cells[c]['n_steps'], cells[c]['n_atmodeller']) for c in tier['cells']]
            total = (sum(costs)*points*n_k + n_shift*cost_model(cells[ref]['n_steps'], cells[ref]['n_atmodeller'])
                     + (n_select*select_runs*cost_model(*cfg_cell) if tier['select'] else 0))
            wall += max(total/tier['workers'], max(costs, default = 0.0))
        return wall

    wall = plan_seconds(runs)
    n_evals = len(computed)*points*n_k + points
    why = ['the second run of each pair, at a perturbed theta, measures the noise floor'] if n_k > 1 else []
    why.append(f"the {points} more at the reference cell give d/d{cfg['free'][shift_j]} for the equivalent shift")
    plan = [f"Resolution study of {cfg['name']}: {len(computed)} cells x {_plural(points, 'test point')} x {n_k} "
            f"run{'s' if n_k > 1 else ''} + {points} = {n_evals} evaluations ({'; '.join(why)})",
            f"  n_steps: {', '.join(map(fmt_n, n_steps))};  n_atmodeller: {', '.join(map(str, n_atm))}",
            f'  reference: {cell_name(cells[ref])} (the finest computed cell)']
    if solving:
        sv = cfg['solve']
        plan.append(f"  solve mode: each cell solves for {sv['param']} at fixed ({sv['name']}, the other parameters), "
                    f'about {runs} model runs per evaluation (a guess until measured on the test-point selection)')
    if source in ('prescan', 'prior', 'solve'):
        plan.append(f'  test points: {source}, {n_select} evaluations at the config resolution (n_steps = '
                    f'{fmt_n(cfg_cell[0])}, n_atmodeller = {cfg_cell[1]})'
                    + (f', up to {search.BATCHES} batches' if source == 'solve' else ''))
    elif source == 'bisect':
        param, name = config.bisect_target(cfg)
        plan.append(f'  test points: bisect, {bisect_draws(points)} draws of {param} bisected on {name}, at most '
                    f'{n_select} evaluations per batch (usually fewer; up to {search.BATCHES} batches) at the config '
                    f'resolution (n_steps = {fmt_n(cfg_cell[0])}, n_atmodeller = {cfg_cell[1]})')
    else:
        plan.append(f'  test points: {source}')
    plan += [f"  skipped {cell_name(cell)}: {cell['skip']}" for cell in cells if cell['skip']]
    pools = ' then '.join(f"{t['workers']} worker{'s' if t['workers'] > 1 else ''}" for t in tiers)
    plan.append(f'  estimated time: about {fmt_duration(wall)} with {pools} (cost model of the real isocalc; it '
                'can take up to twice that), plus about 10 s of warm-up per pool')
    print('\n'.join(plan), flush = True)
    if dry_run:
        return None

    out_dir = _new_dir(cfg, out_dir)
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    saved = copy.deepcopy(cfg)
    saved['resolution'].update(n_steps = n_steps, n_atmodeller = n_atm, points = points, source = source, tol = tol)
    config.save_resolved(saved, os.path.join(out_dir, 'config.json'))
    print(f'[resolution] writing to {out_dir}', flush = True)

    rng = np.random.default_rng(cfg['sampler']['seed'])
    warnings, selection = [], {}
    if source.startswith('chain:'):
        thetas, selection = _chain_points(os.path.abspath(source[len('chain:'):]), cfg, points, rng, warnings, solving)
    blobs = lp = thetas_perturbed = progress = None
    errors, ref_errors = {}, {}
    for tier in tiers:
        with WorkerPool(cfg, tier['workers'], tier['n_steps_max'], run_dir = out_dir) as pool:
            if tier['select']:
                print(f"[resolution] choosing test points ({source}): {'up to ' if source == 'bisect' else ''}"
                      f'{n_select} evaluations at the config resolution', flush = True)
                pick = {'prescan': _prescan_points, 'prior': _prior_points, 'bisect': _bisect_points,
                        'solve': _solve_points}[source]
                thetas, selection = pick(cfg, pool, points, rng, warnings, solving)
                measured = selection.get('candidates_runs')
                if solving and measured is not None and np.isfinite(measured).any():
                    runs = float(np.nanmean(measured))
                    print(f'[resolution] {runs:.3g} model runs per solve measured on the selection: the cells will '
                          f'take about {fmt_duration(plan_seconds(runs))} in total', flush = True)
            if blobs is None:
                thetas = np.asarray(thetas, dtype = float)
                thetas_perturbed = np.array([perturbed(theta, space, perturb, noise_j) for theta in thetas])
                blobs = np.empty((len(thetas), len(cells), n_k), dtype = model.blob_dtype())
                blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
                lp = np.full(blobs.shape, nan)
                if not solving:
                    shift_runs = [shifted(theta, cfg, shift_j) for theta in thetas]
                    thetas_shifted = np.array([t for t, _ in shift_runs])
                    shift_h = np.array([h for _, h in shift_runs])
                else: # set after the reference cell has solved: (P* + h, phi)
                    thetas_shifted, shift_h = np.full(thetas.shape, nan), np.full(len(thetas), nan)
                shift_blobs = np.empty(len(thetas), dtype = model.blob_dtype())
                shift_blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
                tasks_per_cell = len(thetas)*n_k
                ref_cost = cost_model(cells[ref]['n_steps'], cells[ref]['n_atmodeller'])
                progress = _Progress(len(computed)*tasks_per_cell + len(thetas), tasks_per_cell*runs*sum(
                    cost_model(cells[c]['n_steps'], cells[c]['n_atmodeller']) for c in computed)
                    + len(thetas)*ref_cost)
            tasks = [{'theta': thetas[p] if k == 0 else thetas_perturbed[p], 'n_steps': cells[c]['n_steps'],
                      'n_atmodeller': cells[c]['n_atmodeller'], 'id': (p, c, k), 'solve': solving}
                     for c in tier['cells'] for p in range(len(thetas)) for k in range(n_k)]
            if ref in tier['cells'] and not solving: # k = 2: theta + h e_j at the reference cell
                tasks += [{'theta': thetas_shifted[p], 'n_steps': cells[ref]['n_steps'],
                           'n_atmodeller': cells[ref]['n_atmodeller'], 'id': (p, ref, 2)} for p in range(len(thetas))]
            tasks.sort(key = lambda task: -cells[task['id'][1]]['est_s']) # longest first
            loop_start = time.perf_counter()
            for r in pool.imap_unordered(model.eval_task, tasks):
                p, c, k = r['id']
                if k == 2:
                    shift_blobs[p] = model.blob(r)
                    progress.update(ref_cost, loop_start)
                    continue
                blobs[p, c, k] = model.blob(r)
                lp[p, c, k] = r['lp']
                if r['error'] and r['status'] not in RAN:
                    errors.setdefault(c, r['error'])
                    if c == ref and k == 0:
                        ref_errors[p] = r['error']
                cell = cells[c]
                progress.update(runs*cost_model(cell['n_steps'], cell['n_atmodeller']), loop_start)
            if ref in tier['cells'] and solving: # the derivative runs at (P* + h, phi) of the reference cell
                tasks = []
                for p in range(len(thetas)):
                    value = blobs['solve_value'][p, ref, 0]
                    if math.isfinite(value):
                        thetas_shifted[p], shift_h[p] = shifted(solve.physical(thetas[p], value, cfg), cfg, shift_j)
                        tasks.append({'theta': thetas_shifted[p], 'n_steps': cells[ref]['n_steps'],
                                      'n_atmodeller': cells[ref]['n_atmodeller'], 'id': (p, ref, 2)})
                progress.total -= len(thetas) - len(tasks)
                for r in pool.imap_unordered(model.eval_task, tasks):
                    shift_blobs[r['id'][0]] = model.blob(r)
                    progress.update(ref_cost, loop_start)
            progress.busy += time.perf_counter() - loop_start

    noise_step = thetas_perturbed[:, noise_j] - thetas[:, noise_j]
    study = {'cfg': cfg, 'cells': cells, 'n_steps': n_steps, 'n_atmodeller': n_atm, 'reference': ref, 'tol': tol,
             'dlogl_tol': dlogl_tol, 'perturb': perturb, 'thetas': thetas, 'thetas_perturbed': thetas_perturbed,
             'blobs': blobs, 'lp': lp, 'source': source, 'selection': selection, 'stamp': stamp,
             'source_text': _source_text(source, len(thetas), selection), 'errors': errors,
             'ref_errors': ref_errors, 'n_unexpected': int(np.sum(blobs['status'] == model.UNEXPECTED)),
             'selection_warnings': warnings, 'out_dir': out_dir, 'shift_blobs': shift_blobs,
             'thetas_shifted': thetas_shifted, 'solve': solving, 'space': space, 'noise_j': noise_j}
    study['metrics'] = analyse(cfg, cells, blobs, ref, tol, dlogl_tol, shift_blobs, shift_h, shift_j, shift_tol,
                               solving = solving, noise_step = noise_step)
    study['warnings'] = _warnings(study)
    study['wall_s'] = time.perf_counter() - t_start
    study['table'] = format_table(study)
    _write_outputs(study, out_dir)
    print('\n' + study['table'], flush = True)

    try:
        from isofate.mcmc import plots
        plots.plot_resolution(out_dir)
    except Exception as e: # the results are saved already
        print(f'[resolution] warning: the plots failed ({type(e).__name__}: {e})', flush = True)

    rec, rec_shift = study['metrics']['recommended'], study['metrics'].get('recommended_shift')
    if rec is not None:
        print(f'Recommended: {options_text(cells[rec])}   ({cell_name(cells[rec])}; details in table.txt)')
    if rec_shift is not None and rec_shift != rec:
        print(f'Shift-based: {options_text(cells[rec_shift])}   (equivalent shift <= shift_tol; see table.txt)')
    print(f'Results: {out_dir}', flush = True)
    return out_dir
