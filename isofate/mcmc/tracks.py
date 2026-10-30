'''
Posterior evolutionary histories: the time series of the MAP and of posterior samples of a run.

The chain's blobs hold only the final state of each evaluation (a whole isocalc solution is 31 MB at 1e5 steps,
far too much to keep for every walker and step). So posterior samples of a run (and its MAP) are re-run through
a worker pool with save_molecules = True, and their time series are stored on one log-spaced time grid. In solve
mode they are re-run at the physical point (P*, phi), P* from the blob field solve_value, without solving again.
The model is deterministic, so each re-run reproduces its chain log-probability; compute_tracks checks and logs
that.

<run_dir>/tracks.npz holds
    grid_yr       [yr]: track_points log-spaced times from t0 to the oldest sample's age, plus each sample's age
                  (sorted, so each track ends on the grid)
    theta         (n, ndim) and free (ndim,): the re-run physical parameters; row 0 is the MAP (is_map)
    is_map, status, lp, error (str)   (n,) per track
    final         (n,) structured, model.BLOB_FIELDS: the final observables of each re-run
    n_steps, n_atmodeller
    one (n, grid size) array per key of model.TRACK_KEYS (see model.make_track), float32 except the atom
    numbers N_X, N_X_int (float64); NaN after a sample's age, composition NaN once the atmosphere is gone.
'''

import math
import os
import time

import numpy as np

from isofate.mcmc import model, sampler
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool

TRACKS = 'tracks.npz'


###_____Samples and the time grid_____###

def pick_samples(run, n, seed = 0):
    '''
    Picks the points to re-run: the MAP of the chain, then min(n, N) of the N posterior samples
    The MAP is the highest log_prob over every step (burn-in included) and walker of the chain; in solve mode that
    is the sampled density, which includes ln|dP*/du|. The samples come from sampler.posterior (after burn-in and
    thinning), drawn without replacement.

    Inputs:
        - run: the run, as sampler.load_run returns it [dict]
        - n: number of posterior samples [int]
        - seed: seed of the draw [int]

    Output: tuple
        - thetas: physical parameters (k, ndim), k = 1 + min(n, N); row 0 is the MAP (sampler.physical_samples:
          in solve mode P* from the blobs in place of u) [array]
        - is_map: True for row 0 only (k,) [Bool array]
        - blobs: the chain's blobs of those points (k,) [structured array]
    '''
    lp = run['log_prob']
    i_step, i_walker = np.unravel_index(np.argmax(lp), lp.shape) # the MAP
    samples, blobs = sampler.posterior(run)
    rng = np.random.default_rng(seed)
    pick = rng.choice(samples.shape[0], min(int(n), samples.shape[0]), replace = False)
    thetas = np.vstack([run['chain'][i_step, [i_walker]], samples[pick]])
    chosen = np.concatenate([run['blobs'][i_step, [i_walker]], blobs[pick]])
    is_map = np.zeros(len(thetas), dtype = bool)
    is_map[0] = True
    return sampler.physical_samples(thetas, chosen, run['cfg']), is_map, chosen


def time_grid(cfg, thetas, n_points):
    '''
    Builds the common time grid of the tracks
    n_points log-spaced times from t0 to the largest age among thetas, plus every theta's age (sorted, without
    duplicates). With its own age on the grid, each track ends at the state the likelihood compared with the data:
    between two log-spaced times a steep quantity, e.g. the outflow H fraction falling by 0.5 dex in the last 100
    Myr, would otherwise end early.

    Inputs:
        - cfg: resolved config [dict]
        - thetas: physical parameters (k, ndim) [array]
        - n_points: number of log-spaced times (sampler.track_points) [int]

    Output: times, at most n_points + k of them [yr; array]
    '''
    ages = [prm.theta_to_params(theta, cfg)['age'] for theta in thetas]
    return np.union1d(np.geomspace(cfg['model']['t0'], max(ages), int(n_points)), ages)


###_____Re-running the samples_____###

def _run_tasks(pool, tasks, log):
    '''
    Runs the tasks through pool.imap_unordered and returns their results in task order (by id), logging the
    progress and an ETA about 10 times.
    '''
    results = [None]*len(tasks)
    start = time.perf_counter()
    every = max(10, math.ceil(len(tasks)/10)) # about 10 progress lines
    for done, res in enumerate(pool.imap_unordered(model.eval_task, tasks), 1):
        results[res['id']] = res
        if done % every == 0 or done == len(tasks):
            elapsed = time.perf_counter() - start # [s]
            log(f'tracks: {done}/{len(tasks)} done in {sampler.duration_text(elapsed)}, ETA '
                f'{sampler.duration_text(elapsed*(len(tasks) - done)/done)}')
    return results


def compute_tracks(run_dir, n = None, workers = None, seed = 0, pool = None):
    '''
    Computes the posterior tracks of a run and writes <run_dir>/tracks.npz (see the module docstring)
    Re-runs the MAP and n posterior samples at the run's n_steps and n_atmodeller, with save_molecules = True,
    and logs how many reproduce the chain's log-probability.

    Inputs:
        - run_dir: the run directory [str]
        - n: number of posterior samples; default sampler.tracks of the run's config [int]
        - workers: worker count of a new pool; default parallel.workers [int]
        - seed: seed of the choice of samples [int]
        - pool: a running WorkerPool of the same cfg (the sampler's), else a new one is started

    Output: path of tracks.npz [str]
    '''
    run = sampler.load_run(run_dir)
    cfg, run_dir = run['cfg'], run['run_dir']
    log = sampler.RunLog(run_dir) # prints, and appends to run.log
    n = cfg['sampler']['tracks'] if n is None else int(n)
    thetas, is_map, chain_blobs = pick_samples(run, n, seed)
    grid = time_grid(cfg, thetas, cfg['sampler']['track_points'])
    m = cfg['model']
    tasks = [{'theta': theta, 'n_steps': m['n_steps'], 'n_atmodeller': m['n_atmodeller'], 'save_molecules': True,
              'track_grid': grid, 'id': i} for i, theta in enumerate(thetas)]
    log(f'tracks: re-running the MAP and {len(tasks) - 1} posterior samples (n_steps = {m["n_steps"]}, '
        f'n_atmodeller = {m["n_atmodeller"]}, save_molecules) on {grid.size} times, {grid[0]:.3g} to '
        f'{grid[-1]:.3g} yr')
    if pool is None:
        with WorkerPool(cfg, workers, run_dir = run_dir) as wp:
            results = _run_tasks(wp, tasks, log)
    else:
        results = _run_tasks(pool, tasks, log)

    out = {'grid_yr': grid, 'theta': thetas, 'free': np.array(cfg['free']), 'is_map': is_map,
           'status': np.array([r['status'] for r in results], dtype = float),
           'lp': np.array([r['lp'] for r in results], dtype = float),
           'error': np.array([r['error'] or '' for r in results], dtype = str),
           'final': np.array([model.blob(r) for r in results], dtype = model.blob_dtype()),
           'n_steps': m['n_steps'], 'n_atmodeller': m['n_atmodeller']}
    for key in model.TRACK_KEYS:
        out[key] = np.stack([r['track'][key] for r in results]) # (n, grid size)
    # written under a temporary name, then renamed: a kill mid-write never leaves a broken tracks.npz
    path = os.path.join(run_dir, TRACKS)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)

    # the re-runs repeat the chain's evaluations (save_molecules does not change the physics). They are compared
    # with log_prior + log_like of the chain's blobs, not with its log_prob, which in solve mode also holds log_jac
    same = [math.isclose(r['lp'], b['log_prior'] + b['log_like'], rel_tol = 1e-9, abs_tol = 1e-12)
            for r, b in zip(results, chain_blobs) if r['status'] == model.OK]
    log(f'tracks: {len(results)} runs ({sampler.status_text(sampler.status_counts(out["status"]))}), '
        f'{sum(same)}/{len(same)} reproduce the chain\'s log-probability; {path} '
        f'({os.path.getsize(path)/1e6:.1f} MB)')
    if not all(same):
        log('tracks: warning: some re-runs differ from the chain; is the model deterministic?')
    return path


def load_tracks(run_dir):
    '''
    Reads <run_dir>/tracks.npz

    Output: its arrays by name (see the module docstring) [dict]
    '''
    with np.load(os.path.join(run_dir, TRACKS), allow_pickle = False) as d:
        return {key: d[key] for key in d.files}
