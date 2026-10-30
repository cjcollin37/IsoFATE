'''
Posterior evolutionary histories: posterior samples of a run (and its MAP) are re-run through a worker pool
with save_molecules = True, and their time series are stored on one log-spaced time grid. In solve mode they are
re-run at the physical point (P*, phi), P* from the blob field solve_value, without solving again.

<run_dir>/tracks.npz holds
    grid_yr       [yr]: track_points log-spaced times from t0 to the oldest sample's age, plus each sample's age
                  (sorted, so each track ends on the grid)
    theta         (n, ndim) and free (ndim,): the re-run physical parameters; row 0 is the MAP (is_map)
    is_map, status, lp, error (str)   (n,) per track
    final         (n,) structured, model.BLOB_FIELDS: the final observables of each re-run
    n_steps, n_atmodeller
    one (n, track_points) array per key of model.TRACK_KEYS (see model.make_track), float32 except the atom
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


def pick_samples(run, n, seed = 0):
    '''
    (thetas (k, ndim), is_map (k,), blobs (k,)): the MAP of the chain, then min(n, N) samples of the posterior
    (sampler.posterior) drawn without replacement; thetas are physical (sampler.physical_samples).
    '''
    lp = run['log_prob']
    i_step, i_walker = np.unravel_index(np.argmax(lp), lp.shape)
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
    Log-spaced times [yr] from t0 to the largest age among thetas, plus every theta's age, so that each track
    ends on the grid at the state the likelihood compared with the data (between two log-spaced times a steep
    quantity, e.g. the outflow H fraction falling by 0.5 dex in the last 100 Myr, would otherwise end early).
    '''
    ages = [prm.theta_to_params(theta, cfg)['age'] for theta in thetas]
    return np.union1d(np.geomspace(cfg['model']['t0'], max(ages), int(n_points)), ages)


def _run_tasks(pool, tasks, log):
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
    Re-runs n posterior samples (default sampler.tracks) plus the MAP of the run in run_dir and writes
    tracks.npz (see the module docstring); returns its path. pool: a running WorkerPool of the same cfg (the
    sampler's), else a new one with `workers` workers.
    '''
    run = sampler.load_run(run_dir)
    cfg, run_dir = run['cfg'], run['run_dir']
    log = sampler.RunLog(run_dir)
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
        out[key] = np.stack([r['track'][key] for r in results])
    path = os.path.join(run_dir, TRACKS)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)

    # the re-runs repeat the chain's evaluations (save_molecules does not change the physics)
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
    The arrays of <run_dir>/tracks.npz as a dict.
    '''
    with np.load(os.path.join(run_dir, TRACKS), allow_pickle = False) as d:
        return {key: d[key] for key in d.files}
