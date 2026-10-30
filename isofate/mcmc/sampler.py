'''
The MCMC run: emcee's EnsembleSampler over a WorkerPool of model.log_prob, with prescan initialization, atomic
checkpoints and resume, an autocorrelation-time stopping rule, an optional stuck-walker reset and the summary.

A run directory, <output_root>/<name>/mcmc_<YYYYmmdd-HHMMSS>/, holds
    config.json      the resolved cfg (config.load_resolved reads it back)
    config.toml      a copy of the TOML, with the --set overrides listed at the top
    provenance.json  git commit and dirty flag, package versions, host, start time of each session
    run.log          the progress lines, also printed
    prescan.npz      the prior points of init = "prescan" (theta, lp, blobs)
    bisect.npz       the draws of init = "bisect" (search.save: brackets, final points, g, lp, blobs, history)
    solve_init.npz   the draws of init = "solve" (theta in the sampler's coordinates, lp, blobs)
    solve_table.npz  solve mode: the warm-start table of the solves (solve.py), written before the first step and
                     refitted to the chain after the steps of solve.refit_steps (each checkpoint holds the active one)
    chain.npz        the chain (NpzBackend), rewritten at most every checkpoint_every steps (and at most once a
                     minute, so that the saves cost at most 5% of the run)
    burnin.npz       the chain before the stuck-walker reset (sampler.reset_stuck_at > 0)
    summary.json     posterior summary (write_summary)
    tracks.npz       posterior evolutionary tracks (tracks.py), then the PNGs of plots.py
    worker_errors/   tracebacks of unexpected errors (status 5), one file per worker

Steps are counted over the whole run: after a reset at step k, chain.npz holds steps k+1, k+2, ... and
max_steps still counts from the start.

In solve mode (solve.py) the chain's coordinates are the sampler's (u_<observable> in place of the solved
parameter; config.sampled_names); the physical value of the solved parameter is the blob field solve_value, and the
summary, the figures and the tracks report the physical parameters (physical_samples).

A session holds <run_dir>/.lock (pid and host) while it runs, so a second `run --resume` of the same directory is
refused instead of interleaving its checkpoints with the first.

The parent process never imports jax, atmodeller or the coupled model; only the pool's workers do.
'''

import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

import emcee
import numpy as np

from isofate.mcmc import __version__, config, likelihood, model, search, solve
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool, worker_rss_gb

nan, inf = math.nan, math.inf
CHAIN, BURNIN, PRESCAN, BISECT, SUMMARY = 'chain.npz', 'burnin.npz', 'prescan.npz', 'bisect.npz', 'summary.json'
SOLVE_INIT, LOCK = 'solve_init.npz', '.lock'
SOLVE_INIT_FACTOR = 3 # init = "solve": draws per batch, per walker ...
SOLVE_INIT_BATCHES = 5 # ... and at most this many batches
JITTER_SCALE = 1e-3 # of the prior box: jitter of the starting points and of the reset copies
JITTER_DLP = 5.0 # a jittered point may lose at most this much lp against the unjittered one ...
JITTER_HALVINGS = 8 # ... else its jitter is halved and drawn again, at most this often
MIN_FREE_GB, LOW_FREE_GB = 1.0, 3.0 # [GB] refuse to start below / warn below
STUCK_ACCEPTANCE = 0.05 # a walker accepting less than this is stuck ...
STUCK_RELATIVE = 0.2 # ... if it also accepts less than this fraction of the median walker (or nothing at all)
STUCK_AFTER = 50 # [steps] progress lines count stuck walkers from this chain length on
RESET_DLP = 20.0 # the reset also replaces walkers with lp < median(lp) - RESET_DLP
PERCENTILES = (2.5, 16, 50, 84, 97.5)
TAU_WINDOW_C = 5.0 # emcee's integrated_time window factor c (M = c tau), used by tau_noise
SAMPLER_TAU_CHECKS = config.SAMPLER_DEFAULTS['tau_checks'] # for configs saved before sampler.tau_checks existed
CHECKPOINT_MIN_S = 60.0 # [s] periodic checkpoints at most this often ...
CHECKPOINT_COST = 20.0 # ... and at most every this many times the duration of the last save (<= 5% of the time)
STATUS = 1 + model.BLOB_FIELDS.index('status') # positions in a log_prob tuple
RUNTIME = 1 + model.BLOB_FIELDS.index('runtime')
SOLVE_EVALS = 1 + model.BLOB_FIELDS.index('solve_evals')
MAX_EVALS_HIST = 256 # histogram bins of solve_evals (model runs per evaluation)
N_STATUS = len(model.STATUS_NAMES)
BACKEND_KEYS = ('chain', 'log_prob', 'accepted', 'iteration', 'nwalkers', 'ndim', 'blobs', 'rs_name', 'rs_key',
                'rs_pos', 'rs_has_gauss', 'rs_cached')


###_____Checkpointing_____###

class NpzBackend(emcee.backends.Backend):
    '''
    emcee's in-memory Backend with an atomic .npz checkpoint (no h5py needed). save() writes the filled steps
    and the sampler's RNG state, so a sampler built on load(path) continues bit-identically.
    '''

    def __init__(self, dtype = None):
        super().__init__(dtype = dtype)
        self.iteration = 0 # emcee sets it in reset(); 0 before the sampler exists

    def save(self, path, **extra):
        it = self.iteration
        rs = self.random_state # ('MT19937', uint32[624], pos, has_gauss, cached_gaussian)
        out = dict(chain = self.chain[:it], log_prob = self.log_prob[:it], accepted = self.accepted,
                   iteration = it, nwalkers = self.nwalkers, ndim = self.ndim,
                   rs_name = np.array(rs[0]), rs_key = rs[1], rs_pos = rs[2], rs_has_gauss = rs[3],
                   rs_cached = rs[4])
        if self.blobs is not None:
            out['blobs'] = self.blobs[:it]
        out.update(extra)
        tmp = path + '.tmp.npz' # np.savez_compressed appends .npz unless present
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path) # atomic: a kill mid-write never corrupts the last good checkpoint

    @classmethod
    def load(cls, path):
        '''
        The backend saved at path; the extra arrays of save() are in .extra.
        '''
        with np.load(path, allow_pickle = False) as d:
            self = cls(dtype = d['chain'].dtype)
            self.reset(int(d['nwalkers']), int(d['ndim']))
            self.chain, self.log_prob = d['chain'].copy(), d['log_prob'].copy()
            self.accepted, self.iteration = d['accepted'].copy(), int(d['iteration'])
            self.blobs = d['blobs'].copy() if 'blobs' in d.files else None
            self.random_state = (str(d['rs_name']), d['rs_key'].copy(), int(d['rs_pos']),
                                 int(d['rs_has_gauss']), float(d['rs_cached']))
            self.extra = {key: d[key] for key in d.files if key not in BACKEND_KEYS}
        return self


def _get(extra, key, default):
    '''
    A checkpoint extra as a Python scalar (0-d arrays) or array, or default when absent.
    '''
    if key not in extra:
        return default
    value = extra[key]
    return value.item() if value.ndim == 0 else value


###_____Small helpers_____###

class RunLog:
    '''
    Prints a line and appends it, time-stamped, to <run_dir>/run.log.
    '''

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'run.log')

    def __call__(self, text = '', echo = True):
        if echo:
            print(text, flush = True)
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.path, 'a') as f:
            for line in str(text).splitlines() or ['']:
                f.write(f'{stamp}  {line}\n')


def duration_text(seconds):
    '''
    '45 s', '12m05s', '3h20m' (or '?' when unknown).
    '''
    if not math.isfinite(seconds):
        return '?'
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f'{seconds:.0f} s'
    if seconds < 3600:
        return f'{int(seconds//60)}m{int(seconds % 60):02d}s'
    return f'{int(seconds//3600)}h{int(seconds % 3600//60):02d}m'


def _status_index(status):
    return int(status) if math.isfinite(status) and 0 <= status < N_STATUS else int(model.UNEXPECTED)


def status_counts(statuses):
    '''
    Number of evaluations per status code 0-5.
    '''
    return np.bincount([_status_index(float(s)) for s in np.ravel(statuses)], minlength = N_STATUS)


def status_text(counts):
    '''
    'ok 120, outside prior 3' from counts per status code (non-zero ones only).
    '''
    parts = [f'{model.STATUS_NAMES[k]} {int(n)}' for k, n in enumerate(counts) if n]
    return ', '.join(parts) if parts else 'none'


def _read_json(path, default = None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _dump_json(obj, path):
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(f'{type(x).__name__} is not JSON serializable')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent = 1, default = default) # NaN and Infinity as Python's json writes them
        f.write('\n')
    os.replace(tmp, path)


def check_disk(path):
    '''
    Free space [GB] on the disk holding path (or its nearest existing parent). Raises below 1 GB.
    '''
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free/1e9 # [GB]
    if free < MIN_FREE_GB:
        raise RuntimeError(f'only {free:.2f} GB free on the disk of {path}; a run needs at least {MIN_FREE_GB:g} GB. '
                           'Free some space first.')
    return free


def provenance():
    '''
    Git commit and dirty flag of the IsoFATE repo, package versions (read from the installed metadata, so the
    parent never imports jax or atmodeller), Python, host and time.
    '''
    import isofate
    repo = os.path.dirname(os.path.dirname(os.path.abspath(isofate.__file__)))
    git = {'repo': repo, 'commit': None, 'dirty': None, 'status': None}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS = '0') # git status must not write the index
    try:
        head = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output = True, text = True,
                              timeout = 20, env = env)
        if head.returncode == 0:
            status = subprocess.run(['git', '-C', repo, 'status', '--porcelain'], capture_output = True,
                                    text = True, timeout = 20, env = env).stdout.splitlines()
            git.update(commit = head.stdout.strip(), dirty = bool(status), status = status[:50])
    except (OSError, subprocess.SubprocessError):
        pass
    versions = {'isofate.mcmc': __version__}
    for package in ('isofate', 'emcee', 'numpy', 'scipy', 'jax', 'atmodeller', 'numba'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'git': git, 'versions': versions,
            'python': sys.version.split()[0], 'platform': platform.platform(), 'host': platform.node()}


def make_moves(name):
    '''
    emcee moves of sampler.moves: "de" = DE 0.7 + DE(gamma0 = 1, mode jumps) 0.1 + DESnooker 0.2,
    "de_only" = DE 0.9 + DE(gamma0 = 1) 0.1, "stretch" = the affine-invariant stretch move.
    '''
    if name == 'stretch':
        return [(emcee.moves.StretchMove(), 1.0)]
    if name == 'de_only':
        return [(emcee.moves.DEMove(), 0.9), (emcee.moves.DEMove(gamma0 = 1.0), 0.1)]
    return [(emcee.moves.DEMove(), 0.7), (emcee.moves.DEMove(gamma0 = 1.0), 0.1),
            (emcee.moves.DESnookerMove(), 0.2)]


def stuck_mask(acceptance):
    '''
    Mask of the stuck walkers: acceptance < min(0.05, 0.2 x the median walker's), or no accepted move at all. The
    threshold is relative because on some posteriors every walker accepts only 1-3% of its proposals and still
    mixes (e.g. emcee's moves on the LHS 1140 b ridge without solve mode, with tau of 1e3-2e3 steps): an absolute
    0.05 would call all of them stuck, and tau and the convergence test would never use any.
    '''
    acceptance = np.asarray(acceptance, dtype = float)
    if not acceptance.size:
        return np.zeros(0, dtype = bool)
    threshold = min(STUCK_ACCEPTANCE, STUCK_RELATIVE*float(np.median(acceptance)))
    return (acceptance < threshold) | (acceptance <= 0)


def stuck_walkers(acceptance, lp):
    '''
    Mask of the walkers a reset replaces: stuck (stuck_mask), lp below median(lp) - 20, or lp not finite.
    '''
    acceptance, lp = np.asarray(acceptance, dtype = float), np.asarray(lp, dtype = float)
    finite = np.isfinite(lp)
    median = np.median(lp[finite]) if finite.any() else -inf
    bad = stuck_mask(acceptance) | ~finite
    bad[finite] |= lp[finite] < median - RESET_DLP
    return bad


def behind_walkers(lp):
    '''
    Mask of the walkers whose lp is more than 20 below the median (or not finite): lost in another region, e.g.
    the flat stripped-core plateau, where they still accept moves and so are not "stuck" by acceptance.
    '''
    lp = np.asarray(lp, dtype = float)
    finite = np.isfinite(lp)
    if not finite.any():
        return ~finite
    return ~finite | (np.where(finite, lp, 0.0) < np.median(lp[finite]) - RESET_DLP)


def autocorr_info(chain, acceptance):
    '''
    (tau, steps, walkers): the integrated autocorrelation time [steps] of each coordinate over the last 2/3 of chain
    (nsteps, nwalkers, ndim), using only walkers that are not stuck (stuck_mask) and moved in that window (a frozen
    walker makes the estimate NaN), and the steps and walkers it used. tau is NaN if impossible.
    '''
    n, nwalkers, ndim = chain.shape
    if n < 3:
        return np.full(ndim, nan), 0, 0
    window = chain[n//3:]
    good = ~stuck_mask(acceptance) & np.all(np.ptp(window, axis = 0) > 0, axis = 1)
    if not good.any():
        return np.full(ndim, nan), len(window), 0
    try:
        with np.errstate(all = 'ignore'):
            tau = emcee.autocorr.integrated_time(window[:, good], tol = 0)
    except Exception:
        return np.full(ndim, nan), len(window), int(good.sum())
    tau = np.asarray(tau, dtype = float).reshape(ndim)
    tau[~(tau > 0)] = nan # 0 or negative: no estimate (too few steps or walkers)
    return tau, len(window), int(good.sum())


def autocorr_time(chain, acceptance):
    '''
    autocorr_info's tau alone.
    '''
    return autocorr_info(chain, acceptance)[0]


def tau_noise(tau, steps, walkers):
    '''
    Relative 1-sigma statistical error of each tau estimate: sqrt(2 (2M + 1)/(steps walkers)) with emcee's window
    M = 5 tau (Sokal 1997; the walkers' autocorrelation functions are averaged, counted here as independent). On
    independent AR(1) walkers the scatter of tau over replicate ensembles is 0.75 times this (tests); on the LHS
    1140 b chain of 2026-10-29, whose walkers are not independent, the spread of tau between random halves of the
    walkers was 0.6-1 times it. NaN without an estimate.
    '''
    tau = np.asarray(tau, dtype = float)
    if steps <= 0 or walkers <= 0:
        return np.full(tau.shape, nan)
    with np.errstate(all = 'ignore'):
        return np.sqrt(2*(2*TAU_WINDOW_C*tau + 1)/(steps*walkers))


def tau_rule(tau, previous, noise, n, s, stable_before, complete = True):
    '''
    One convergence check. tau stable: every walker is in the estimate (complete), and every coordinate's tau
    changed since the previous check by less than max(tau_rtol, its statistical noise). Converged: n >= min_steps,
    n > n_tau*max(tau), and tau stable at this check and the tau_checks - 1 before it (stable_before: the stable
    flags of the earlier checks, oldest first). Returns (stable, converged, change, tol): change and tol of the
    coordinate closest to failing (NaN without a previous check).

    Why complete: autocorr_info leaves out stuck walkers and walkers that did not move in its window, and the
    noise grows as 1/sqrt(walkers used), so 4 moving walkers of 24 get a tolerance of about 40% and pass. The frozen
    ones are still in the posterior samples (posterior() uses every walker), so such a check must not count, however
    long the run; sampler.reset_stuck_at is the remedy.
    Limits (simulations of 2026-10-29): the estimate is biased low, by about 15-30% at N = 50 tau, so at the stop
    N/tau_true is about 40-45 rather than 50; from N of about 1000 on, successive estimates share most of their
    window and the stability test almost always passes on a stationary chain, so the stop is then set mainly by
    n > n_tau*max(tau); and a slow component that the 5 tau window cannot see (rare switches between regions, a
    plateau) is not detected at all. A "converged" flag therefore deserves a look at traces.png.
    '''
    tau, noise = np.asarray(tau, dtype = float), np.asarray(noise, dtype = float)
    finite = bool(np.isfinite(tau).all())
    if previous is None or not finite or not np.isfinite(previous).all():
        return False, False, nan, nan
    with np.errstate(all = 'ignore'):
        change = np.abs(tau - previous)/tau # relative to the newer estimate
        tol = np.maximum(s['tau_rtol'], np.nan_to_num(noise, nan = 0.0))
        worst = int(np.argmax(change/tol))
    stable = bool(complete and np.all(change < tol))
    k = int(s.get('tau_checks', SAMPLER_TAU_CHECKS))
    recent = list(stable_before)[-(k - 1):] if k > 1 else [] # the k - 1 checks before this one
    steady = stable and len(recent) == k - 1 and all(recent)
    converged = bool(steady and n >= s['min_steps'] and n > s['n_tau']*tau.max())
    return stable, converged, float(change[worst]), float(tol[worst])


def check_text(step, tau, noise, walkers, nwalkers, n, s, previous, stable, converged, change, tol):
    '''
    The run.log line of one convergence check (_Run.check_convergence); walkers: those the estimate used,
    previous: the tau of the check before (None at the first).
    '''
    tau, noise = np.asarray(tau, dtype = float), np.asarray(noise, dtype = float)
    known = noise[np.isfinite(noise)]
    spread = f'+-{100*known.max():.2g}%' if known.size else 'no estimate' # nanmax of all-NaN warns
    finite = bool(np.isfinite(tau).all())
    n_over_tau = n/tau.max() if finite else nan
    k = int(s.get('tau_checks', SAMPLER_TAU_CHECKS))
    if math.isfinite(change):
        verdict = 'stable' if stable else 'not stable'
        if walkers < nwalkers:
            verdict += (f', {nwalkers - walkers} of {nwalkers} walkers left out of tau (stuck or frozen): a check '
                        'counts only with every walker')
        change_text = (f'change {100*change:.3g}% (needs < {100*tol:.3g}%, max of tau_rtol and the noise: '
                       f'{verdict}; {k} in a row needed)')
    else:
        why = 'no estimate of tau' if not finite else 'first check' if previous is None else \
            'no estimate at the previous check'
        change_text = f'change - ({why}: not stable)'
    return (f'step {step}: tau = {", ".join(f"{t:.3g}" for t in tau)} steps ({spread}; {walkers}/{nwalkers} '
            f'walkers); N/tau = {n_over_tau:.3g} (needs > {s["n_tau"]:g}), {change_text}, N = {n} (needs >= '
            f'{s["min_steps"]})' + ('  -> converged' if converged else ''))


def tau_bookkeeping(extra, ndim):
    '''
    The per-check record of a checkpoint's extras, every list as long as steps: steps, tau (one array per check),
    stable, change, tol (of the coordinate closest to failing) and walkers (those the estimate used). Checkpoints
    written before the noise-aware rule (2026-10-29) have steps and tau only: their checks get stable False, change
    and tol NaN and walkers -1 (unknown), padded at the front so that the lists stay aligned by index (tau_rule then
    sees no stable check before the first new one, as it saw none before).
    '''
    steps = [int(i) for i in _get(extra, 'tau_steps', [])]
    tau = [np.array(t, dtype = float) for t in _get(extra, 'tau', np.empty((0, ndim)))]
    k = len(steps)

    def aligned(key, fill, kind):
        values = [kind(x) for x in _get(extra, key, [])][-k:] if k else []
        return [fill]*(k - len(values)) + values
    return {'steps': steps, 'tau': tau, 'stable': aligned('tau_stable', False, bool),
            'change': aligned('tau_change', nan, float), 'tol': aligned('tau_tol', nan, float),
            'walkers': aligned('tau_walkers', -1, int)}


def run_derivative(cfg, extra):
    '''
    The solve.derivative of a solve-mode run: its config's; for a config saved before the option existed
    (2026-10-29), the scheme its last session used, so that a resume continues the chain's own target density:
    the checkpoint's 'derivative', else "forward" when the checkpoint holds a warm-start table (only code with the
    forward default wrote one; e.g. the final LHS 1140 b chain, forward from its step 251 on), else "central".
    '''
    if 'derivative' in cfg['solve']:
        return cfg['solve']['derivative']
    if 'derivative' in extra:
        return str(_get(extra, 'derivative', solve.OLD_DERIVATIVE))
    return 'forward' if 'table_version' in extra else solve.OLD_DERIVATIVE


def burn_and_thin(tau, n, burn_factor):
    '''
    (discard, thin): discard = ceil(burn_factor*max tau) capped at n/2, thin = floor(0.5*min tau), at least 1
    and leaving at least one step. Without a finite tau: half the chain, no thinning.
    '''
    tau = np.asarray(tau, dtype = float)
    if tau.size and np.isfinite(tau).all() and tau.max() > 0:
        discard = min(max(math.ceil(burn_factor*tau.max()), 0), n//2)
        thin = max(1, math.floor(0.5*tau.min()))
    else:
        discard, thin = n//2, 1
    return discard, max(1, min(thin, n - discard))


###_____Evaluations through the pool_____###

class _CountingPool:
    '''
    The pool emcee sees: WorkerPool.map, tallying the status, the isocalc time and the model runs (solve_evals, a
    histogram) of every evaluation, rejected proposals included (the chain's blobs only keep accepted ones). phase
    selects the tally.
    '''

    def __init__(self, pool):
        self.pool, self.phase = pool, 'init'
        self.counts = {'init': np.zeros(N_STATUS, dtype = np.int64), 'sampling': np.zeros(N_STATUS, dtype = np.int64)}
        self.runtime = {'init': 0.0, 'sampling': 0.0} # [s]
        self.evals = {phase: np.zeros(MAX_EVALS_HIST, dtype = np.int64) for phase in ('init', 'sampling')}

    def map(self, f, iterable):
        results = self.pool.map(f, iterable)
        counts, evals = self.counts[self.phase], self.evals[self.phase]
        for r in results:
            counts[_status_index(r[STATUS])] += 1
            self.runtime[self.phase] += r[RUNTIME]
            k = r[SOLVE_EVALS]
            evals[min(int(k), MAX_EVALS_HIST - 1) if math.isfinite(k) else 0] += 1
        return results

    def unexpected(self):
        return int(self.counts['init'][int(model.UNEXPECTED)] + self.counts['sampling'][int(model.UNEXPECTED)])


evaluate_many = model.evaluate_many # (lp, blobs) of model.log_prob at thetas through a pool


def _no_start_advice(n_finite, n_needed, what):
    return (f'{what}: only {n_finite} points with a finite posterior, {n_needed} walkers needed. Check the model '
            'against the data at a sensible point (python -m isofate.mcmc check CONFIG --theta ...), widen the priors, '
            'raise sampler.prescan_factor, or start from sampler.init = "ball" around a good params.<name>.value.')


def safe_jitter(cfg, pool, coords, lp, blobs, rng, others = None, scale = JITTER_SCALE):
    '''
    Jittered copies of the points coords (n, ndim), whose lp and blobs are known: each is moved by scale x the
    prior box (params.jitter) and re-evaluated. When its lp falls more than 5 below the unjittered point's, the
    jitter is halved and drawn again, at most 8 times, so starts on a thin ridge stay on it. A point that still
    loses more keeps its unjittered position if that is distinct (no other row of coords or of others, e.g. the
    walkers a reset keeps, sits there); a copy that is not distinct takes its best try with a finite lp, or as
    the last resort stays an exact copy.
    Returns (coords, lp, blobs, {'halved': points whose jitter was halved, 'unjittered', 'copies'}).
    '''
    coords, lp, blobs = np.array(coords, dtype = float), np.asarray(lp, dtype = float), np.asarray(blobs)
    new = prm.jitter(coords, cfg, rng, scale = scale)
    lp_new, blobs_new = evaluate_many(pool, new)
    best = (new.copy(), lp_new.copy(), blobs_new.copy()) # the best try of each point
    bad = ~(lp_new >= lp - JITTER_DLP)
    halved = bad.copy()
    for k in range(1, JITTER_HALVINGS + 1):
        idx = np.flatnonzero(bad)
        if not idx.size:
            break
        trial = prm.jitter(coords[idx], cfg, rng, scale = scale/2**k)
        lp_t, blobs_t = evaluate_many(pool, trial)
        new[idx], lp_new[idx], blobs_new[idx] = trial, lp_t, blobs_t
        better = idx[lp_t > best[1][idx]]
        best[0][better], best[1][better], best[2][better] = new[better], lp_new[better], blobs_new[better]
        bad[idx] = ~(lp_t >= lp[idx] - JITTER_DLP)
    everyone = coords if others is None else np.vstack([coords, np.asarray(others, dtype = float)])
    n_unjittered = n_copies = 0
    for i in np.flatnonzero(bad):
        if np.sum(np.all(everyone == coords[i], axis = 1)) <= 1: # only itself
            new[i], lp_new[i], blobs_new[i] = coords[i], lp[i], blobs[i]
            n_unjittered += 1
        elif np.isfinite(best[1][i]):
            new[i], lp_new[i], blobs_new[i] = best[0][i], best[1][i], best[2][i]
        else:
            new[i], lp_new[i], blobs_new[i] = coords[i], lp[i], blobs[i]
            n_copies += 1
    return new, lp_new, blobs_new, {'halved': int(halved.sum()), 'unjittered': n_unjittered, 'copies': n_copies}


def _jitter_text(info, n):
    text = f'jittered by {JITTER_SCALE:g} of the prior box'
    if info['halved']:
        text += (f"; {info['halved']} of {n} lost more than {JITTER_DLP:g} in lp and had the jitter halved (up to "
                 f"{JITTER_HALVINGS} times)")
        if info['unjittered']:
            text += f", {info['unjittered']} kept their unjittered position"
        if info['copies']:
            text += f", {info['copies']} stayed exact copies"
    return text


def init_prescan(cfg, pool, nwalkers, rng, run_dir, log):
    '''
    Starting walkers: prescan_factor*nwalkers Latin-hypercube prior points (saved to prescan.npz), nwalkers of
    the best 2*nwalkers finite ones chosen at random, jittered by 1e-3 of the prior box (safe_jitter: halved
    while the lp drops by more than 5, else unjittered) and re-evaluated.
    Returns (coords, lp, blobs).
    '''
    n = cfg['sampler']['prescan_factor']*nwalkers
    start = time.perf_counter()
    thetas = prm.sample_prior(cfg, n, rng)
    lp, blobs = evaluate_many(pool, thetas)
    finite = np.flatnonzero(np.isfinite(lp))
    log(f'prescan: {n} Latin-hypercube prior points in {time.perf_counter() - start:.1f} s, {finite.size} finite '
        f'({status_text(status_counts(blobs["status"]))})')
    if finite.size < nwalkers:
        np.savez_compressed(os.path.join(run_dir, PRESCAN), theta = thetas, lp = lp, blobs = blobs,
                            free = np.array(cfg['free']))
        raise RuntimeError(_no_start_advice(finite.size, nwalkers, f'prescan of {n} prior points'))
    best = finite[np.argsort(-lp[finite], kind = 'stable')][:2*nwalkers]
    source = rng.permutation(best)[:nwalkers]
    coords, lp_w, blobs_w, info = safe_jitter(cfg, pool, thetas[source], lp[source], blobs[source], rng)
    np.savez_compressed(os.path.join(run_dir, PRESCAN), theta = thetas, lp = lp, blobs = blobs,
                        free = np.array(cfg['free']), start = source)
    log(f'walkers start from {nwalkers} of the best {best.size} prescan points (lp {lp[best].max():.4g} to '
        f'{lp[best].min():.4g}), {_jitter_text(info, nwalkers)}')
    return coords, lp_w, blobs_w


def init_bisect(cfg, pool, nwalkers, rng, run_dir, log, warn = None):
    '''
    Starting walkers on the surface where sampler.bisect_observable fits its data (search.py): batches of
    Latin-hypercube prior draws are bisected along sampler.bisect_param (saved to bisect.npz), and nwalkers of
    the 2*nwalkers converged points with the highest lp are chosen at random. They are not jittered: they are
    distinct already, and the surface is thinner than any sensible jitter. With fewer than nwalkers converged
    after 3 batches, sampler.bisect_fallback = true falls back to init_prescan with a warning; false raises.
    Returns (coords, lp, blobs).
    '''
    result = search.bisect_points(cfg, pool, nwalkers, rng, log)
    search.save(result, os.path.join(run_dir, BISECT))
    ok = search.good(result)
    if ok.size < nwalkers:
        text = search.failure_text(result, nwalkers)
        if not cfg['sampler']['bisect_fallback']:
            raise RuntimeError(text + '\n  (sampler.bisect_fallback = false; true falls back to init = "prescan")')
        (warn or log)('BISECTION FAILED: falling back to init = "prescan", so the walkers may start off the '
                      f'surface where {cfg["sampler"]["bisect_observable"]} fits; ' + text)
        return init_prescan(cfg, pool, nwalkers, rng, run_dir, log)
    pick = search.choose(result, nwalkers, rng)
    log(f'walkers start from {nwalkers} of the {min(ok.size, 2*nwalkers)} converged bisection points with the '
        f"highest lp (lp {result['lp'][pick].max():.4g} to {result['lp'][pick].min():.4g}; {ok.size} converged), "
        'not jittered')
    return result['point'][pick].copy(), result['lp'][pick].copy(), result['blobs'][pick].copy()


def solve_counts(lp, blobs):
    '''
    {ok: finite lp, no root: status 1, lp -inf: solved but ruled out by the likelihood, failed: everything else
    (model failures, solves that did not converge or have no usable derivative)} of solve-mode evaluations.
    '''
    status, solved = blobs['status'], np.isfinite(blobs['solve_value'])
    ok = np.isfinite(lp)
    no_root = status == model.OUTSIDE_PRIOR
    ruled_out = ~ok & ~no_root & solved
    return {'ok': int(ok.sum()), 'no root': int(no_root.sum()), 'lp -inf': int(ruled_out.sum()),
            'failed': int((~ok & ~no_root & ~solved).sum())}


def init_solve(cfg, pool, nwalkers, rng, run_dir, log):
    '''
    Starting walkers of solve mode (init = "solve"): batches of 3*nwalkers draws (params.sample_solve: a Latin
    hypercube with the other free parameters from their priors and u from the data distribution of the solve
    observable), each solved from both prior bounds of the solved parameter, until 2*nwalkers have a finite lp (at
    most 5 batches). nwalkers of the best 2*nwalkers are chosen at random, without replacement, and not jittered
    (the draws are distinct in every coordinate, u included). Saves solve_init.npz, then fits the warm-start table
    to every converged solve and writes solve_table.npz. Returns (coords, lp, blobs).
    '''
    sv = cfg['solve']
    parts = []
    start = time.perf_counter()
    for batch in range(1, SOLVE_INIT_BATCHES + 1):
        thetas = prm.sample_solve(cfg, SOLVE_INIT_FACTOR*nwalkers, rng)
        lp, blobs = evaluate_many(pool, thetas)
        parts.append((thetas, lp, blobs))
        thetas_all, lp_all, blobs_all = (np.concatenate([p[k] for p in parts]) for k in range(3))
        c = solve_counts(lp_all, blobs_all)
        runs = int(np.nansum(blobs_all['solve_evals']))
        log(f"solve init (batch {batch}): {len(lp_all)} draws{' so far' if batch > 1 else ''} of u_{sv['observable']} "
            f"solved for {sv['param']}: ok {c['ok']}, no root {c['no root']}, lp = -inf {c['lp -inf']}, failed "
            f"{c['failed']}; {runs} model runs ({runs/len(lp_all):.3g} per draw) in "
            f'{time.perf_counter() - start:.1f} s')
        if c['ok'] >= 2*nwalkers:
            break
    np.savez_compressed(os.path.join(run_dir, SOLVE_INIT), theta = thetas_all, lp = lp_all, blobs = blobs_all,
                        free = np.array(config.sampled_names(cfg)))
    finite = np.flatnonzero(np.isfinite(lp_all))
    if finite.size < nwalkers:
        raise RuntimeError(f"solve init: only {finite.size} of {len(lp_all)} draws have a finite posterior "
                           f"({c['no root']} without a root of {sv['param']} in its prior, {c['lp -inf']} ruled out by "
                           f"another observable, {c['failed']} failed), {nwalkers} walkers needed. Check where the "
                           'data lie with python -m isofate.mcmc check CONFIG (it prints a solve), and a scan of '
                           f"{sv['param']}; widen the prior of {sv['param']}, or choose another solve.observable or "
                           'solve.param')
    best = finite[np.argsort(-lp_all[finite], kind = 'stable')][:2*nwalkers]
    pick = rng.permutation(best)[:nwalkers]
    table = solve.fit_table(cfg, thetas_all, blobs_all['solve_value'], blobs_all['log_jac'])
    solve.save_table(table, os.path.join(run_dir, solve.TABLE))
    kind = 'warm-start table' if sv['warm_start'] else 'table (unused: solve.warm_start = false)'
    log(f'walkers start from {nwalkers} of the best {best.size} solved draws (lp {lp_all[best].max():.4g} to '
        f'{lp_all[best].min():.4g}), not jittered; {kind} of {len(table["p"])} solved points in '
        f"{solve.TABLE} (leave-one-out error of its guess of {'ln ' if table['log'] else ''}{sv['param']}: median "
        f"{np.median(np.abs(table['loo'])):.3g}; first expansion step {table['step']:.3g})")
    return thetas_all[pick].copy(), lp_all[pick].copy(), blobs_all[pick].copy()


def physical_samples(samples, blobs, cfg):
    '''
    The physical free parameters (n, ndim; cfg['free'] order) of sampler positions with their blobs: in solve mode
    the solved parameter from the blob field solve_value in place of u (NaN when the blobs lack it), else samples.
    '''
    samples = np.asarray(samples, dtype = float)
    if not config.solve_on(cfg):
        return samples
    has = blobs is not None and 'solve_value' in blobs.dtype.names
    return solve.physical(samples, np.asarray(blobs['solve_value'], dtype = float).reshape(samples.shape[:-1])
                          if has else nan, cfg)


def init_prior(cfg, pool, nwalkers, rng, log, max_rounds = 20):
    '''
    Starting walkers: random prior draws with a finite posterior. Returns (coords, lp, blobs).
    '''
    keep_theta, keep_lp, keep_blobs, n_tried = [], [], [], 0
    for _ in range(max_rounds):
        thetas = prm.sample_prior(cfg, 2*nwalkers, rng, lhs = False)
        lp, blobs = evaluate_many(pool, thetas)
        n_tried += len(thetas)
        ok = np.isfinite(lp)
        keep_theta.append(thetas[ok])
        keep_lp.append(lp[ok])
        keep_blobs.append(blobs[ok])
        if sum(len(x) for x in keep_lp) >= nwalkers:
            break
    lp = np.concatenate(keep_lp)
    if lp.size < nwalkers:
        raise RuntimeError(_no_start_advice(lp.size, nwalkers, f'{n_tried} random prior draws'))
    log(f'walkers start from {nwalkers} random prior draws with a finite posterior ({lp.size} of {n_tried})')
    return np.concatenate(keep_theta)[:nwalkers], lp[:nwalkers], np.concatenate(keep_blobs)[:nwalkers]


def init_ball(cfg, pool, nwalkers, rng, log, max_rounds = 10):
    '''
    Starting walkers: params.ball around the params' values (init_scale), redrawing points with lp = -inf.
    Returns (coords, lp, blobs).
    '''
    coords = prm.ball(cfg, nwalkers, rng)
    lp, blobs = evaluate_many(pool, coords)
    for _ in range(max_rounds):
        bad = np.flatnonzero(~np.isfinite(lp))
        if not bad.size:
            break
        coords[bad] = prm.ball(cfg, bad.size, rng)
        lp[bad], blobs[bad] = evaluate_many(pool, coords[bad])
    n_finite = int(np.isfinite(lp).sum())
    if n_finite < nwalkers:
        raise RuntimeError(_no_start_advice(n_finite, nwalkers, 'ball around the params values'))
    log(f'walkers start in a ball around the params values (lp {lp.max():.4g} to {lp.min():.4g})')
    return coords, lp, blobs


###_____Text_____###

def _data_text(o):
    '''
    The data of an observable config, e.g. '1.73 +- 0.025', '< 0.001 (0.3 dex)'.
    '''
    s = o['sigma']
    err = f'-{s[0]:.3g}/+{s[1]:.3g}' if isinstance(s, (tuple, list)) else f'+- {s:.3g}'
    if o['space'] == 'log10':
        err += ' dex'
    if o['sigma_model']:
        err += f' (+ model {o["sigma_model"]:.3g})'
    if o['kind'] == 'upper':
        return f'< {o["value"]:.4g}, width {err}'
    if o['kind'] == 'lower':
        return f'> {o["value"]:.4g}, width {err}'
    return f'{o["value"]:.4g} {err}'


def comparison_text(obs, terms, cfg):
    '''
    Table of the active observables: model value, data and log-likelihood term.
    '''
    width = max([13] + [len(name) for name in cfg['observables']])
    lines = [f'  {"observable":<{width}} {"model":>12}   {"data":<38} {"term":>10}']
    for name, o in cfg['observables'].items():
        unit = f' [{o["unit"]}]' if o['unit'] else ''
        lines.append(f'  {name:<{width}} {obs.get(name, nan):>12.5g}   {_data_text(o) + unit:<38} '
                     f'{terms.get(name, nan) + 0.0:>10.4g}') # + 0.0: no '-0'
    return '\n'.join(lines)


def _theta_text(theta, cfg):
    '''
    'name = value, ...' of a theta in the sampler's coordinates (config.sampled_names).
    '''
    return ', '.join(f'{name} = {x:.5g}' for name, x in zip(config.sampled_names(cfg), theta))


def _median_text(st):
    '''
    'median +up -down' of a _stats dict.
    '''
    return f"{st['p50']:>11.5g}  +{st['p84'] - st['p50']:.3g} -{st['p50'] - st['p16']:.3g}"


def summary_text(summary, cfg = None, full = False):
    '''
    Human-readable version of summary.json: sampling and convergence, the parameters (median and 68%). With
    full = True also the timing, evaluations, posterior of the active observables (against the data of cfg),
    the MAP's log-likelihood terms and the warnings.
    '''
    st = summary['settings']
    lines = [f"posterior of {summary['name']} ({summary['state']}): {summary['n_samples']} samples (discard "
             f"{summary['discard']}, thin {summary['thin']}, {summary['steps']} steps x {st['nwalkers']} walkers)"]
    tau, acc = summary['tau_max'], summary['acceptance']
    behind = acc.get('behind', []) # absent in summaries written before it existed
    lines.append(f"  tau max {tau:.3g} steps, N/tau {summary['n_over_tau']:.3g}, "
                 f"{'converged' if summary['converged'] else 'NOT converged'}; acceptance mean "
                 f"{acc['mean']:.2f}, {acc['n_stuck']} stuck walkers, {len(behind)} far behind "
                 f"(lp < median - {RESET_DLP:g})")
    if full:
        t, ev = summary['timing'], summary['evaluations']
        lines.append(f"  model: n_steps = {st['n_steps']}, n_atmodeller = {st['n_atmodeller']}, backend "
                     f"{st['backend']}; {st['workers']} workers, moves {st['moves']}, init {st['init']}, seed "
                     f"{st['seed']}")
        lines.append(f"  time: {duration_text(t['wall_s'])} in total, {t['s_per_step']:.3g} s/step, isocalc "
                     f"{t['isocalc_s_per_eval']:.3g} s/eval; {ev['total']} evaluations (sampling: "
                     f"{', '.join(f'{k} {n}' for k, n in ev['sampling'].items() if n) or 'none'})")
        if summary.get('reset_step'):
            lines.append(f"  stuck-walker reset at step {summary['reset_step']}: walkers "
                         f"{summary['replaced_walkers']} replaced; steps before it are in {BURNIN}")
    sv = summary.get('solve') # solve mode (absent in summaries written before it existed)
    if full and sv:
        e = sv['evals_sampling']
        lines.append(f"  solve mode: {sv['param']} solved from u = T({sv['observable']}) in [{sv['bounds'][0]:.4g}, "
                     f"{sv['bounds'][1]:.4g}]; model runs per solve (sampling, {e['n']} solves): mean {e['mean']:.3g}, "
                     f"median {e['p50']:.3g}, 90% {e['p90']:.3g}, max {e['max']:.3g}; {e['none']} proposals without "
                     'a model run (outside the prior)')
    lines.append('  parameters (median, 68% interval)' + (f": {sv['param']} from the solve (blob solve_value)"
                                                         if sv else '') + ':')
    for name, p in summary['params'].items():
        lines.append(f'    {name:<22} {_median_text(p)}')
    for name, p in (summary.get('sampled') or {}).items():
        if name not in summary['params']:
            lines.append(f'    {name + " (sampled)":<22} {_median_text(p)}')
    if full:
        lines.append('  observables (median, 68% interval; data):')
        for name in st['observables']:
            b = summary['blobs'][name]
            data = _data_text(cfg['observables'][name]) if cfg and name in cfg['observables'] else ''
            undefined = f"  ({100*b['frac_nan']:.0f}% undefined)" if b.get('frac_nan') else ''
            lines.append(f'    {name:<22} {_median_text(b)}   {data}{undefined}')
    m = summary['map']
    lines.append(f"  MAP: lp = {m['lp']:.5g} at step {m['step']}: "
                 + ', '.join(f'{k} = {v:.5g}' for k, v in m['theta'].items())
                 + ''.join(f', {k} = {v:.5g}' for k, v in (m.get('sampled') or {}).items() if k not in m['theta']))
    if full:
        lines.append('    log-likelihood terms: ' + ', '.join(f'{k} {v:.4g}' for k, v in m['terms'].items()))
        lines.append('  warnings:' + ('' if summary['warnings'] else ' none'))
        lines += [f'    - {w}' for w in summary['warnings']]
    return '\n'.join(lines)


###_____The run_____###

def _new_run_dir(cfg, run_dir):
    '''
    The (new, empty) directory of a fresh run; never an existing non-empty one. Without run_dir:
    <output_root>/<name>/mcmc_<stamp>, claimed atomically (another run started in the same second gets _1, _2, ...).
    '''
    if run_dir is None:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        return config.new_output_dir(os.path.join(cfg['output_root'], cfg['name'], f'mcmc_{stamp}'))
    return config.empty_output_dir(os.path.abspath(run_dir), 'a run (resume = True continues it)')


def acquire_lock(run_dir):
    '''
    Takes <run_dir>/.lock (pid and host) for this session. Raises FileExistsError when a live process on this host
    holds it; a stale lock (its process is gone) is taken over. Returns (path, note: None or a message).
    '''
    import psutil
    path, host, note = os.path.join(run_dir, LOCK), platform.node(), None
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(path) as f:
                    pid_text, _, other = f.read().strip().partition(' ')
                pid = int(pid_text)
            except (OSError, ValueError):
                pid, other = -1, ''
            if other == host and pid != os.getpid() and pid > 0 and psutil.pid_exists(pid):
                raise FileExistsError(f'{run_dir} is in use by another session (pid {pid} on {host}): wait for it to '
                                      f'finish or stop it first. If no run is going, delete {path}') from None
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            note = f'took over a stale lock of pid {pid} ({other or "unknown host"}): that session is gone'
            continue
        with os.fdopen(fd, 'w') as f:
            f.write(f'{os.getpid()} {host}\n')
        return path, note
    raise FileExistsError(f'could not take the lock {path}')


def release_lock(path):
    try:
        with open(path) as f:
            mine = f.read().split()[:1] == [str(os.getpid())]
        if mine:
            os.remove(path)
    except (OSError, TypeError):
        pass


def _copy_toml(cfg, path):
    '''
    config.toml: the original TOML text, preceded by the overrides applied on top of it.
    '''
    source = cfg.get('config_path')
    if not source or not os.path.exists(source):
        return False
    with open(source) as f:
        text = f.read()
    header = f'# copy of {source}\n'
    if cfg.get('overrides'):
        header += ('# run with these overrides on top (config.json has the resolved values):\n'
                   + ''.join(f'#     --set {o}\n' for o in cfg['overrides']))
    with open(path, 'w') as f:
        f.write(header + '\n' + text)
    return True


class _Run:
    '''
    One session of run_mcmc: set-up or resume, the pool, initialization, the sampling loop and the finish.
    '''

    def __init__(self, cfg, workers, run_dir, resume, max_steps):
        self.t_start = time.perf_counter()
        if resume:
            if run_dir is None:
                raise ValueError('resume = True needs the run_dir to continue')
            run_dir = os.path.abspath(run_dir)
            if cfg is None:
                cfg = config.load_resolved(os.path.join(run_dir, 'config.json'))
        elif cfg is None:
            raise ValueError('run_mcmc needs a cfg (config.load_config)')
        self.cfg, self.s, self.resume = cfg, cfg['sampler'], resume
        self.names, self.scfg = config.sampled_names(cfg), prm.sampled_cfg(cfg) # the sampler's coordinates
        self.seed, self.ndim = int(self.s['seed']), len(self.names)
        if self.ndim == 0:
            raise config.ConfigError('no free parameters to sample (set vary = true in a [params.NAME] table)')
        self.workers, self.max_steps_given = workers, max_steps is not None
        self.max_steps = int(self.s['max_steps'] if max_steps is None else max_steps)
        free_gb = check_disk(run_dir if resume else cfg['output_root'])

        if resume:
            self.run_dir = run_dir
            path = os.path.join(run_dir, CHAIN)
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found: this run has no checkpoint to resume; start a new run')
        else:
            self.run_dir = _new_run_dir(cfg, run_dir)
        self.lock, lock_note = acquire_lock(self.run_dir) # before writing anything into the directory
        try:
            self.log = RunLog(self.run_dir)
            if lock_note:
                self.log(f'note: {lock_note}')
            if resume:
                self.backend = NpzBackend.load(path)
                extra = self.backend.extra
                prov_path = os.path.join(run_dir, 'provenance.json')
                prov = _read_json(prov_path, {'created': None})
                prov.setdefault('resumed', []).append(provenance())
                _dump_json(prov, prov_path)
            else:
                config.save_resolved(cfg, os.path.join(self.run_dir, 'config.json'))
                _copy_toml(cfg, os.path.join(self.run_dir, 'config.toml'))
                _dump_json({'created': provenance(), 'resumed': []}, os.path.join(self.run_dir, 'provenance.json'))
                self.backend, extra = NpzBackend(), {}
        except BaseException:
            release_lock(self.lock)
            raise

        # bookkeeping, cumulative over sessions (chain.npz extras)
        self.offset = int(_get(extra, 'reset_step', 0)) # steps before the reset (0: none)
        self.replaced = [int(i) for i in _get(extra, 'replaced', [])]
        self.reset_pending = self.s['reset_stuck_at'] > 0 and self.offset == 0
        # per check: its step, tau, whether tau was stable, the change and tolerance of its worst coordinate and the
        # walkers the estimate used (tau_bookkeeping pads the checks of older checkpoints)
        bk = tau_bookkeeping(extra, self.ndim)
        self.tau_steps, self.tau_hist, self.tau_stable = bk['steps'], bk['tau'], bk['stable']
        self.tau_change, self.tau_tol, self.tau_walkers = bk['change'], bk['tol'], bk['walkers']
        self.converged = bool(_get(extra, 'converged', False))
        # the step of the first converged check (0: none); it keeps --max-steps extensions free of the stop even when a
        # later check says "not converged" (restore)
        self.converged_at = int(_get(extra, 'converged_at', self.tau_steps[-1] if self.converged and self.tau_steps
                                     else 0))
        self.stop = False
        self.nwalkers = self.backend.nwalkers if resume else None
        self.counts0 = {phase: np.array(_get(extra, f'counts_{phase}', np.zeros(N_STATUS)), dtype = np.int64)
                        for phase in ('init', 'sampling')}
        self.runtime0 = {phase: float(_get(extra, f'runtime_{phase}', 0.0)) for phase in ('init', 'sampling')}
        self.evals0 = {phase: np.array(_get(extra, f'evals_{phase}', np.zeros(MAX_EVALS_HIST)), dtype = np.int64)
                       for phase in ('init', 'sampling')} # histograms of the model runs per evaluation
        self.sampling_s = float(_get(extra, 'sampling_s', 0.0)) # [s] wall time in the sampling loop
        self.steps_timed = int(_get(extra, 'steps_timed', 0))
        self.init_s = float(_get(extra, 'init_s', 0.0)) # [s] preflight and initialization
        self.wall0 = float(_get(extra, 'wall_s', 0.0)) # [s] of the previous sessions
        self.warnings = [str(w) for w in _get(extra, 'run_warnings', [])]
        self.n_workers, self.warmup_s, self.pool = None, nan, None
        # the previous session's pool, kept by a checkpoint of a session that starts none (nothing to sample, or an
        # interrupt before the pool is up)
        self.workers0, self.warmup0 = int(_get(extra, 'workers', 0)), float(_get(extra, 'warmup_s', nan))
        self.notes = [] # config notes of this session (logged by execute)
        # solve mode: the active warm-start table and the step it was fitted after (0: the initialization's;
        # solve.NO_TABLE: none yet; None: a checkpoint from before refits existed), parsed from the checkpoint at once
        # so that a checkpoint written before restore_table runs keeps them; refit_done: the last refit step handled
        # (a refit that failed is not redone)
        self.table, self.table_version = None, solve.NO_TABLE
        self.refit_at, self.refit_done = [], 0
        if config.solve_on(cfg):
            if resume:
                self.table, self.table_version = solve.table_from_extras(extra)
                self.refit_done = int(_get(extra, 'refit_step', max(self.table_version or 0, 0)))
                self.resume_defaults(extra)
            self.refit_at = solve.refit_steps(self.cfg)
        if resume and 'tau_checks' not in self.s:
            self.notes.append(f"sampler.tau_checks is not in this run's config.json (saved before the option existed): "
                              f'the default {SAMPLER_TAU_CHECKS} applies, with the noise-aware tolerance of 2026-10-29')
        self.sampled = 0 # steps sampled in this session
        self.t_saved, self.save_s = time.perf_counter(), 0.0 # time of the last checkpoint and its duration [s]
        self.saved_step = self.step # the step in chain.npz
        if free_gb < LOW_FREE_GB:
            self._warn(f'only {free_gb:.1f} GB free on the disk of {self.run_dir}')

    def resume_defaults(self, extra):
        '''
        Solve-mode options that a config.json saved before they existed (2026-10-29) lacks, noted in run.log.
        solve.derivative goes into the session's cfg, which the workers get, as the scheme the chain's last session
        used (run_derivative), so a resume never switches the target density; solve.refit_steps takes its default
        (refits due by now are caught up in restore_table).
        '''
        sv = self.cfg['solve']
        if 'derivative' not in sv:
            derivative = run_derivative(self.cfg, extra)
            self.cfg = dict(self.cfg, solve = dict(sv, derivative = derivative))
            self.notes.append(f"solve.derivative is not in this run's config.json (saved before the option existed): "
                              f'the Jacobian uses the "{derivative}" difference, as in its last session')
        if 'refit_steps' not in sv:
            self.notes.append(f"solve.refit_steps is not in this run's config.json: the default "
                              f"{config.SOLVE_DEFAULTS['refit_steps']} applies")

    @property
    def step(self):
        '''
        Steps done over the whole run (including those before a reset).
        '''
        return self.offset + self.backend.iteration

    def _warn(self, text):
        self.warnings.append(text)
        self.log(f'warning: {text}')

    def extras(self):
        counts = {phase: self.counts0[phase] + self.pool.counts[phase] if self.pool else self.counts0[phase]
                  for phase in ('init', 'sampling')}
        runtime = {phase: self.runtime0[phase] + (self.pool.runtime[phase] if self.pool else 0.0)
                   for phase in ('init', 'sampling')}
        evals = {phase: self.evals0[phase] + self.pool.evals[phase] if self.pool else self.evals0[phase]
                 for phase in ('init', 'sampling')}
        solving = {}
        if config.solve_on(self.cfg):
            # the active table (or NO_TABLE), unless this session holds a checkpoint from before refits that it
            # has not restored yet (table_version None: write none, as that checkpoint did); the derivative scheme,
            # for resumes of configs without solve.derivative (run_derivative); the last refit step handled
            if self.table_version is not None:
                solving.update(solve.table_extras(self.table, self.table_version))
            solving.update(derivative = self.cfg['solve'].get('derivative', solve.OLD_DERIVATIVE),
                           refit_step = self.refit_done)
        return dict(solving, reset_step = self.offset, replaced = np.array(self.replaced, dtype = np.int64),
                    tau_steps = np.array(self.tau_steps, dtype = np.int64),
                    tau_stable = np.array(self.tau_stable, dtype = bool),
                    tau_change = np.array(self.tau_change, dtype = float),
                    tau_tol = np.array(self.tau_tol, dtype = float),
                    tau_walkers = np.array(self.tau_walkers, dtype = np.int64),
                    tau = np.array(self.tau_hist, dtype = float).reshape(-1, self.ndim),
                    converged = self.converged, converged_at = self.converged_at, max_steps = self.max_steps,
                    workers = self.n_workers or self.workers0,
                    warmup_s = self.warmup_s if self.n_workers else self.warmup0,
                    counts_init = counts['init'], counts_sampling = counts['sampling'],
                    runtime_init = runtime['init'], runtime_sampling = runtime['sampling'],
                    evals_init = evals['init'], evals_sampling = evals['sampling'],
                    sampling_s = self.sampling_s, steps_timed = self.steps_timed, init_s = self.init_s,
                    wall_s = self.wall0 + time.perf_counter() - self.t_start,
                    run_warnings = np.array(self.warnings, dtype = str))

    def checkpoint(self, name = CHAIN):
        '''
        Writes the backend and the bookkeeping to <run_dir>/<name> (nothing before the first step).
        '''
        if self.backend.iteration == 0:
            return False
        start = time.perf_counter()
        self.backend.save(os.path.join(self.run_dir, name), **self.extras())
        self.t_saved = time.perf_counter()
        self.save_s = self.t_saved - start
        if name == CHAIN:
            self.saved_step = self.step
        return True

    def checkpoint_due(self, step):
        '''
        Every checkpoint_every steps, but at most once a minute and at most every 20 times the duration of the
        last save: the file grows with the chain, and a save of 1e5 steps takes about 20 s.
        '''
        return (step % self.s['checkpoint_every'] == 0
                and time.perf_counter() - self.t_saved >= max(CHECKPOINT_MIN_S, CHECKPOINT_COST*self.save_s))

    def make_sampler(self):
        return emcee.EnsembleSampler(self.nwalkers, self.ndim, model.log_prob, pool = self.pool,
                                     moves = make_moves(self.s['moves']), backend = self.backend,
                                     blobs_dtype = model.blob_dtype())

    ###_____session_____###

    def execute(self, make_plots):
        cfg, log = self.cfg, self.log
        m = cfg['model']
        if self.resume:
            log(f'\nresuming {self.run_dir} at step {self.step} of {self.max_steps}')
        else:
            log(f'isofate.mcmc run of {cfg["name"]} in {self.run_dir}')
            log(f'  {self.ndim} free parameters ({", ".join(cfg["free"])}); observables: '
                f'{", ".join(cfg["observables"]) or "none"}')
            if config.solve_on(cfg):
                sv = cfg['solve']
                log(f"  solve mode: the sampler moves in {sv['name']} = T({sv['observable']}) in "
                    f"[{sv['bounds'][0]:.5g}, {sv['bounds'][1]:.5g}] (+-{sv['range_sigma']:g} sigma) in place of "
                    f"{sv['param']}, which each evaluation solves for (to {sv['tol']:g} sigma); coordinates: "
                    f"{', '.join(self.names)}")
        log(f'  model: backend {m["backend"]}, n_steps = {m["n_steps"]}, n_atmodeller = {m["n_atmodeller"]}, '
            f'composition {m["composition"]}')
        if config.solve_on(cfg): # said at every session, so that a change between sessions shows in run.log
            refits = ', '.join(map(str, self.refit_at)) or 'none'
            log(f"  solve: Jacobian by the {cfg['solve'].get('derivative', solve.OLD_DERIVATIVE)} difference "
                f'(solve.derivative); warm-start table refits after steps {refits} (solve.refit_steps)')
        for w in cfg['warnings']:
            log(f'  config warning: {w}')
        for note in cfg.get('notes', []) + self.notes:
            log(f'  config note: {note}')
        if self.resume and (self.step >= self.max_steps or (self.converged and self.s['stop_on_convergence']
                                                             and not self.max_steps_given)):
            self.nothing_to_sample()
            return
        phase, handled = 'start', False
        try:
            with WorkerPool(cfg, self.workers, run_dir = self.run_dir, verbose = False) as wp:
                try:
                    self.n_workers, self.warmup_s = wp.n, wp.warmup_s
                    self.pool = _CountingPool(wp)
                    log(f'pool: {wp.n} worker{"s" if wp.n > 1 else ""} ready in {wp.warmup_s:.1f} s, '
                        f'{worker_rss_gb(wp):.2f} GB resident')
                    phase = 'init'
                    start = time.perf_counter()
                    if self.resume:
                        self.restore()
                    else:
                        self.nwalkers = config.choose_nwalkers(cfg, wp.n)
                        self.preflight(wp)
                        self.initialize()
                    self.init_s += time.perf_counter() - start
                    phase = 'sampling'
                    self.pool.phase = 'sampling'
                    self.sample()
                    phase = 'finish'
                    self.finish(wp)
                except BaseException as e:
                    # save the chain before the pool is torn down: after a signal to the whole process group (kill
                    # -TERM -PGID) the workers are dead too, and the teardown can take WorkerPool.terminate's timeout
                    handled = True
                    self.failed(phase, e)
                    raise
        except BaseException as e:
            if not handled: # the pool did not start (or its teardown failed after a clean session)
                self.failed(phase, e)
            raise
        if make_plots:
            self.plots()
        log(f'done: {self.run_dir}')

    def failed(self, phase, error):
        '''
        The end of a session by an interrupt (Ctrl-C; SIGTERM and SIGHUP through __main__) or an error: the chain
        is checkpointed (an interrupt also writes the summary and prints the resume command; interrupted), an
        error's traceback goes to run.log.
        '''
        if isinstance(error, KeyboardInterrupt):
            self.interrupted(phase)
        elif isinstance(error, Exception):
            if phase in ('init', 'sampling') and self.checkpoint():
                self.log(f'the chain up to step {self.step} is saved in {os.path.join(self.run_dir, CHAIN)}')
            self.log(traceback.format_exc(), echo = False)

    def nothing_to_sample(self):
        '''
        A resume with nothing to sample: the chain has max_steps steps, or it converged and no --max-steps was
        given. Redoes a convergence check the previous session lost (catch_up_check) and writes the summary.
        '''
        if self.catch_up_check():
            self.checkpoint()
        if self.step >= self.max_steps:
            what = (f'the chain has {self.step} of max_steps = {self.max_steps} steps'
                    + (' and converged' if self.converged else '') + '; give a larger --max-steps to continue'
                    + (' (without the convergence stop)' if self.converged else ''))
        else:
            what = (f'the run converged (at step {self.converged_at or self.step}; {self.step} steps): give '
                    f'--max-steps N, N > {self.step}, to sample on without the convergence stop')
        self.log(f'nothing to sample: {what}')
        write_summary(self.run_dir)

    def preflight(self, wp):
        '''
        One evaluation at the centre (the params' values): fails clearly on a bug or a model failure there.
        '''
        cfg, log = self.cfg, self.log
        solving = config.solve_on(cfg)
        theta = prm.centre_theta(self.scfg) # in solve mode u at the datum
        res = wp.map(model.eval_task, [{'theta': theta, 'solve': solving}])[0]
        self.pool.counts['init'][_status_index(res['status'])] += 1
        status = int(res['status'])
        log(f'preflight at the centre ({_theta_text(theta, cfg)}): {model.STATUS_NAMES[status]}, isocalc '
            f'{res["runtime"]:.3g} s')
        if solving:
            sv = cfg['solve']
            log(f"  solve: {sv['param']} = {res['solve_value']:.7g}, ln|d{sv['param']}/du| = {res['log_jac']:.5g}, "
                f"{res['solve_evals']} model runs (from both prior bounds; warm starts need fewer)"
                + (f"; {res['error']}" if res['error'] else ''))
        if status == model.UNEXPECTED:
            raise RuntimeError(f'unexpected error at the centre ({res["error"]}): a bug or a bad config. The '
                               f'traceback is in {os.path.join(self.run_dir, "worker_errors")}; '
                               'python -m isofate.mcmc check CONFIG shows it in-process.')
        if status == model.MODEL_FAILED:
            raise RuntimeError(f'isocalc fails at the centre ({res["error"]}). Check the parameter values with '
                               'python -m isofate.mcmc check CONFIG.')
        log(comparison_text(res['obs'], res['terms'], cfg))
        log(f'  log_like = {res["log_like"]:.5g}, log_prior = {res["log_prior"]:.5g}')
        if status != model.OK:
            self._warn(f'the centre has status {status} ({model.STATUS_NAMES[status]}'
                       f'{": " + res["error"] if res["error"] else ""}); the initialization looks elsewhere')
        elif res['runtime'] > 0:
            per_step = res['runtime']*self.nwalkers/self.n_workers # [s] if every walker cost as much (solve: cold)
            log(f'  about {per_step:.3g} s/step ({3600/per_step:.0f} steps/h) with {self.nwalkers} walkers on '
                f'{self.n_workers} workers, if every evaluation costs as much as the centre')

    def initialize(self):
        cfg, log, init = self.cfg, self.log, self.s['init']
        scfg = self.scfg # prescan, prior and ball draw in the sampler's coordinates (u uniform on its support)
        rng = np.random.default_rng(self.seed)
        if init == 'solve':
            coords, lp, blobs = init_solve(cfg, self.pool, self.nwalkers, rng, self.run_dir, log)
        elif init == 'prescan':
            coords, lp, blobs = init_prescan(scfg, self.pool, self.nwalkers, rng, self.run_dir, log)
        elif init == 'bisect':
            coords, lp, blobs = init_bisect(cfg, self.pool, self.nwalkers, rng, self.run_dir, log, self._warn)
        elif init == 'prior':
            coords, lp, blobs = init_prior(scfg, self.pool, self.nwalkers, rng, log)
        else:
            coords, lp, blobs = init_ball(scfg, self.pool, self.nwalkers, rng, log)
        self.check_unexpected()
        path = os.path.join(self.run_dir, solve.TABLE)
        if config.solve_on(cfg) and os.path.exists(path):
            self.table, self.table_version = solve.load_table(path), 0
        np.random.seed(self.seed) # emcee seeds its RNG from the global one
        self.sampler = self.make_sampler()
        self.initial = emcee.State(coords, log_prob = lp, blobs = blobs,
                                   random_state = np.random.RandomState(self.seed).get_state())
        self.skip_check = False

    def restore(self):
        b, log = self.backend, self.log
        self.nwalkers = b.nwalkers
        if b.ndim != self.ndim:
            raise ValueError(f'chain.npz has {b.ndim} parameters but the config {self.ndim}')
        if b.blobs is None or b.blobs.dtype != np.dtype(model.blob_dtype()):
            n_fields = len(b.blobs.dtype.names or ()) if b.blobs is not None else 0
            raise ValueError(f'chain.npz was written with other blob fields ({n_fields}) than this version of '
                             f'isofate.mcmc ({len(model.BLOB_FIELDS)}), so it cannot be resumed; summary, plot and '
                             'compare still read it')
        if self.cfg['sampler']['nwalkers'] not in ('auto', self.nwalkers):
            log(f'  sampler.nwalkers is ignored on resume: the chain has {self.nwalkers} walkers')
        if self.reset_pending and self.step > self.s['reset_stuck_at']:
            self.reset_pending = False
            self._warn(f'the chain is past reset_stuck_at = {self.s["reset_stuck_at"]} without a reset; no reset')
        # an extension with --max-steps of a run that has converged samples on without the stop, also when a later
        # check of the extension said "not converged" (converged_at); a run that has not converged keeps the stop
        if (self.converged or self.converged_at) and self.s['stop_on_convergence'] and self.max_steps_given:
            log(f'the run converged at step {self.converged_at or self.step}; sampling on to {self.max_steps} steps '
                'without the convergence stop')
            self.s = dict(self.s, stop_on_convergence = False)
        self.sampler = self.make_sampler() # the RNG state comes from the backend
        self.initial = b.get_last_sample()
        self.skip_check = True
        # the work of the checkpoint's step that the previous session may not have done (the uninterrupted order:
        # refit, then check; a pending reset follows in sample)
        if config.solve_on(self.cfg):
            self.restore_table()
        self.catch_up_check()

    def restore_table(self):
        '''
        Solve mode: the warm-start table active at the checkpoint, written back to solve_table.npz for the workers
        (fresh in each session, they read it at their first evaluation). A checkpoint written while the run had no
        table (solve.NO_TABLE: init other than "solve", before the first refit) removes the file instead: a refit
        of a session that was then killed may have left one, and the workers must run cold, as they did. A
        checkpoint from before table refits existed holds no table: the file's (the initialization's). Then every
        refit due by the checkpoint's step that was not done is redone, once, from the chain up to that step (which
        is the checkpoint's, so the table is the uninterrupted run's): a refit that the end of the session cut short
        (an interrupt during build_table checkpoints the step with the old table), or the catch-up of a checkpoint
        from before refits existed.
        '''
        path = os.path.join(self.run_dir, solve.TABLE)
        if self.table_version is None: # from before refits existed
            self.table = solve.load_table(path) if os.path.exists(path) else None
            self.table_version = 0
            why = 'catch-up, the checkpoint predates table refits'
        else:
            if self.table is not None:
                solve.save_table(self.table, path)
            elif os.path.exists(path):
                os.remove(path)
            why = 'redone: the previous session ended before this refit was done'
        if any(self.refit_done < k <= self.step for k in self.refit_at):
            self.refit_table(why)

    def catch_up_check(self):
        '''
        Redoes the convergence check of the checkpoint's step when the session that wrote it did not run it: an
        interrupt during the check checkpoints the step without it (and the periodic checkpoints of the code before
        this one came before the check of their step). The check depends only on the chain and the acceptance up to
        that step, so the resumed run checks, and stops, where the uninterrupted one did. Returns True when it ran.
        '''
        b = self.backend
        if not b.iteration or self.step % self.s['check_every'] or (self.tau_steps and self.tau_steps[-1] == self.step):
            return False
        self.log(f'the convergence check of step {self.step} did not run in the previous session: running it now')
        self.check_convergence()
        return True

    def check_unexpected(self):
        n, limit = self.pool.unexpected(), self.s['max_unexpected_errors']
        if n > limit:
            raise RuntimeError(f'{n} unexpected errors (status 5: bugs, not model failures), more than '
                               f'sampler.max_unexpected_errors = {limit}. Tracebacks: '
                               f'{os.path.join(self.run_dir, "worker_errors")}')

    ###_____sampling loop_____###

    def reset_due(self):
        return self.reset_pending and self.step == self.s['reset_stuck_at'] and self.step < self.max_steps

    def sample(self):
        s, log = self.s, self.log
        if self.step < self.max_steps and not self.stop:
            log(f'sampling: {self.nwalkers} walkers x {self.ndim} parameters, steps {self.step + 1} to '
                f'{self.max_steps}, moves "{s["moves"]}", seed {self.seed}')
        self.mark = (time.perf_counter(), self.step, self.pool.counts['sampling'].copy(),
                     self.pool.runtime['sampling'], self.pool.evals['sampling'].copy())
        while self.step < self.max_steps and not self.stop:
            if self.reset_due():
                self.reset_walkers()
            t_last = time.perf_counter()
            for _ in self.sampler.sample(self.initial, iterations = self.max_steps - self.step,
                                         skip_initial_state_check = self.skip_check):
                now = time.perf_counter()
                self.sampling_s += now - t_last
                self.steps_timed += 1
                self.sampled += 1
                t_last = now
                step = self.step
                if step % s['print_every'] == 0 or step == self.max_steps:
                    self.progress()
                if step in self.refit_at and step > self.refit_done:
                    self.refit_table()
                # the check before the periodic checkpoint, so that the checkpoint holds it (a stop checkpoints in
                # finish); a check lost to an interrupt is redone at the resume (catch_up_check)
                stop = step % s['check_every'] == 0 and self.check_convergence()
                if not stop and self.checkpoint_due(step):
                    self.checkpoint()
                if stop:
                    break
                self.check_unexpected()
                if self.reset_due():
                    break

    def table_text(self):
        '''
        'no table (cold solves)', "the initialization's table" or 'the table of step k'.
        '''
        if self.table is None:
            return 'no table (cold solves)'
        return "the initialization's table" if not self.table_version else f'the table of step {self.table_version}'

    def refit_table(self, why = None):
        '''
        Solve mode: refits the warm-start table to the initialization's solves and the chain's solved states so
        far, between two steps (no evaluation is running), and replaces solve_table.npz; the workers reload it at
        their next evaluation. An error is a warning, and the active table stays: the table changes only the path
        of each solve, not its root, so a failed refit must not end the run (nor, redone at every resume, make it
        impossible to resume); refit_done marks it handled, so that a resume does not redo it from a longer chain.
        An interrupt is not caught: the step is then checkpointed with the old table, and the resume redoes the
        refit (restore_table).
        '''
        start = time.perf_counter()
        b, sv, step = self.backend, self.cfg['solve'], self.step
        before = np.median(np.abs(self.table['loo'])) if self.table is not None else None
        try:
            table = solve.build_table(self.cfg, self.run_dir, b.get_chain(), b.get_blobs())
            solve.save_table(table, os.path.join(self.run_dir, solve.TABLE))
        except Exception as e:
            self.refit_done = step
            self._warn(f'the warm-start table refit after step {step} failed ({type(e).__name__}: {e}); the solves go '
                       f'on with {self.table_text()} (the traceback is in run.log)')
            self.log(traceback.format_exc(), echo = False)
            return False
        self.table, self.table_version, self.refit_done = table, step, step
        source = 'initialization and chain' if os.path.exists(os.path.join(self.run_dir, SOLVE_INIT)) else 'chain only'
        was = f' (was {before:.3g})' if before is not None else ''
        self.log(f"warm-start table refitted after step {step}{f' ({why})' if why else ''}: "
                 f"{len(table['p'])} solved points ({source}), leave-one-out error of its guess of "
                 f"{'ln ' if table['log'] else ''}{sv['param']}: median {np.median(np.abs(table['loo'])):.3g}{was}; "
                 f"first expansion step {table['step']:.3g}; {time.perf_counter() - start:.1f} s")
        return True

    def progress(self):
        b = self.backend
        now = time.perf_counter()
        t0, step0, counts0, runtime0, evals0 = self.mark
        per_step = (now - t0)/(self.step - step0) if self.step > step0 else nan # [s]
        counts = self.pool.counts['sampling'] - counts0
        n_eval = int(counts.sum())
        evals = self.pool.evals['sampling'] - evals0
        runs = int(np.sum(np.arange(evals.size)*evals)) # model runs
        per_eval = (self.pool.runtime['sampling'] - runtime0)/runs if runs else nan # [s] isocalc only, per run
        acceptance = b.accepted/b.iteration
        lp = b.log_prob[b.iteration - 1]
        stuck = '-'
        if b.iteration >= STUCK_AFTER:
            stuck = f'{int(np.sum(stuck_mask(acceptance)))}/{self.nwalkers}, behind {int(np.sum(behind_walkers(lp)))}'
        solving = ''
        if config.solve_on(self.cfg):
            e = solve.evals_stats(evals)
            solving = f"model runs/solve mean {e['mean']:.3g} p50 {e['p50']:.0f} p90 {e['p90']:.0f}  "
        width = len(str(self.max_steps))
        self.log(f'step {self.step:>{width}}/{self.max_steps}  {per_step:.3g} s/step  acc {acceptance.mean():.2f}  '
                 f'lp max {lp.max():.5g} med {np.median(lp):.5g}  stuck {stuck}  {n_eval} evals: '
                 f'{status_text(counts)} {solving}(isocalc {per_eval:.3g} s/run)  '
                 f'ETA {duration_text(per_step*(self.max_steps - self.step))}')
        self.mark = (now, self.step, self.pool.counts['sampling'].copy(), self.pool.runtime['sampling'],
                     self.pool.evals['sampling'].copy())

    def check_convergence(self):
        '''
        Estimates tau and decides the stop (tau_rule): n >= min_steps, n > n_tau*max(tau), and tau stable, i.e.
        every walker in the estimate and tau changed by less than max(tau_rtol, its statistical noise), at
        tau_checks checks in a row (n: steps in the current chain).
        '''
        s, b = self.s, self.backend
        n = b.iteration
        acceptance = b.accepted/n
        tau, used, n_good = autocorr_info(b.chain[:n], acceptance)
        noise = tau_noise(tau, used, n_good)
        previous = self.tau_hist[-1] if self.tau_hist else None
        stable, self.converged, change, tol = tau_rule(tau, previous, noise, n, s, self.tau_stable,
                                                       complete = n_good == self.nwalkers)
        if self.converged and not self.converged_at:
            self.converged_at = self.step
        self.tau_steps.append(self.step)
        self.tau_hist.append(tau)
        self.tau_stable.append(stable)
        self.tau_change.append(change)
        self.tau_tol.append(tol)
        self.tau_walkers.append(n_good)
        self.log(check_text(self.step, tau, noise, n_good, self.nwalkers, n, s, previous, stable, self.converged,
                            change, tol))
        if self.converged and s['stop_on_convergence']:
            self.stop = True
        return self.stop

    def reset_walkers(self):
        '''
        The stuck-walker reset: saves the chain so far as burnin.npz, replaces stuck walkers by jittered copies
        of random good ones, and restarts with a fresh backend and sampler seeded from seed + 1.
        '''
        b, log, cfg = self.backend, self.log, self.cfg
        step = self.step
        self.checkpoint() # a resume from here redoes the same reset
        shutil.copyfile(os.path.join(self.run_dir, CHAIN), os.path.join(self.run_dir, BURNIN))
        it = b.iteration
        coords, lp, blobs = b.chain[it - 1].copy(), b.log_prob[it - 1].copy(), b.blobs[it - 1].copy()
        bad = np.flatnonzero(stuck_walkers(b.accepted/it, lp))
        good = np.setdiff1d(np.arange(self.nwalkers), bad)
        rng = np.random.default_rng(self.seed + 1)
        if bad.size and not good.size:
            self._warn(f'reset at step {step}: every walker looks stuck, none replaced')
            bad = bad[:0]
        elif bad.size > self.nwalkers//2:
            self._warn(f'reset at step {step}: {bad.size} of {self.nwalkers} walkers look stuck; copying the '
                       f'{good.size} others onto them would collapse the ensemble, so none is replaced (a posterior '
                       'thinner than the moves can follow, e.g. a thin ridge: see solve mode)')
            bad = bad[:0]
        jittered = ''
        if bad.size:
            source = rng.choice(good, bad.size)
            # copies of good walkers, jittered without falling off a thin ridge (safe_jitter)
            new, lp_new, blobs_new, info = safe_jitter(self.scfg, self.pool, coords[source], lp[source], blobs[source],
                                                       rng, others = coords[good])
            coords[bad], lp[bad], blobs[bad] = new, lp_new, blobs_new
            jittered = f', copies of random good walkers {_jitter_text(info, bad.size)}'
        log(f'reset at step {step}: {bad.size} of {self.nwalkers} walkers replaced (stuck: acceptance < '
            f'min({STUCK_ACCEPTANCE:g}, {STUCK_RELATIVE:g} x median), or lp < median - {RESET_DLP:g})'
            + (f': {bad.tolist()}' if bad.size else '')
            + f'{jittered}; steps 1-{step} saved as {BURNIN}, sampling restarts from a fresh chain')
        # the empty backend first: an interrupt in between must not save the old chain under the new offset
        self.backend = NpzBackend()
        self.offset, self.replaced, self.reset_pending = step, bad.tolist(), False
        self.tau_steps, self.tau_hist, self.converged, self.converged_at = [], [], False, 0
        self.tau_stable, self.tau_change, self.tau_tol, self.tau_walkers = [], [], [], []
        np.random.seed(self.seed + 1)
        self.sampler = self.make_sampler()
        self.initial = emcee.State(coords, log_prob = lp, blobs = blobs,
                                   random_state = np.random.RandomState(self.seed + 1).get_state())
        self.skip_check = True

    ###_____end of the session_____###

    def finish(self, wp):
        log = self.log
        self.checkpoint()
        why = 'converged' if self.stop else f'max_steps = {self.max_steps} reached'
        log(f'sampling finished at step {self.step} ({why}); sampling took {duration_text(self.sampling_s)}, '
            f'{self.sampling_s/max(self.steps_timed, 1):.3g} s/step')
        summary = write_summary(self.run_dir)
        log(summary_text(summary))
        for w in summary['warnings']: # the summary's own warnings too (e.g. not converged), not only the config's
            if w not in self.cfg['warnings'] and w not in self.warnings:
                log(f'warning: {w}')
        if self.s['tracks'] > 0 and (self.sampled or not os.path.exists(os.path.join(self.run_dir, 'tracks.npz'))):
            try:
                from isofate.mcmc import tracks
                tracks.compute_tracks(self.run_dir, n = self.s['tracks'], pool = wp)
            except Exception as e:
                log(f'tracks failed ({type(e).__name__}: {e}); the chain is safe. Retry with: python -m '
                    f'isofate.mcmc tracks {self.run_dir}')
                log(traceback.format_exc(), echo = False)

    def plots(self):
        try:
            from isofate.mcmc import plots
            paths = plots.plot_run(self.run_dir)
            self.log(f'plots: {", ".join(os.path.basename(p) for p in paths or [])}')
        except Exception as e:
            self.log(f'plots failed ({type(e).__name__}: {e}); the chain is safe. Retry with: python -m '
                     f'isofate.mcmc plot {self.run_dir}')
            self.log(traceback.format_exc(), echo = False)

    def interrupted(self, phase):
        log = self.log
        if phase == 'finish':
            if self.saved_step != self.step: # the interrupt cut short the checkpoint of finish
                self.checkpoint()
            log(f'\ninterrupted after sampling: the chain is complete in {os.path.join(self.run_dir, CHAIN)}. '
                f'Tracks and plots later with: python -m isofate.mcmc tracks {self.run_dir}; '
                f'python -m isofate.mcmc plot {self.run_dir}')
            return
        self.checkpoint()
        if not os.path.exists(os.path.join(self.run_dir, CHAIN)):
            log('\ninterrupted before the first step: nothing to resume; start a new run')
            return
        try:
            write_summary(self.run_dir, state = 'interrupted')
        except Exception as e:
            log(f'(no summary: {type(e).__name__}: {e})', echo = False)
        # an explicit --max-steps also keeps the extension of a converged run free of the stop (restore)
        given = self.max_steps_given or self.max_steps != self.cfg['sampler']['max_steps']
        more = f' --max-steps {self.max_steps}' if given else ''
        log(f'\ninterrupted at step {self.step} of {self.max_steps}; the chain is saved in '
            f'{os.path.join(self.run_dir, CHAIN)}. Resume with:\n'
            f'    python -m isofate.mcmc run --resume {self.run_dir}{more}')


def run_mcmc(cfg, workers = None, run_dir = None, resume = False, max_steps = None, make_plots = True):
    '''
    Runs the MCMC of cfg (config.load_config) and returns the run directory (see the module docstring).
    workers: default parallel.workers, reduced to fit the memory. max_steps: total steps, default
    sampler.max_steps. resume = True continues run_dir from its chain.npz (cfg None: its config.json) and gives
    the same chain as an uninterrupted run. Ctrl-C checkpoints and prints the resume command.
    make_plots = False skips plots.plot_run at the end.
    '''
    run = _Run(cfg, workers, run_dir, resume, max_steps)
    try:
        run.execute(make_plots)
    finally:
        release_lock(run.lock)
    return run.run_dir


###_____Reading a run_____###

def load_run(run_dir):
    '''
    {run_dir, cfg, free, chain (nsteps, nwalkers, ndim), log_prob (nsteps, nwalkers), blobs (nsteps, nwalkers)
    structured, accepted (nwalkers,), iteration, extra (the bookkeeping of chain.npz), summary (or None)}.
    '''
    run_dir = os.path.abspath(run_dir)
    path = os.path.join(run_dir, CHAIN)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: the run has no checkpoint yet')
    cfg = config.load_resolved(os.path.join(run_dir, 'config.json'))
    with np.load(path, allow_pickle = False) as d:
        data = {key: d[key] for key in d.files}
    summary_path = os.path.join(run_dir, SUMMARY)
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    return {'run_dir': run_dir, 'cfg': cfg, 'free': list(cfg['free']), 'chain': data['chain'],
            'log_prob': data['log_prob'], 'blobs': data.get('blobs'), 'accepted': data['accepted'],
            'iteration': int(data['iteration']), 'summary': summary,
            'extra': {key: value for key, value in data.items() if key not in BACKEND_KEYS}}


def discard_thin(run):
    '''
    (discard, thin) [steps] of a load_run dict: the summary's when it was written for this chain length, else
    the tau rule of write_summary (burn_and_thin).
    '''
    n = run['chain'].shape[0]
    summary = run.get('summary') or {}
    if summary.get('steps') == n and summary.get('discard') is not None:
        return int(summary['discard']), max(1, int(summary.get('thin') or 1))
    tau = autocorr_time(run['chain'], run['accepted']/max(n, 1))
    return burn_and_thin(tau, n, run['cfg']['sampler']['burn_factor'])


def posterior(run, discard = None, thin = None):
    '''
    (samples (N, ndim), blobs (N,)) of a load_run dict after burn-in and thinning; the defaults come from
    discard_thin (the summary, else the tau rule).
    '''
    chain = run['chain']
    if discard is None or thin is None:
        auto = discard_thin(run)
        discard = auto[0] if discard is None else discard
        thin = auto[1] if thin is None else thin
    start = discard + thin - 1 # emcee's get_chain(discard, thin) convention
    samples = chain[start::thin].reshape(-1, chain.shape[2])
    blobs = run['blobs'][start::thin].reshape(-1) if run['blobs'] is not None else None
    return samples, blobs


def _stats(x, with_nan = False):
    '''
    Percentiles (2.5, 16, 50, 84, 97.5), mean and std of x ignoring NaN; with inf values the percentiles are
    sample values (no interpolation) and mean/std cover the finite values.
    '''
    x = np.asarray(x, dtype = float).ravel()
    defined = x[~np.isnan(x)]
    out = {}
    if defined.size:
        method = 'nearest' if np.isinf(defined).any() else 'linear'
        p = np.percentile(defined, PERCENTILES, method = method)
        out.update({f'p{q:g}': float(v) for q, v in zip(PERCENTILES, p)})
        finite = defined[np.isfinite(defined)]
        out['mean'] = float(finite.mean()) if finite.size else nan
        out['std'] = float(finite.std()) if finite.size else nan
    else:
        out.update({f'p{q:g}': nan for q in PERCENTILES}, mean = nan, std = nan)
    if with_nan:
        out['frac_nan'] = (x.size - defined.size)/x.size if x.size else nan
    return out


def _check_failure(checks, j, nwalkers):
    '''
    Why check j of a tau_bookkeeping record did not count as stable ('' if it did).
    '''
    if checks['stable'][j]:
        return ''
    change, tol, walkers = checks['change'][j], checks['tol'][j], checks['walkers'][j]
    if not math.isfinite(change): # walkers -1: padded (tau_bookkeeping)
        return ('the check predates the noise-aware rule of 2026-10-29' if walkers < 0 else
                'there was no earlier estimate of tau to compare with')
    if not change < tol:
        return f'tau changed by {100*change:.3g}% (the rule needs < {100*tol:.3g}%, the max of tau_rtol and its noise)'
    if 0 <= walkers < nwalkers:
        return (f'{nwalkers - walkers} of {nwalkers} walkers were left out of tau (stuck or frozen in its window; a '
                'check counts only with every walker, see sampler.reset_stuck_at)')
    return 'the check did not count'


def not_converged_text(checks, s, reset_step, n, nwalkers):
    '''
    Why a chain whose final N/tau is above n_tau has not converged: the condition of the stopping rule (tau_rule)
    that its last check failed, from that check's own record (checks: tau_bookkeeping), e.g. N/tau, which the
    final chain passes, may have been below n_tau at that check. n: steps of the current chain.
    '''
    steps = checks['steps']
    if not steps:
        return 'no convergence check has run yet'
    i = len(steps) - 1
    step, tau = steps[i], checks['tau'][i]
    N = step - reset_step # steps of the current chain at that check
    k = int(s.get('tau_checks', SAMPLER_TAU_CHECKS))
    at = f'at the last check (step {step})'
    if not np.isfinite(tau).all():
        return f'tau could not be estimated {at}'
    if N < s['min_steps']:
        return f'N = {N} < min_steps = {s["min_steps"]} {at}'
    if not N > s['n_tau']*tau.max():
        later = f'; the next check comes at step {step + s["check_every"]}' if n > N else ''
        return (f'N/tau = {N/tau.max():.3g} {at}, not above n_tau = {s["n_tau"]:g}: N/tau passed it after that '
                f'check{later}')
    if checks['walkers'][i] < 0 and not math.isfinite(checks['change'][i]): # padded: a check of the rule before
        # 2026-10-29, a change below tau_rtol at one check (tau_bookkeeping)
        previous = checks['tau'][i - 1] if i else None
        if previous is not None and np.isfinite(previous).all():
            with np.errstate(all = 'ignore'):
                change = float(np.max(np.abs(tau - previous)/tau))
            return (f'tau changed by {100*change:.3g}% {at}, a check of the rule before 2026-10-29 (a change below '
                    f'tau_rtol = {100*s["tau_rtol"]:g}% at one check); a resume checks with the current rule')
        return f'the last check (step {step}) predates the noise-aware rule of 2026-10-29'
    failure = _check_failure(checks, i, nwalkers)
    if failure:
        return f'{failure} {at}'
    earlier = [j for j in range(max(0, i - (k - 1)), i) if not checks['stable'][j]]
    if earlier:
        j = earlier[-1]
        return (f'tau was stable at the last check (step {step}) but not at step {steps[j]} '
                f'({_check_failure(checks, j, nwalkers)}), and the rule needs {k} stable checks in a row')
    if i < k - 1:
        return f'tau was stable at the last check (step {step}), but the rule needs {k} stable checks in a row'
    return f'no check confirmed it (the last, step {step}, met every condition but did not count)'


def write_summary(run_dir, state = 'finished'):
    '''
    Writes <run_dir>/summary.json from chain.npz, config.json and provenance.json, and returns it: provenance,
    settings, timing, evaluations per status, acceptance, tau, convergence, discard and thin, percentiles of
    every physical parameter and blob field over the posterior, the MAP sample with its log-likelihood terms,
    warnings; in solve mode also the sampled coordinates ('sampled') and the model runs per evaluation ('solve').
    '''
    run = load_run(run_dir)
    cfg, chain, lp, blobs, extra = run['cfg'], run['chain'], run['log_prob'], run['blobs'], run['extra']
    n, nwalkers, ndim = chain.shape
    s = cfg['sampler']
    acceptance = run['accepted']/n
    tau = autocorr_time(chain, acceptance)
    discard, thin = burn_and_thin(tau, n, s['burn_factor'])
    finite_tau = bool(np.isfinite(tau).all())
    samples, post_blobs = posterior(run, discard, thin)

    prov = _read_json(os.path.join(run['run_dir'], 'provenance.json'))
    counts = {phase: np.array(_get(extra, f'counts_{phase}', np.zeros(N_STATUS)), dtype = np.int64)
              for phase in ('init', 'sampling')}
    n_sampling = int(counts['sampling'].sum())
    sampling_s, steps_timed = float(_get(extra, 'sampling_s', nan)), int(_get(extra, 'steps_timed', 0))
    reset_step = int(_get(extra, 'reset_step', 0))
    converged = bool(_get(extra, 'converged', False))

    warnings = list(cfg['warnings']) + [str(w) for w in _get(extra, 'run_warnings', [])]
    n_over = n/tau.max() if finite_tau else nan
    checks = tau_bookkeeping(extra, ndim)
    if not converged and finite_tau and n_over > s['n_tau']: # long enough now, but the stopping rule did not pass
        why = not_converged_text(checks, s, reset_step, n, nwalkers)
        warnings.append(f'not converged by the stopping rule: N/tau = {n_over:.3g} > {s["n_tau"]:g} on the final '
                        f'chain, but {why}; resume with a larger max_steps')
    elif not converged:
        warnings.append(f'not converged: N/tau = {n_over:.3g} (needs > {s["n_tau"]:g}); resume with a larger max_steps')
    n_stuck = int(np.sum(stuck_mask(acceptance)))
    has_reset = s['reset_stuck_at'] > 0
    if n < STUCK_AFTER:
        warnings.append(f'only {n} steps: the acceptance and tau are not meaningful yet')
    elif acceptance.mean() < STUCK_ACCEPTANCE and not (finite_tau and n_over > s['n_tau']):
        warnings.append(f'mean acceptance {acceptance.mean():.3g} < {STUCK_ACCEPTANCE:g} ({n_stuck} of {nwalkers} '
                        'walkers stuck) and no converged tau: the moves barely mix, so the samples may still be close '
                        'to the starting points, not a posterior. A posterior much thinner than the spread of the '
                        'walkers does this, e.g. a thin curved ridge such as the LHS 1140 b outflow island; sample it '
                        'in solve mode ([solve] in the config; see "Solve mode" in the README), or run far longer')
    elif n_stuck:
        warnings.append(f'{n_stuck} walkers with acceptance < min({STUCK_ACCEPTANCE:g}, {STUCK_RELATIVE:g} x the '
                        'median) (stuck)' + ('' if has_reset else '; consider sampler.reset_stuck_at'))
    behind = np.flatnonzero(behind_walkers(lp[-1])).tolist()
    if behind:
        one = len(behind) == 1
        warnings.append(f'{len(behind)} walker{"" if one else "s"} {behind} end{"s" if one else ""} with lp more '
                        f'than {RESET_DLP:g} below the median: lost in another region (e.g. a stripped-core '
                        f'plateau), {"it is" if one else "they are"} part of the posterior samples; run longer'
                        + ('' if has_reset else ' or set sampler.reset_stuck_at'))

    i_step, i_walker = np.unravel_index(np.argmax(lp), lp.shape)
    best = blobs[i_step, i_walker]
    stored = blobs.dtype.names # a run of an older version may lack newer fields: NaN
    obs = {f: float(best[f]) if f in stored else nan for f in model.OBS_FIELDS}
    log_like, terms = likelihood.log_like(obs, cfg)
    names = config.sampled_names(cfg)
    physical = physical_samples(samples, post_blobs, cfg)
    map_theta = physical_samples(chain[i_step, i_walker][None], blobs[i_step, i_walker:i_walker + 1], cfg)[0]
    evals = {phase: np.asarray(_get(extra, f'evals_{phase}', np.zeros(MAX_EVALS_HIST)), dtype = np.int64)
             for phase in ('init', 'sampling')}
    runs = int(np.sum(np.arange(MAX_EVALS_HIST)*evals['sampling'])) # model runs in sampling
    solving = None
    if config.solve_on(cfg):
        sv = cfg['solve']
        solving = {'param': sv['param'], 'observable': sv['observable'], 'name': sv['name'],
                   'bounds': list(sv['bounds']), 'tol': sv['tol'], 'range_sigma': sv['range_sigma'],
                   'warm_start': sv['warm_start'], 'derivative': run_derivative(cfg, extra),
                   'refit_steps': solve.refit_steps(cfg), 'evals_init': solve.evals_stats(evals['init']),
                   'evals_sampling': solve.evals_stats(evals['sampling']),
                   'evals_posterior': _stats(post_blobs['solve_evals']) if 'solve_evals' in stored else None}
    summary = {
        'state': state, 'name': cfg['name'], 'description': cfg['description'], 'run_dir': run['run_dir'],
        'provenance': prov,
        'settings': {'nwalkers': nwalkers, 'ndim': ndim, 'free': cfg['free'],
                     'n_steps': cfg['model']['n_steps'], 'n_atmodeller': cfg['model']['n_atmodeller'],
                     'backend': cfg['model']['backend'], 'workers': int(_get(extra, 'workers', 0)),
                     'moves': s['moves'], 'init': s['init'], 'seed': s['seed'],
                     'max_steps': int(_get(extra, 'max_steps', s['max_steps'])),
                     'tau_checks': int(s.get('tau_checks', SAMPLER_TAU_CHECKS)),
                     'observables': list(cfg['observables'])},
        'steps': n, 'total_steps': reset_step + n, 'reset_step': reset_step or None,
        'replaced_walkers': [int(i) for i in _get(extra, 'replaced', [])],
        'timing': {'wall_s': float(_get(extra, 'wall_s', nan)), 'warmup_s': float(_get(extra, 'warmup_s', nan)),
                   'init_s': float(_get(extra, 'init_s', nan)), 'sampling_s': sampling_s,
                   's_per_step': sampling_s/steps_timed if steps_timed else nan,
                   'isocalc_s_per_eval': (float(_get(extra, 'runtime_sampling', nan))/n_sampling
                                          if n_sampling else nan),
                   'isocalc_s_per_run': float(_get(extra, 'runtime_sampling', nan))/runs if runs else nan},
        'evaluations': {'total': int(counts['init'].sum() + n_sampling),
                        'init': {model.STATUS_NAMES[k]: int(c) for k, c in enumerate(counts['init'])},
                        'sampling': {model.STATUS_NAMES[k]: int(c) for k, c in enumerate(counts['sampling'])}},
        'acceptance': {'mean': float(acceptance.mean()), 'min': float(acceptance.min()), 'n_stuck': n_stuck,
                       'behind': behind, 'per_walker': acceptance.tolist()},
        'tau': dict(zip(names, tau.tolist())), 'tau_max': float(tau.max()) if finite_tau else nan,
        'n_over_tau': n/tau.max() if finite_tau else nan,
        'tau_history': {'steps': checks['steps'], 'tau': [t.tolist() for t in checks['tau']],
                        'stable': checks['stable'], 'change': checks['change'], 'tol': checks['tol'],
                        'walkers': checks['walkers']}, # aligned by check (tau_bookkeeping)
        'converged': converged, # at the last check; converged_at: the step of the first converged check
        'converged_at': int(_get(extra, 'converged_at', checks['steps'][-1] if converged and checks['steps'] else 0))
                        or None,
        'discard': int(discard), 'thin': int(thin), 'n_samples': int(samples.shape[0]),
        'params': {name: _stats(physical[:, i]) for i, name in enumerate(cfg['free'])},
        'sampled': {name: _stats(samples[:, i]) for i, name in enumerate(names)} if solving else None,
        'solve': solving,
        'blobs': {f: _stats(post_blobs[f], with_nan = True) for f in model.BLOB_FIELDS if f in stored},
        'map': {'step': int(i_step) + 1 + reset_step, 'walker': int(i_walker), 'lp': float(lp[i_step, i_walker]),
                'theta': dict(zip(cfg['free'], map_theta.tolist())),
                'sampled': dict(zip(names, chain[i_step, i_walker].tolist())) if solving else None,
                'log_prior': float(best['log_prior']), 'log_like': float(log_like),
                'log_jac': float(best['log_jac']) if 'log_jac' in stored else 0.0, 'terms': terms, 'obs': obs},
        'warnings': warnings,
    }
    _dump_json(summary, os.path.join(run['run_dir'], SUMMARY))
    return summary
