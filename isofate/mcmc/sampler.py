'''
Runs the MCMC: emcee's EnsembleSampler (Foreman-Mackey et al 2013) over a WorkerPool of model.log_prob.
Sets up or resumes the run directory, starts the walkers (prescan, bisect, solve, prior or ball), samples with
atomic checkpoints, stops on an autocorrelation-time rule, can reset stuck walkers once, and writes the summary.

emcee moves an ensemble of walkers through the parameter space. At each step every walker proposes a new position
built from the positions of the others (the moves of make_moves) and accepts it with the Metropolis-Hastings
probability, so that after a burn-in the walkers' positions sample the posterior. The proposals of half the
ensemble (a quarter with DESnooker) are evaluated at once, through the pool's map. The workers are spawned
processes (pool.py), since fork deadlocks once JAX is imported: the parent process never imports jax, atmodeller
or the coupled model, only the pool's workers do, and the parent stays at about 0.1 GB.

A run directory, <output_root>/<name>/mcmc_<YYYYmmdd-HHMMSS>/, holds (see "Outputs" in the README)
    config.json      the resolved cfg (config.load_resolved reads it back)
    config.toml      a copy of the TOML, with the --set overrides listed at the top
    provenance.json  git commit and dirty flag, package versions, host, start time of each session
    run.log          the progress lines (also printed), time-stamped, and the tracebacks of failures
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
    .lock            pid and host of the session running in the directory (removed at its end)

A run may take several sessions: the first `run`, then each `run --resume`. A session holds <run_dir>/.lock while
it runs, so a second `run --resume` of the same directory is refused instead of interleaving its checkpoints with
the first. Each checkpoint carries the run's bookkeeping (the extras of chain.npz) over to the next session, and a
resume gives the same chain, bit for bit, as an uninterrupted run.

Steps are counted over the whole run: after a reset at step k, chain.npz holds steps k+1, k+2, ... and
max_steps still counts from the start.

In solve mode (solve.py) the chain's coordinates are the sampler's (u_<observable> in place of the solved
parameter; config.sampled_names); the physical value of the solved parameter is the blob field solve_value, and the
summary, the figures and the tracks report the physical parameters (physical_samples).
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
# file names in the run directory
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
STUCK_AFTER = 50 # [steps] chain length from which progress lines count stuck walkers and the summary judges them
# the reset also replaces walkers with lp < median(lp) - RESET_DLP, and walkers that far down count as "behind":
# a factor e^-20 (about 2e-9) in probability, so such a walker sits in another, much worse region
RESET_DLP = 20.0
# solve mode with the plateau branch: walkers that cross between stripped planets and those that keep an atmosphere
# fewer times than this after the burn-in (median) leave the split between the two uncertain (a summary warning)
PLATEAU_CROSSINGS = 20
PERCENTILES = (2.5, 16, 50, 84, 97.5) # the median and the 68% and 95% intervals
TAU_WINDOW_C = 5.0 # emcee's integrated_time window factor c (M = c tau), used by tau_noise
SAMPLER_TAU_CHECKS = config.SAMPLER_DEFAULTS['tau_checks'] # for configs saved before sampler.tau_checks existed
CHECKPOINT_MIN_S = 60.0 # [s] periodic checkpoints at most this often ...
CHECKPOINT_COST = 20.0 # ... and at most every this many times the duration of the last save (<= 5% of the time)
# positions in a log_prob tuple, which is (lp,) + the blob values in BLOB_FIELDS order
STATUS = 1 + model.BLOB_FIELDS.index('status')
RUNTIME = 1 + model.BLOB_FIELDS.index('runtime')
SOLVE_EVALS = 1 + model.BLOB_FIELDS.index('solve_evals')
MAX_EVALS_HIST = 256 # histogram bins of solve_evals (model runs per evaluation); the last one holds 255 and more
N_STATUS = len(model.STATUS_NAMES) # the status codes 0-5 (model.py)
# the arrays of NpzBackend.save; every other key of a checkpoint is the run's bookkeeping (the extras)
BACKEND_KEYS = ('chain', 'log_prob', 'accepted', 'iteration', 'nwalkers', 'ndim', 'blobs', 'rs_name', 'rs_key',
                'rs_pos', 'rs_has_gauss', 'rs_cached')


###_____Checkpointing_____###

class NpzBackend(emcee.backends.Backend):
    '''
    emcee's in-memory Backend with an atomic .npz checkpoint, so no h5py is needed
    It holds what emcee's Backend holds: the chain (nsteps, nwalkers, ndim), log_prob (nsteps, nwalkers), the
    blobs, the accepted proposals of each walker, the step count (iteration) and the sampler's RNG state; after
    load() also the checkpoint's other arrays (.extra). save() writes the filled steps and the sampler's RNG
    state, so a sampler built on load(path) continues bit-identically.
    '''

    def __init__(self, dtype = None):
        '''
        Starts an empty backend; emcee's sampler sizes it (reset) when it is built on it.
        '''
        super().__init__(dtype = dtype)
        self.iteration = 0 # emcee sets it in reset(); 0 before the sampler exists

    def save(self, path, **extra):
        '''
        Writes the backend to an .npz file, atomically

        Inputs:
            - path: the file to write, e.g. <run_dir>/chain.npz [str]
            - extra: more arrays and scalars to store with it, e.g. the run's bookkeeping (_Run.extras) [keywords]

        Output: none; path holds chain, log_prob, accepted, iteration, nwalkers, ndim, blobs (when the run has
            them), the RNG state (rs_name, rs_key, rs_pos, rs_has_gauss, rs_cached) and the extras
        '''
        it = self.iteration
        rs = self.random_state # ('MT19937', uint32[624], pos, has_gauss, cached_gaussian)
        # emcee allocates the arrays of a whole sample() call at its start: only the first it steps are filled
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
        Reads a backend written by save()

        Inputs:
            - path: the checkpoint, e.g. <run_dir>/chain.npz [str]

        Output: the backend, ready for a sampler to continue from [NpzBackend]; the extra arrays of save() are
            in its .extra [dict]
        '''
        with np.load(path, allow_pickle = False) as d:
            self = cls(dtype = d['chain'].dtype)
            # reset marks the backend initialized, so that emcee's sampler continues this chain (and its RNG
            # state) instead of starting a new one
            self.reset(int(d['nwalkers']), int(d['ndim']))
            self.chain, self.log_prob = d['chain'].copy(), d['log_prob'].copy()
            self.accepted, self.iteration = d['accepted'].copy(), int(d['iteration'])
            self.blobs = d['blobs'].copy() if 'blobs' in d.files else None
            # back to the tuple of numpy's RandomState.get_state
            self.random_state = (str(d['rs_name']), d['rs_key'].copy(), int(d['rs_pos']),
                                 int(d['rs_has_gauss']), float(d['rs_cached']))
            self.extra = {key: d[key] for key in d.files if key not in BACKEND_KEYS}
        return self


def _get(extra, key, default):
    '''
    Returns a checkpoint extra as a Python scalar (np.savez stores scalars as 0-d arrays) or array, or default
    when absent.
    '''
    if key not in extra:
        return default
    value = extra[key]
    return value.item() if value.ndim == 0 else value


###_____Small helpers_____###

class RunLog:
    '''
    The run's log: called with a text, it prints it and appends it, time-stamped, to <run_dir>/run.log.
    Holds only the path of run.log.
    '''

    def __init__(self, run_dir):
        '''
        Logs to <run_dir>/run.log (created at the first line, appended to by every later session).
        '''
        self.path = os.path.join(run_dir, 'run.log')

    def __call__(self, text = '', echo = True):
        '''
        Logs a text

        Inputs:
            - text: the message; each of its lines gets its own time stamp in run.log [str]
            - echo: False writes to run.log only, e.g. a traceback [Bool]
        '''
        if echo:
            print(text, flush = True) # flushed, so that a redirected output (nohup) is up to date
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        # opened for each text, so that every line is on disk before the run goes on
        with open(self.path, 'a') as f:
            for line in str(text).splitlines() or ['']: # an empty text still writes one stamped line
                f.write(f'{stamp}  {line}\n')


def duration_text(seconds):
    '''
    Formats a duration for the log, e.g. '45 s', '12m05s', '3h20m'

    Inputs:
        - seconds: the duration [s]; a negative one counts as 0

    Output: the text, or '?' when the duration is unknown (not finite) [str]
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
    '''
    Returns a status code as an index 0-5; a NaN or unknown code counts as 5 (unexpected error).
    '''
    return int(status) if math.isfinite(status) and 0 <= status < N_STATUS else int(model.UNEXPECTED)


def status_counts(statuses):
    '''
    Counts the evaluations per status code 0-5 (model.STATUS_NAMES)

    Inputs:
        - statuses: status codes of the evaluations, any shape [array]; a NaN or unknown code counts as 5

    Output: number of evaluations with each code (6,) [array]
    '''
    return np.bincount([_status_index(float(s)) for s in np.ravel(statuses)], minlength = N_STATUS)


def status_text(counts):
    '''
    Formats counts per status code as e.g. 'ok 120, outside prior 3' (non-zero ones only; 'none' if every count
    is 0).
    '''
    parts = [f'{model.STATUS_NAMES[k]} {int(n)}' for k, n in enumerate(counts) if n]
    return ', '.join(parts) if parts else 'none'


def _read_json(path, default = None):
    '''
    Reads a JSON file, or returns default when it does not exist.
    '''
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _dump_json(obj, path):
    '''
    Writes obj to path as JSON, atomically (through a .tmp file), with numpy scalars and arrays as Python numbers
    and lists.
    '''
    def default(x):
        '''
        Converts what json cannot write itself: numpy scalars and arrays.
        '''
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(f'{type(x).__name__} is not JSON serializable')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent = 1, default = default) # NaN and Infinity as Python's json writes them
        f.write('\n')
    os.replace(tmp, path) # atomic: a kill mid-write leaves the old file whole


def check_disk(path):
    '''
    Checks the free space on the disk that will hold a run, before anything is written

    Inputs:
        - path: the run directory or the output root, which need not exist yet [str]; the nearest existing parent
          is probed

    Output: free space [GB]; raises RuntimeError below MIN_FREE_GB = 1 GB
    '''
    probe = os.path.abspath(path)
    while not os.path.exists(probe): # climb to the nearest existing parent
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free/1e9 # [GB]
    if free < MIN_FREE_GB:
        raise RuntimeError(f'only {free:.2f} GB free on the disk of {path}; a run needs at least {MIN_FREE_GB:g} GB. '
                           'Free some space first.')
    return free


def provenance():
    '''
    Records what a session runs with, for provenance.json
    Package versions are read from the installed metadata, so the parent never imports jax or atmodeller.

    Output: dict
        - time: the start time, local, with the UTC offset [str]
        - git: repo (the IsoFATE repo), commit, dirty flag and status (the first 50 lines of git status
          --porcelain); commit, dirty and status stay None without git [dict]
        - versions: isofate.mcmc, isofate, emcee, numpy, scipy, jax, atmodeller and numba (None when not
          installed) [dict]
        - python: the Python version [str]
        - platform, host: platform.platform() and the host name [str]
    '''
    import isofate
    repo = os.path.dirname(os.path.dirname(os.path.abspath(isofate.__file__))) # the folder above the package
    git = {'repo': repo, 'commit': None, 'dirty': None, 'status': None}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS = '0') # git status must not write the index
    try:
        head = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output = True, text = True,
                              timeout = 20, env = env)
        if head.returncode == 0: # a git checkout
            status = subprocess.run(['git', '-C', repo, 'status', '--porcelain'], capture_output = True,
                                    text = True, timeout = 20, env = env).stdout.splitlines()
            git.update(commit = head.stdout.strip(), dirty = bool(status), status = status[:50])
    except (OSError, subprocess.SubprocessError): # no git, or it hung: a run must not fail on its provenance
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
    Returns the emcee moves of sampler.moves, as (move, weight) pairs
    DE: differential evolution, ter Braak 2006; DESnooker: ter Braak & Vrugt 2008; stretch: Goodman & Weare 2010

    A DE move proposes x + gamma0 (x_j - x_k) for a walker at x, from two walkers j and k of the rest of the
    ensemble; emcee's default gamma0 = 2.38/sqrt(2 ndim) suits moves within one mode, while gamma0 = 1 takes the
    whole difference and so can carry a walker from one mode to another (mode jumps). In tests on bimodal and
    strongly correlated targets the "de" mix beat the stretch move: tau 19 against 53, and it kept 50/50 mode
    occupancy where stretch never moved walkers between modes.

    Inputs:
        - name: sampler.moves [str]

    Output: list of (emcee move, weight)
        - "de" (default): DE 0.7 + DE(gamma0 = 1, mode jumps) 0.1 + DESnooker 0.2
        - "de_only": DE 0.9 + DE(gamma0 = 1) 0.1
        - "stretch": the affine-invariant stretch move alone
    '''
    if name == 'stretch':
        return [(emcee.moves.StretchMove(), 1.0)]
    if name == 'de_only':
        return [(emcee.moves.DEMove(), 0.9), (emcee.moves.DEMove(gamma0 = 1.0), 0.1)]
    return [(emcee.moves.DEMove(), 0.7), (emcee.moves.DEMove(gamma0 = 1.0), 0.1),
            (emcee.moves.DESnookerMove(), 0.2)]


###_____Stuck walkers_____###

def stuck_mask(acceptance):
    '''
    Finds the stuck walkers: acceptance < min(0.05, 0.2 x the median walker's), or no accepted move at all
    The threshold is relative because on some posteriors every walker accepts only 1-3% of its proposals and still
    mixes (e.g. emcee's moves on the LHS 1140 b ridge without solve mode, with tau of 1e3-2e3 steps): an absolute
    0.05 would call all of them stuck, and tau and the convergence test would never use any.

    Inputs:
        - acceptance: fraction of accepted proposals of each walker (nwalkers,) [array]

    Output: True for the stuck walkers (nwalkers,) [Bool array]
    '''
    acceptance = np.asarray(acceptance, dtype = float)
    if not acceptance.size:
        return np.zeros(0, dtype = bool)
    threshold = min(STUCK_ACCEPTANCE, STUCK_RELATIVE*float(np.median(acceptance)))
    return (acceptance < threshold) | (acceptance <= 0) # <= 0 also when the median itself is 0


def stuck_walkers(acceptance, lp):
    '''
    Finds the walkers a stuck-walker reset replaces: stuck (stuck_mask), lp below median(lp) - 20, or lp not finite

    Inputs:
        - acceptance: fraction of accepted proposals of each walker (nwalkers,) [array]
        - lp: log-probability of each walker at the reset (nwalkers,) [array]

    Output: True for the walkers to replace (nwalkers,) [Bool array]
    '''
    acceptance, lp = np.asarray(acceptance, dtype = float), np.asarray(lp, dtype = float)
    finite = np.isfinite(lp)
    median = np.median(lp[finite]) if finite.any() else -inf # of the finite lp only
    bad = stuck_mask(acceptance) | ~finite
    bad[finite] |= lp[finite] < median - RESET_DLP # far behind: lost in a much worse region
    return bad


def behind_walkers(lp):
    '''
    Finds the walkers far behind the others: lp more than 20 below the median (or not finite)
    Such a walker is lost in another region, e.g. the flat stripped-core plateau, where it still accepts moves and
    so is not "stuck" by acceptance.

    Inputs:
        - lp: log-probability of each walker (nwalkers,) [array]

    Output: True for the walkers behind (nwalkers,) [Bool array]; every walker when no lp is finite
    '''
    lp = np.asarray(lp, dtype = float)
    finite = np.isfinite(lp)
    if not finite.any():
        return ~finite
    # the non-finite ones are in ~finite already; the 0.0 only keeps them out of the comparison
    return ~finite | (np.where(finite, lp, 0.0) < np.median(lp[finite]) - RESET_DLP)


###_____Convergence: the autocorrelation time_____###

def autocorr_info(chain, acceptance):
    '''
    Estimates the integrated autocorrelation time tau of each coordinate, with emcee's estimator
    From Sokal 1997, as in emcee (Foreman-Mackey et al 2013)

    tau is about the number of steps a walker needs to forget where it was: a chain of N steps per walker holds
    about nwalkers x N/tau independent samples. The estimate uses the last 2/3 of the chain (the first third, the
    nearest to the starting points, is left out as burn-in), and only the walkers that are not stuck (stuck_mask)
    and moved in that window: a frozen walker has zero variance, and its normalized autocorrelation function (0/0)
    would make the estimate NaN.

    Inputs:
        - chain: walker positions (nsteps, nwalkers, ndim) [array]
        - acceptance: fraction of accepted proposals of each walker (nwalkers,) [array]

    Output: tuple
        - tau: integrated autocorrelation time of each coordinate (ndim,) [steps; array]; NaN when impossible
          (fewer than 3 steps, no walker to use, an error of emcee's estimator, or an estimate <= 0)
        - steps: the steps of the window it used [steps]
        - walkers: the number of walkers it used [int]
    '''
    n, nwalkers, ndim = chain.shape
    if n < 3:
        return np.full(ndim, nan), 0, 0
    window = chain[n//3:]
    # not stuck, and moved in the window in every coordinate
    good = ~stuck_mask(acceptance) & np.all(np.ptp(window, axis = 0) > 0, axis = 1)
    if not good.any():
        return np.full(ndim, nan), len(window), 0
    # tol = 0: never raise on a short chain (emcee's own N > 50 tau test); the stopping rule judges N/tau itself
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
    Returns the tau of autocorr_info alone (ndim,) [steps; array].
    '''
    return autocorr_info(chain, acceptance)[0]


def tau_noise(tau, steps, walkers):
    '''
    Estimates the relative 1-sigma statistical error of each tau estimate
    From Sokal 1997: sqrt(2 (2M + 1)/L) for a window of M steps and L samples, here with emcee's window M = 5 tau
    and L = steps x walkers (the walkers' autocorrelation functions are averaged, counted here as independent)

    On independent AR(1) walkers the scatter of tau over replicate ensembles is about 0.75 times this (0.74-0.84
    in the tests); on an LHS 1140 b chain, whose walkers are not independent, the spread of tau between random
    halves of the walkers was 0.6-1 times it. On LHS 1140 b (24 walkers, tau of about 19 steps) it is about
    10-15% at N = 1000-1500.

    Inputs:
        - tau: the estimate of each coordinate [steps; array]
        - steps: the steps of the window the estimate used [steps]
        - walkers: the number of walkers it used [int]

    Output: relative error of each tau [ndim; array]; NaN without an estimate (no steps or walkers, or tau NaN)
    '''
    tau = np.asarray(tau, dtype = float)
    if steps <= 0 or walkers <= 0:
        return np.full(tau.shape, nan)
    with np.errstate(all = 'ignore'):
        return np.sqrt(2*(2*TAU_WINDOW_C*tau + 1)/(steps*walkers))


def tau_rule(tau, previous, noise, n, s, stable_before, complete = True):
    '''
    Decides one convergence check: is tau stable, and has the run converged?
    tau is stable when every walker is in the estimate (complete), and every coordinate's tau changed since the
    previous check by less than max(tau_rtol, its statistical noise). The run has converged when n >= min_steps,
    n > n_tau*max(tau) (the chain is long compared with its own memory), and tau is stable at this check and the
    tau_checks - 1 before it.

    Why complete: autocorr_info leaves out stuck walkers and walkers that did not move in its window, and the
    noise grows as 1/sqrt(walkers used), so 4 moving walkers of 24 get a tolerance of about 40% and pass. The frozen
    ones are still in the posterior samples (posterior() uses every walker), so such a check must not count, however
    long the run; sampler.reset_stuck_at is the remedy.

    Limits, from simulations: the estimate is biased low, by about 15-30% at N = 50 tau, so at the stop N/tau_true
    is about 40-45 rather than 50; from N of about 1000 on, successive estimates share most of their window and the
    stability test almost always passes on a stationary chain, so the stop is then set mainly by
    n > n_tau*max(tau); and a slow component that the 5 tau window cannot see (rare switches between regions, a
    plateau) is not detected at all. A "converged" flag therefore deserves a look at traces.png.

    Inputs:
        - tau: this check's estimate of each coordinate [steps; array]
        - previous: the tau of the previous check [steps; array], or None at the first check
        - noise: the relative error of each tau (tau_noise) [ndim; array]; NaN counts as 0 (tau_rtol alone)
        - n: steps in the current chain (since the reset, if there was one) [steps]
        - s: the sampler config: tau_rtol, n_tau, min_steps and tau_checks (configs saved before tau_checks
          existed get SAMPLER_TAU_CHECKS) [dict]
        - stable_before: the stable flags of the earlier checks, oldest first [list of Bool]
        - complete: True when every walker is in the estimate [Bool]

    Output: tuple
        - stable: tau stable at this check [Bool]
        - converged: the stopping rule passes [Bool]
        - change: relative change of tau since the previous check, of the coordinate closest to failing [ndim]
        - tol: the tolerance of that coordinate [ndim]
        change and tol are NaN without a previous check, or when this or the previous tau is not finite.
    '''
    tau, noise = np.asarray(tau, dtype = float), np.asarray(noise, dtype = float)
    finite = bool(np.isfinite(tau).all())
    if previous is None or not finite or not np.isfinite(previous).all():
        return False, False, nan, nan
    with np.errstate(all = 'ignore'):
        change = np.abs(tau - previous)/tau # relative to the newer estimate
        tol = np.maximum(s['tau_rtol'], np.nan_to_num(noise, nan = 0.0))
        worst = int(np.argmax(change/tol)) # the coordinate closest to failing
    stable = bool(complete and np.all(change < tol))
    k = int(s.get('tau_checks', SAMPLER_TAU_CHECKS))
    recent = list(stable_before)[-(k - 1):] if k > 1 else [] # the k - 1 checks before this one
    steady = stable and len(recent) == k - 1 and all(recent) # k stable checks in a row (and that many exist)
    converged = bool(steady and n >= s['min_steps'] and n > s['n_tau']*tau.max())
    return stable, converged, float(change[worst]), float(tol[worst])


def check_text(step, tau, noise, walkers, nwalkers, n, s, previous, stable, converged, change, tol):
    '''
    Formats the run.log line of one convergence check (_Run.check_convergence)
    The line gives the tau of each coordinate (with the largest of their noises) and the walkers used, N/tau
    against n_tau, the change of tau against its tolerance with the verdict, and N against min_steps;
    '  -> converged' ends it when the rule passed.

    Inputs:
        - step: the step of the check, counted over the whole run [steps]
        - tau: the estimate of each coordinate [steps; array]
        - noise: its relative error (tau_noise) [ndim; array]
        - walkers: the walkers the estimate used [int]
        - nwalkers: the walkers of the ensemble [int]
        - n: steps in the current chain [steps]
        - s: the sampler config [dict]
        - previous: the tau of the check before [steps; array], None at the first
        - stable, converged, change, tol: the result of tau_rule

    Output: the line [str]
    '''
    tau, noise = np.asarray(tau, dtype = float), np.asarray(noise, dtype = float)
    known = noise[np.isfinite(noise)]
    # the largest noise of the coordinates
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


###_____Bookkeeping of a checkpoint_____###

def tau_bookkeeping(extra, ndim):
    '''
    Reads the record of the convergence checks from a checkpoint's extras, every list as long as steps
    Checkpoints written before the noise-aware rule existed have steps and tau only: their checks get stable
    False, change and tol NaN and walkers -1 (unknown), padded at the front so that the lists stay aligned by
    index (tau_rule then sees no stable check before the first new one, as it saw none before).

    Inputs:
        - extra: the extras of chain.npz (NpzBackend.load, load_run) [dict]
        - ndim: the number of sampled coordinates [int]

    Output: dict of lists, one entry per check
        - steps: the step of each check [steps]
        - tau: tau of each coordinate, one array (ndim,) per check [steps]
        - stable: tau stable at that check [Bool]
        - change, tol: the change of tau and its tolerance, of the coordinate closest to failing [ndim]
        - walkers: the walkers the estimate used [int]
    '''
    steps = [int(i) for i in _get(extra, 'tau_steps', [])]
    tau = [np.array(t, dtype = float) for t in _get(extra, 'tau', np.empty((0, ndim)))]
    k = len(steps)

    def aligned(key, fill, kind):
        '''
        The last k values of the extra key, padded at the front with fill to k.
        '''
        values = [kind(x) for x in _get(extra, key, [])][-k:] if k else []
        return [fill]*(k - len(values)) + values
    return {'steps': steps, 'tau': tau, 'stable': aligned('tau_stable', False, bool),
            'change': aligned('tau_change', nan, float), 'tol': aligned('tau_tol', nan, float),
            'walkers': aligned('tau_walkers', -1, int)}


def run_derivative(cfg, extra):
    '''
    Returns the solve.derivative scheme of a solve-mode run
    Solve mode samples u in place of the solved parameter P, so its density carries the Jacobian |dP*/du| of that
    change of variables, which keeps the posterior over P unchanged. The Jacobian comes from a finite difference,
    "forward" or "central", and the two give slightly different densities (on LHS 1140 b their ln|J| differ by
    0.009, median), so a chain must keep one scheme.

    The scheme is the config's. A config saved before the option existed gets the scheme its last session used,
    so that a resume continues the chain's own target density: the checkpoint's 'derivative', else "forward" when
    the checkpoint holds a warm-start table (only code with the forward default wrote one; e.g. the final
    LHS 1140 b chain, forward from its step 251 on), else "central" (solve.OLD_DERIVATIVE).

    Inputs:
        - cfg: the run's config [dict]
        - extra: the extras of its checkpoint [dict]

    Output: "forward" or "central" [str]
    '''
    if 'derivative' in cfg['solve']:
        return cfg['solve']['derivative']
    if 'derivative' in extra:
        return str(_get(extra, 'derivative', solve.OLD_DERIVATIVE))
    return 'forward' if 'table_version' in extra else solve.OLD_DERIVATIVE


###_____Burn-in and thinning_____###

def burn_and_thin(tau, n, burn_factor):
    '''
    Chooses the burn-in and the thinning of a chain from its tau, as emcee's documentation advises: drop a few
    tau at the start, then keep one step in every half of the shortest tau
    discard = ceil(burn_factor*max tau), capped at n/2; thin = floor(0.5*min tau), at least 1 and leaving at
    least one step. Without a finite tau: discard half the chain, no thinning.

    Inputs:
        - tau: the autocorrelation time of each coordinate [steps; array]
        - n: the steps of the chain [steps]
        - burn_factor: sampler.burn_factor [ndim]

    Output: tuple
        - discard: steps dropped at the start [steps]
        - thin: every thin-th step is kept after that [steps]
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
    The pool emcee sees: WorkerPool.map, plus a tally of every evaluation, rejected proposals included (the
    chain's blobs only keep accepted ones)
    Holds, per phase ('init' and 'sampling'; phase selects the tally): the evaluations per status code (counts),
    the isocalc time [s] (runtime) and a histogram of the model runs per evaluation (solve_evals; evals).
    '''

    def __init__(self, pool):
        '''
        Wraps pool (a WorkerPool); the tallies start at 0, in the 'init' phase.
        '''
        self.pool, self.phase = pool, 'init'
        self.counts = {'init': np.zeros(N_STATUS, dtype = np.int64), 'sampling': np.zeros(N_STATUS, dtype = np.int64)}
        self.runtime = {'init': 0.0, 'sampling': 0.0} # [s]
        self.evals = {phase: np.zeros(MAX_EVALS_HIST, dtype = np.int64) for phase in ('init', 'sampling')}

    def map(self, f, iterable):
        '''
        Evaluates f over iterable through the WorkerPool and tallies the results in the current phase

        Inputs:
            - f: model.log_prob (or log_prob_physical), whose results are (lp,) + the blob values [function]
            - iterable: the thetas

        Output: the results, in the order of iterable [list of tuples]
        '''
        results = self.pool.map(f, iterable)
        counts, evals = self.counts[self.phase], self.evals[self.phase]
        for r in results:
            counts[_status_index(r[STATUS])] += 1
            self.runtime[self.phase] += r[RUNTIME]
            k = r[SOLVE_EVALS]
            # bin k: evaluations with k model runs (a NaN count goes to bin 0, no model run)
            evals[min(int(k), MAX_EVALS_HIST - 1) if math.isfinite(k) else 0] += 1
        return results

    def unexpected(self):
        '''
        Returns the number of unexpected errors (status 5) in this session, both phases [int].
        '''
        return int(self.counts['init'][int(model.UNEXPECTED)] + self.counts['sampling'][int(model.UNEXPECTED)])


evaluate_many = model.evaluate_many # (lp, blobs) of model.log_prob at thetas through a pool


###_____Initialization of the walkers_____###

def _no_start_advice(n_finite, n_needed, what):
    '''
    Returns the error text of an initialization with too few finite points, with what to try instead.
    '''
    return (f'{what}: only {n_finite} points with a finite posterior, {n_needed} walkers needed. Check the model '
            'against the data at a sensible point (python -m isofate.mcmc check CONFIG --theta ...), widen the priors, '
            'raise sampler.prescan_factor, or start from sampler.init = "ball" around a good params.<name>.value.')


def safe_jitter(cfg, pool, coords, lp, blobs, rng, others = None, scale = JITTER_SCALE):
    '''
    Makes jittered copies of points whose lp and blobs are known, without pushing them off a thin ridge
    Each point is moved by scale x the prior box (params.jitter) and re-evaluated. When its lp falls more than 5
    below the unjittered point's, the jitter is halved and drawn again, at most 8 times, so starts on a thin ridge
    stay on it (there a fixed jitter of 1e-3 of the box can cost tens in lp). A point that still loses more keeps
    its unjittered position if that is distinct (no other row of coords or of others, e.g. the walkers a reset
    keeps, sits there); a copy that is not distinct takes its best try with a finite lp, or as the last resort
    stays an exact copy.

    Inputs:
        - cfg: the config in the sampler's coordinates (prm.sampled_cfg) [dict]
        - pool: the pool that evaluates model.log_prob [_CountingPool]
        - coords: the points (n, ndim) [array]
        - lp: their log-probability (n,) [array]
        - blobs: their blobs (n,) [structured array]
        - rng: random generator [numpy Generator]
        - others: more occupied positions (m, ndim), e.g. the walkers a reset keeps, or None [array]
        - scale: the jitter, as a fraction of the prior box [ndim]

    Output: tuple
        - coords: the jittered points (n, ndim) [array]
        - lp: their log-probability (n,) [array]
        - blobs: their blobs (n,) [structured array]
        - info: halved (points whose jitter was halved), unjittered (points that kept their position) and
          copies (points that stayed exact copies) [dict]
    '''
    coords, lp, blobs = np.array(coords, dtype = float), np.asarray(lp, dtype = float), np.asarray(blobs)
    new = prm.jitter(coords, cfg, rng, scale = scale)
    lp_new, blobs_new = evaluate_many(pool, new)
    best = (new.copy(), lp_new.copy(), blobs_new.copy()) # the best try of each point
    bad = ~(lp_new >= lp - JITTER_DLP) # lost more than JITTER_DLP (a NaN lp counts as lost)
    halved = bad.copy()
    # halve the jitter of the points that lost too much and draw them again: scale/2, scale/4, ...
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
    # every occupied position: a point may stay unjittered only where nothing else sits
    everyone = coords if others is None else np.vstack([coords, np.asarray(others, dtype = float)])
    n_unjittered = n_copies = 0
    for i in np.flatnonzero(bad):
        if np.sum(np.all(everyone == coords[i], axis = 1)) <= 1: # only itself
            new[i], lp_new[i], blobs_new[i] = coords[i], lp[i], blobs[i]
            n_unjittered += 1
        elif np.isfinite(best[1][i]): # a copy: its best try
            new[i], lp_new[i], blobs_new[i] = best[0][i], best[1][i], best[2][i]
        else: # a copy without a finite try: an exact copy, the last resort
            new[i], lp_new[i], blobs_new[i] = coords[i], lp[i], blobs[i]
            n_copies += 1
    return new, lp_new, blobs_new, {'halved': int(halved.sum()), 'unjittered': n_unjittered, 'copies': n_copies}


def _jitter_text(info, n):
    '''
    Formats the jitter part of a log line from the info of safe_jitter, for n points.
    '''
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
    Starts the walkers from the best points of a prescan of the prior (init = "prescan")
    prescan_factor*nwalkers Latin-hypercube prior points are evaluated (saved to prescan.npz, with the chosen
    ones as start); nwalkers of the best 2*nwalkers finite ones are chosen at random, without replacement, then
    jittered by 1e-3 of the prior box (safe_jitter: halved while the lp drops by more than 5, else unjittered)
    and re-evaluated. A Latin hypercube puts one point in each of n equal-probability slices of every parameter,
    so the prescan covers each prior evenly; the random choice among the best 2*nwalkers keeps some spread instead
    of crowding every walker onto the very best points.

    Inputs:
        - cfg: the config in the sampler's coordinates (prm.sampled_cfg) [dict]
        - pool: the pool that evaluates model.log_prob [_CountingPool]
        - nwalkers: the number of walkers [int]
        - rng: random generator [numpy Generator]
        - run_dir: the run directory [str]
        - log: the run's log [RunLog]

    Output: tuple
        - coords: the starting positions (nwalkers, ndim) [array]
        - lp: their log-probability (nwalkers,) [array]
        - blobs: their blobs (nwalkers,) [structured array]
    Raises RuntimeError, with advice, when fewer than nwalkers prescan points are finite (prescan.npz is saved
    first, for a look at what went wrong).
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
    # the 2*nwalkers highest lp (a stable sort keeps ties in draw order), then nwalkers of them at random
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
    Starts the walkers on the surface where sampler.bisect_observable fits its data (init = "bisect"; search.py)
    Batches of Latin-hypercube prior draws are bisected along sampler.bisect_param (saved to bisect.npz), and
    nwalkers of the 2*nwalkers converged points with the highest lp are chosen at random. They are not jittered:
    they are distinct already, and the surface is thinner than any sensible jitter. This is the way onto a thin
    ridge that a prescan almost never hits (e.g. the LHS 1140 b outflow island); sampling it is another matter
    (see "Solve mode" in the README). With fewer than nwalkers converged after 3 batches,
    sampler.bisect_fallback = true falls back to init_prescan with a warning; false raises RuntimeError.

    Inputs:
        - cfg: the run's config; physical coordinates, since in solve mode init = "bisect" becomes "solve" [dict]
        - pool, nwalkers, rng, run_dir, log: as in init_prescan
        - warn: logs a warning and keeps it for the summary (_Run._warn); default log [function]

    Output: tuple (coords, lp, blobs), as in init_prescan
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
    Counts the outcomes of solve-mode evaluations

    Inputs:
        - lp: their log-probability [array]
        - blobs: their blobs [structured array]

    Output: dict
        - ok: finite lp
        - no root: status 1, no value of the solved parameter inside its prior gives the sampled u
        - lp -inf: solved, but ruled out by the likelihood
        - failed: everything else (model failures, solves that did not converge or have no usable derivative)
    '''
    status, solved = blobs['status'], np.isfinite(blobs['solve_value'])
    ok = np.isfinite(lp)
    no_root = status == model.OUTSIDE_PRIOR
    ruled_out = ~ok & ~no_root & solved
    return {'ok': int(ok.sum()), 'no root': int(no_root.sum()), 'lp -inf': int(ruled_out.sum()),
            'failed': int((~ok & ~no_root & ~solved).sum())}


def init_solve(cfg, pool, nwalkers, rng, run_dir, log):
    '''
    Starts the walkers of solve mode from solved draws (init = "solve")
    Batches of 3*nwalkers draws (params.sample_solve: a Latin hypercube with the other free parameters from their
    priors and u from the data distribution of the solve observable) are each solved from both prior bounds of
    the solved parameter (cold solves: no table exists yet), until 2*nwalkers have a finite lp (at most 5
    batches). nwalkers of the best 2*nwalkers are chosen at random, without replacement, and not jittered (the
    draws are distinct in every coordinate, u included). Saves solve_init.npz, then fits the warm-start table to
    every converged solve and writes solve_table.npz, which the workers pick up at their next evaluation
    (solve.worker_table; unless solve.warm_start = false): from then on each solve starts from the table's guess
    of the root instead of from both prior bounds.

    Inputs:
        - cfg: the run's config [dict]
        - pool, nwalkers, rng, run_dir, log: as in init_prescan

    Output: tuple (coords, lp, blobs), as in init_prescan; coords in the sampler's coordinates (u in place of the
        solved parameter)
    Raises RuntimeError, with what to check, when fewer than nwalkers draws have a finite lp.
    '''
    sv = cfg['solve']
    parts = []
    start = time.perf_counter()
    for batch in range(1, SOLVE_INIT_BATCHES + 1):
        thetas = prm.sample_solve(cfg, SOLVE_INIT_FACTOR*nwalkers, rng)
        lp, blobs = evaluate_many(pool, thetas)
        parts.append((thetas, lp, blobs))
        thetas_all, lp_all, blobs_all = (np.concatenate([p[k] for p in parts]) for k in range(3)) # all batches
        c = solve_counts(lp_all, blobs_all)
        runs = int(np.nansum(blobs_all['solve_evals'])) # model runs so far, the derivatives' included
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
    # the table from every draw with a solved value, the ones ruled out by another observable included (not the
    # stripped points of the plateau branch: table_values)
    table = solve.fit_table(cfg, thetas_all, solve.table_values(cfg, thetas_all, blobs_all), blobs_all['log_jac'])
    solve.save_table(table, os.path.join(run_dir, solve.TABLE))
    kind = 'warm-start table' if sv['warm_start'] else 'table (unused: solve.warm_start = false)'
    log(f'walkers start from {nwalkers} of the best {best.size} solved draws (lp {lp_all[best].max():.4g} to '
        f'{lp_all[best].min():.4g}), not jittered; {kind} of {len(table["p"])} solved points in '
        f"{solve.TABLE} (leave-one-out error of its guess of {'ln ' if table['log'] else ''}{sv['param']}: median "
        f"{np.median(np.abs(table['loo'])):.3g}; first expansion step {table['step']:.3g})")
    return thetas_all[pick].copy(), lp_all[pick].copy(), blobs_all[pick].copy()


def physical_samples(samples, blobs, cfg):
    '''
    Converts sampler positions to the physical free parameters, for the summary, the figures and the tracks
    In solve mode the solved parameter comes from the blob field solve_value, in place of u (NaN when the blobs
    lack it); without solve mode the positions are the physical parameters already.

    Inputs:
        - samples: the sampler positions (n, ndim) [array]
        - blobs: their blobs (n,) [structured array], or None
        - cfg: the run's config [dict]

    Output: the physical parameters (n, ndim), in cfg['free'] order [array]
    '''
    samples = np.asarray(samples, dtype = float)
    if not config.solve_on(cfg):
        return samples
    has = blobs is not None and 'solve_value' in blobs.dtype.names
    return solve.physical(samples, np.asarray(blobs['solve_value'], dtype = float).reshape(samples.shape[:-1])
                          if has else nan, cfg)


def init_prior(cfg, pool, nwalkers, rng, log, max_rounds = 20):
    '''
    Starts the walkers from random prior draws with a finite posterior (init = "prior")
    Each round draws 2*nwalkers plain (not Latin-hypercube) prior points, at most max_rounds rounds; the walkers
    take the first nwalkers finite ones.

    Inputs:
        - cfg: the config in the sampler's coordinates (prm.sampled_cfg) [dict]
        - pool, nwalkers, rng, log: as in init_prescan
        - max_rounds: the most rounds of draws [int]

    Output: tuple (coords, lp, blobs), as in init_prescan
    Raises RuntimeError, with advice, when fewer than nwalkers draws are finite.
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
    Starts the walkers in a small ball around the params' values (init = "ball")
    params.ball draws value + init_scale*N(0, 1) inside the prior support (init_scale defaults to 1e-2 of the
    prior box); points with lp = -inf are redrawn, at most max_rounds times.

    Inputs:
        - cfg: the config in the sampler's coordinates (prm.sampled_cfg; in solve mode u centred on the datum) [dict]
        - pool, nwalkers, rng, log: as in init_prescan
        - max_rounds: the most redraws [int]

    Output: tuple (coords, lp, blobs), as in init_prescan
    Raises RuntimeError, with advice, when some walkers still have lp = -inf.
    '''
    coords = prm.ball(cfg, nwalkers, rng)
    lp, blobs = evaluate_many(pool, coords)
    for _ in range(max_rounds): # redraw the walkers with lp = -inf
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
    Formats the data of an observable config, e.g. '1.73 +- 0.025', '-2.1 -0.1/+0.2 dex', '< 0.001, width +- 0.3 dex'
    (a model sigma adds ' (+ model s)').
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
    Formats a table of the active observables: model value, data (with the unit) and log-likelihood term

    Inputs:
        - obs: the model value of each observable [dict]
        - terms: the log-likelihood term of each observable [dict]
        - cfg: the config, whose observables are the rows [dict]

    Output: the table, one line per observable, NaN where obs or terms lack one [str]
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
    Formats a theta in the sampler's coordinates (config.sampled_names) as 'name = value, ...'.
    '''
    return ', '.join(f'{name} = {x:.5g}' for name, x in zip(config.sampled_names(cfg), theta))


def _median_text(st):
    '''
    Formats 'median +up -down' (the 68% interval, p16 to p84) of a _stats dict.
    '''
    return f"{st['p50']:>11.5g}  +{st['p84'] - st['p50']:.3g} -{st['p50'] - st['p16']:.3g}"


def summary_text(summary, cfg = None, full = False):
    '''
    Formats summary.json for reading: sampling and convergence, then the parameters (median and 68% interval)
    and the MAP

    Inputs:
        - summary: the dict of write_summary, or a summary.json read back [dict]
        - cfg: the run's config, for the data of the observables (full = True) [dict]
        - full: True adds the model settings, the timing and evaluations, the stuck-walker reset, the model runs
          per solve (solve mode), the posterior of the active observables (against the data of cfg), the MAP's
          log-likelihood terms and the warnings [Bool]

    Output: the text [str]
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
    Creates the new, empty directory of a fresh run (never an existing non-empty one); returns its absolute path
    Without run_dir: <output_root>/<name>/mcmc_<stamp>, claimed atomically (another run started in the same second
    gets _1, _2, ...); a given run_dir is created if missing and refused (FileExistsError) if it holds files.
    '''
    if run_dir is None:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        return config.new_output_dir(os.path.join(cfg['output_root'], cfg['name'], f'mcmc_{stamp}'))
    return config.empty_output_dir(os.path.abspath(run_dir), 'a run (resume = True continues it)')


def acquire_lock(run_dir):
    '''
    Takes <run_dir>/.lock for this session, so that two sessions never write the same run directory
    The lock file holds 'pid host'. It is created with O_CREAT | O_EXCL, which is atomic: of two sessions that try
    at once, exactly one gets it. A lock held by a live process on this host refuses the session; any other lock
    is stale and taken over, with a note: its process is gone, it holds this process's own pid, it was written on
    another host (whose processes cannot be checked from here), or it cannot be read.

    Inputs:
        - run_dir: the run directory [str]

    Output: tuple
        - path: the lock file [str]
        - note: None, or a message about the stale lock taken over [str]
    Raises FileExistsError when a live process on this host holds the lock (or when 3 tries fail).
    '''
    import psutil
    path, host, note = os.path.join(run_dir, LOCK), platform.node(), None
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY) # atomic: fails when the file exists
        except FileExistsError:
            try:
                with open(path) as f:
                    pid_text, _, other = f.read().strip().partition(' ') # 'pid host'
                pid = int(pid_text)
            except (OSError, ValueError): # gone meanwhile, or unreadable: stale
                pid, other = -1, ''
            # only a process on this host can be checked
            if other == host and pid != os.getpid() and pid > 0 and psutil.pid_exists(pid):
                raise FileExistsError(f'{run_dir} is in use by another session (pid {pid} on {host}): wait for it to '
                                      f'finish or stop it first. If no run is going, delete {path}') from None
            try:
                os.remove(path)
            except FileNotFoundError: # another session removed it first
                pass
            note = f'took over a stale lock of pid {pid} ({other or "unknown host"}): that session is gone'
            continue # and try again to create it
        with os.fdopen(fd, 'w') as f:
            f.write(f'{os.getpid()} {host}\n')
        return path, note
    raise FileExistsError(f'could not take the lock {path}')


def release_lock(path):
    '''
    Removes the lock file when this process holds it, never another session's; errors are ignored (no lock, or
    path None).
    '''
    try:
        with open(path) as f:
            mine = f.read().split()[:1] == [str(os.getpid())] # the pid in the file is ours
        if mine:
            os.remove(path)
    except (OSError, TypeError):
        pass


def _copy_toml(cfg, path):
    '''
    Writes config.toml: the original TOML text, preceded by a header with its path and the --set overrides
    applied on top of it

    Inputs:
        - cfg: the run's config, with config_path and overrides [dict]
        - path: the file to write [str]

    Output: True when written, False when the source TOML is unknown or gone [Bool]
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
    One session of run_mcmc: set-up or resume, the pool, initialization, the sampling loop and the finish
    Holds the config, the run directory and its lock, the backend (the chain), the sampler, and the bookkeeping
    that the checkpoints carry from session to session (extras): the reset, the record of the convergence checks,
    the evaluation tallies, the timing, the warnings and, in solve mode, the active warm-start table.
    '''

    ###_____set-up and checkpoints_____###

    def __init__(self, cfg, workers, run_dir, resume, max_steps):
        '''
        Sets up a session: checks the disk, creates the run directory (or opens the one to resume) and takes its
        lock; a fresh run writes config.json, config.toml and provenance.json, a resume loads chain.npz, adds its
        start to provenance.json and reads the bookkeeping of the checkpoint

        Inputs:
            - cfg: the config (config.load_config); None with resume = True reads the run's config.json [dict]
            - workers: the requested worker count, None for parallel.workers [int]
            - run_dir: the directory of a fresh run (None: a new <output_root>/<name>/mcmc_<stamp>), or the run to
              resume [str]
            - resume: True continues run_dir from its chain.npz [Bool]
            - max_steps: the total steps of the run, None for sampler.max_steps [int]
        '''
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
        free_gb = check_disk(run_dir if resume else cfg['output_root']) # before anything is written

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
                # provenance.json: the first session's under 'created', each resume's appended to 'resumed'
                prov_path = os.path.join(run_dir, 'provenance.json')
                prov = _read_json(prov_path, {'created': None})
                prov.setdefault('resumed', []).append(provenance())
                _dump_json(prov, prov_path)
            else:
                config.save_resolved(cfg, os.path.join(self.run_dir, 'config.json'))
                _copy_toml(cfg, os.path.join(self.run_dir, 'config.toml'))
                _dump_json({'created': provenance(), 'resumed': []}, os.path.join(self.run_dir, 'provenance.json'))
                self.backend, extra = NpzBackend(), {}
        except BaseException: # a failed set-up must not leave its lock behind
            release_lock(self.lock)
            raise

        # bookkeeping, cumulative over sessions (chain.npz extras)
        self.offset = int(_get(extra, 'reset_step', 0)) # steps before the reset (0: none)
        self.replaced = [int(i) for i in _get(extra, 'replaced', [])]
        self.reset_pending = self.s['reset_stuck_at'] > 0 and self.offset == 0 # a reset is due, once
        # per check: its step, tau, whether tau was stable, the change and tolerance of its worst coordinate and the
        # walkers the estimate used (tau_bookkeeping pads the checks of older checkpoints)
        bk = tau_bookkeeping(extra, self.ndim)
        self.tau_steps, self.tau_hist, self.tau_stable = bk['steps'], bk['tau'], bk['stable']
        self.tau_change, self.tau_tol, self.tau_walkers = bk['change'], bk['tol'], bk['walkers']
        self.converged = bool(_get(extra, 'converged', False)) # the verdict of the last check
        # the step of the first converged check (0: none); it keeps --max-steps extensions free of the stop even when a
        # later check says "not converged" (restore). A checkpoint without it: the last check's step, if converged
        self.converged_at = int(_get(extra, 'converged_at', self.tau_steps[-1] if self.converged and self.tau_steps
                                     else 0))
        self.stop = False # set by a converged check when stop_on_convergence
        self.nwalkers = self.backend.nwalkers if resume else None # a fresh run chooses it once the pool is up
        # the tallies of the previous sessions; this session's are in self.pool
        self.counts0 = {phase: np.array(_get(extra, f'counts_{phase}', np.zeros(N_STATUS)), dtype = np.int64)
                        for phase in ('init', 'sampling')}
        self.runtime0 = {phase: float(_get(extra, f'runtime_{phase}', 0.0)) for phase in ('init', 'sampling')}
        self.evals0 = {phase: np.array(_get(extra, f'evals_{phase}', np.zeros(MAX_EVALS_HIST)), dtype = np.int64)
                       for phase in ('init', 'sampling')} # histograms of the model runs per evaluation
        self.sampling_s = float(_get(extra, 'sampling_s', 0.0)) # [s] wall time in the sampling loop
        self.steps_timed = int(_get(extra, 'steps_timed', 0)) # the steps that sampling_s covers
        self.init_s = float(_get(extra, 'init_s', 0.0)) # [s] preflight and initialization
        self.wall0 = float(_get(extra, 'wall_s', 0.0)) # [s] of the previous sessions
        self.warnings = [str(w) for w in _get(extra, 'run_warnings', [])]
        self.n_workers, self.warmup_s, self.pool = None, nan, None # set once the pool is up (execute)
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
                # a checkpoint without refit_step: the step of its table (0 for none)
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
        Fills in the solve-mode options that a config.json saved before they existed lacks, with a note in run.log
        solve.derivative goes into the session's cfg, which the workers get, as the scheme the chain's last session
        used (run_derivative), so a resume never switches the target density; solve.refit_steps takes its default
        (refits due by now are caught up in restore_table).

        Inputs:
            - extra: the extras of the resumed checkpoint [dict]
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
        Steps done over the whole run, including those before a reset [steps].
        '''
        return self.offset + self.backend.iteration

    def _warn(self, text):
        '''
        Logs a warning and keeps it for the summary (run_warnings of the checkpoints).
        '''
        self.warnings.append(text)
        self.log(f'warning: {text}')

    def extras(self):
        '''
        Collects the bookkeeping that every checkpoint stores next to the chain, cumulative over sessions

        Output: dict of arrays and scalars for NpzBackend.save
            - reset_step, replaced: the step of the stuck-walker reset (0: none) and the walkers it replaced
            - tau_steps, tau, tau_stable, tau_change, tau_tol, tau_walkers: the record of the convergence checks
            - converged, converged_at: the last check's verdict, and the step of the first converged check (0: none)
            - max_steps, workers, warmup_s: the session's step limit, and its pool (or the previous session's)
            - counts_init, counts_sampling: evaluations per status code
            - runtime_init, runtime_sampling: isocalc time [s]
            - evals_init, evals_sampling: histograms of the model runs per evaluation
            - sampling_s, steps_timed, init_s, wall_s: timing [s] (steps_timed: the steps in sampling_s)
            - run_warnings: the warnings so far
            - solve mode: the active warm-start table (table_*, table_version; solve.table_extras), derivative and
              refit_step (the last refit step handled)
        '''
        # the previous sessions' tallies plus this session's
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
        Writes the backend and the bookkeeping to <run_dir>/<name> (nothing before the first step)

        Inputs:
            - name: the file name, chain.npz by default [str]

        Output: True when written [Bool]
        '''
        if self.backend.iteration == 0:
            return False
        start = time.perf_counter()
        self.backend.save(os.path.join(self.run_dir, name), **self.extras())
        self.t_saved = time.perf_counter()
        self.save_s = self.t_saved - start # [s] sets the spacing of the next periodic checkpoint (checkpoint_due)
        if name == CHAIN:
            self.saved_step = self.step
        return True

    def checkpoint_due(self, step):
        '''
        Decides whether a periodic checkpoint is due after a step
        Every checkpoint_every steps, but at most once a minute and at most every 20 times the duration of the
        last save, so that the saves cost at most 5% of a long run: each save rewrites the whole file, which grows
        with the chain (at 24 walkers about 34 MB and 1.1 s at 5000 steps), and a save of 1e5 steps takes about 20 s.

        Inputs:
            - step: the step just done, counted over the whole run [steps]

        Output: True when due [Bool]
        '''
        return (step % self.s['checkpoint_every'] == 0
                and time.perf_counter() - self.t_saved >= max(CHECKPOINT_MIN_S, CHECKPOINT_COST*self.save_s))

    def make_sampler(self):
        '''
        Builds emcee's EnsembleSampler of model.log_prob on the counting pool and the backend, with the moves of
        sampler.moves and the blob fields of model.blob_dtype()
        A fresh backend takes its RNG state from numpy's global RNG, which the caller seeds first; a loaded one
        continues with its own.

        Output: the sampler [emcee.EnsembleSampler]
        '''
        return emcee.EnsembleSampler(self.nwalkers, self.ndim, model.log_prob, pool = self.pool,
                                     moves = make_moves(self.s['moves']), backend = self.backend,
                                     blobs_dtype = model.blob_dtype())

    ###_____session_____###

    def execute(self, make_plots):
        '''
        Runs the session: logs the set-up, starts the pool, initializes (or restores) the walkers, samples,
        finishes and draws the figures
        An interrupt or error that ends the session early goes to failed (the chain saved, then the resume command
        or the traceback logged) before the pool is torn down. A resume with nothing left to sample (max_steps
        reached, or converged without a --max-steps) starts no pool: it only redoes a lost check and writes the
        summary (nothing_to_sample).

        Inputs:
            - make_plots: False skips plots.plot_run at the end [Bool]
        '''
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
        # a resume at max_steps, or of a converged run without a new --max-steps: no pool needed
        if self.resume and (self.step >= self.max_steps or (self.converged and self.s['stop_on_convergence']
                                                             and not self.max_steps_given)):
            self.nothing_to_sample()
            return
        phase, handled = 'start', False # phase: where an interrupt or error struck; handled: failed has run
        try:
            # spawned workers, each warmed up (imports and compiles) before the first map
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
                        # nwalkers = "auto": max(4 ndim, 16) rounded up to a multiple of 4 x workers (2 x without
                        # DESnooker), so that every worker is busy in each map
                        self.nwalkers = config.choose_nwalkers(cfg, wp.n)
                        self.preflight(wp)
                        self.initialize()
                    self.init_s += time.perf_counter() - start # [s]
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
        Ends a session that an interrupt (Ctrl-C; SIGTERM and SIGHUP through __main__) or an error cut short
        An interrupt checkpoints the chain, writes the summary and prints the resume command (interrupted). An
        error during the initialization or the sampling checkpoints the chain; the traceback of any error goes to
        run.log.

        Inputs:
            - phase: where it struck: 'start' (the pool), 'init', 'sampling' or 'finish' [str]
            - error: the exception [BaseException]
        '''
        if isinstance(error, KeyboardInterrupt):
            self.interrupted(phase)
        elif isinstance(error, Exception):
            if phase in ('init', 'sampling') and self.checkpoint():
                self.log(f'the chain up to step {self.step} is saved in {os.path.join(self.run_dir, CHAIN)}')
            self.log(traceback.format_exc(), echo = False) # not printed: the error goes on up and prints itself

    def nothing_to_sample(self):
        '''
        Ends a resume with nothing to sample: the chain has max_steps steps, or it converged and no --max-steps
        was given
        Redoes a convergence check the previous session lost (catch_up_check; its result is checkpointed) and
        writes the summary.
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
        Evaluates the centre (the params' values) once before the initialization, so that a bug or a model
        failure there stops the run with a clear message
        Logs the status and isocalc time, the solve in solve mode (a cold one, from both prior bounds), the model
        against the data, and a rough time per step, as if every evaluation cost as much as the centre.

        Inputs:
            - wp: the worker pool [WorkerPool]
        Raises RuntimeError on an unexpected error (status 5) or a model failure (status 2) at the centre; any other
        status but ok is a warning, since the initialization looks elsewhere anyway.
        '''
        cfg, log = self.cfg, self.log
        solving = config.solve_on(cfg)
        theta = prm.centre_theta(self.scfg) # in solve mode u at the datum
        # eval_task, not log_prob: the whole result, with the error text and the likelihood terms
        res = wp.map(model.eval_task, [{'theta': theta, 'solve': solving}])[0]
        self.pool.counts['init'][_status_index(res['status'])] += 1 # tallied by hand: wp.map is not the counting pool
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
        '''
        Starts the walkers of a fresh run with sampler.init ("solve", "prescan", "bisect", "prior" or "ball"),
        builds the sampler and sets its initial state
        The initialization's draws, numpy's global RNG and the sampler's RNG are all seeded from sampler.seed, so
        the same config and number of walkers give the same chain.
        '''
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
        self.check_unexpected() # too many bugs already: stop before sampling
        # solve mode: init = "solve" wrote the initialization's table (version 0); the other inits leave none
        path = os.path.join(self.run_dir, solve.TABLE)
        if config.solve_on(cfg) and os.path.exists(path):
            self.table, self.table_version = solve.load_table(path), 0
        np.random.seed(self.seed) # emcee seeds its RNG from the global one
        self.sampler = self.make_sampler()
        # the starting state with its lp and blobs, which emcee then does not evaluate again; its random_state
        # sets the sampler's RNG at the start of sample()
        self.initial = emcee.State(coords, log_prob = lp, blobs = blobs,
                                   random_state = np.random.RandomState(self.seed).get_state())
        self.skip_check = False # emcee checks once that the starting walkers are linearly independent

    def restore(self):
        '''
        Prepares a resume: checks the chain against the config and this version, builds the sampler on the loaded
        backend, and redoes the work of the checkpoint's step that the previous session may have missed
        The chain must have the config's dimensions and this version's blob fields. A sampler.nwalkers other than
        the chain's is ignored, a pending reset whose step has passed is dropped (with a warning), and a --max-steps
        extension of a converged run samples on without the convergence stop. Then the warm-start table is put
        back (restore_table, solve mode) and a lost convergence check redone (catch_up_check).
        Raises ValueError when the chain cannot be resumed by this version.
        '''
        b, log = self.backend, self.log
        self.nwalkers = b.nwalkers
        if b.ndim != self.ndim:
            raise ValueError(f'chain.npz has {b.ndim} parameters but the config {self.ndim}')
        # other blob fields (an older version) cannot go on in the same arrays
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
        self.initial = b.get_last_sample() # the walkers, lp, blobs and RNG state of the last step
        # the chain's own state, not a fresh start: a running ensemble may be nearly degenerate (e.g. on a thin
        # ridge), and the uninterrupted run goes on from it without emcee's start check
        self.skip_check = True
        # the work of the checkpoint's step that the previous session may not have done (the uninterrupted order:
        # refit, then check; a pending reset follows in sample)
        if config.solve_on(self.cfg):
            self.restore_table()
        self.catch_up_check()

    def restore_table(self):
        '''
        Solve mode: puts back the warm-start table that was active at the checkpoint, then redoes a refit that the
        previous session missed
        The table is written back to solve_table.npz for the workers (fresh in each session, they read it at their
        first evaluation). A checkpoint written while the run had no table (solve.NO_TABLE: init other than
        "solve", before the first refit) removes the file instead: a refit of a session that was then killed may
        have left one, and the workers must run cold, as they did. A checkpoint from before table refits existed
        holds no table: the file's (the initialization's) is used.
        Then the refits due by the checkpoint's step that were not done are redone, as one refit, from the chain up
        to that step (which is the checkpoint's, so the table is the uninterrupted run's): a refit that the end of
        the session cut short (an interrupt during build_table checkpoints the step with the old table), or the
        catch-up of a checkpoint from before refits existed.
        '''
        path = os.path.join(self.run_dir, solve.TABLE)
        if self.table_version is None: # from before refits existed
            self.table = solve.load_table(path) if os.path.exists(path) else None
            self.table_version = 0
            why = 'catch-up, the checkpoint predates table refits'
        else:
            if self.table is not None:
                solve.save_table(self.table, path)
            elif os.path.exists(path): # NO_TABLE: the workers must run cold
                os.remove(path)
            why = 'redone: the previous session ended before this refit was done'
        if any(self.refit_done < k <= self.step for k in self.refit_at):
            self.refit_table(why)

    def catch_up_check(self):
        '''
        Redoes the convergence check of the checkpoint's step when the session that wrote it did not run it
        An interrupt during the check checkpoints the step without it (and in older versions the periodic
        checkpoint of a step came before its check). The check depends only on the chain and the acceptance up to
        that step, so the resumed run checks, and stops, where the uninterrupted one did.

        Output: True when the check ran [Bool]
        '''
        b = self.backend
        # nothing to redo: no step yet, no check due at this step, or the record has its check already
        if not b.iteration or self.step % self.s['check_every'] or (self.tau_steps and self.tau_steps[-1] == self.step):
            return False
        self.log(f'the convergence check of step {self.step} did not run in the previous session: running it now')
        self.check_convergence()
        return True

    def check_unexpected(self):
        '''
        Stops the run (RuntimeError) once this session's unexpected errors (status 5: bugs, not model failures)
        exceed sampler.max_unexpected_errors; the tracebacks are in <run_dir>/worker_errors.
        '''
        n, limit = self.pool.unexpected(), self.s['max_unexpected_errors']
        if n > limit:
            raise RuntimeError(f'{n} unexpected errors (status 5: bugs, not model failures), more than '
                               f'sampler.max_unexpected_errors = {limit}. Tracebacks: '
                               f'{os.path.join(self.run_dir, "worker_errors")}')

    ###_____sampling loop_____###

    def reset_due(self):
        '''
        True when the stuck-walker reset is due now: pending, at step reset_stuck_at, and before max_steps [Bool].
        '''
        return self.reset_pending and self.step == self.s['reset_stuck_at'] and self.step < self.max_steps

    def sample(self):
        '''
        The sampling loop: steps the sampler on to max_steps, or to the convergence stop
        After each step come, in this order, the progress line, the table refit (solve mode), the convergence check,
        the periodic checkpoint and the limit on unexpected errors: a checkpoint holds the work of its step, so a
        resume continues exactly. A due reset breaks out of emcee's generator, replaces the walkers (reset_walkers)
        and starts a new generator on the fresh chain.
        '''
        s, log = self.s, self.log
        if self.step < self.max_steps and not self.stop:
            log(f'sampling: {self.nwalkers} walkers x {self.ndim} parameters, steps {self.step + 1} to '
                f'{self.max_steps}, moves "{s["moves"]}", seed {self.seed}')
        # the time, step and tallies at the last progress line: each line reports the rates since the one before
        self.mark = (time.perf_counter(), self.step, self.pool.counts['sampling'].copy(),
                     self.pool.runtime['sampling'], self.pool.evals['sampling'].copy())
        while self.step < self.max_steps and not self.stop: # a second round only after a reset
            if self.reset_due():
                self.reset_walkers()
            t_last = time.perf_counter()
            # each iteration is one emcee step: a proposal for every walker, evaluated through the pool, accepted
            # or not
            for _ in self.sampler.sample(self.initial, iterations = self.max_steps - self.step,
                                         skip_initial_state_check = self.skip_check):
                now = time.perf_counter()
                self.sampling_s += now - t_last # [s] this step, plus the bookkeeping after the step before
                self.steps_timed += 1
                self.sampled += 1
                t_last = now
                step = self.step
                if step % s['print_every'] == 0 or step == self.max_steps:
                    self.progress()
                if step in self.refit_at and step > self.refit_done: # between two steps: no evaluation is running
                    self.refit_table()
                # the check before the periodic checkpoint, so that the checkpoint holds it (a stop checkpoints in
                # finish); a check lost to an interrupt is redone at the resume (catch_up_check)
                stop = step % s['check_every'] == 0 and self.check_convergence()
                if not stop and self.checkpoint_due(step):
                    self.checkpoint()
                if stop:
                    break
                self.check_unexpected()
                if self.reset_due(): # the reset needs a new sampler: leave this generator
                    break

    def table_text(self):
        '''
        Names the active warm-start table for the log: 'no table (cold solves)', "the initialization's table" or
        'the table of step k'.
        '''
        if self.table is None:
            return 'no table (cold solves)'
        return "the initialization's table" if not self.table_version else f'the table of step {self.table_version}'

    def refit_table(self, why = None):
        '''
        Solve mode: refits the warm-start table to the initialization's solves and the chain's solved states so
        far, and replaces solve_table.npz; the workers reload it at their next evaluation
        It runs between two steps, when no evaluation is running. The chain's states lie where the posterior is, so
        the refitted table guesses each root better and the solves get cheaper: on the final LHS 1140 b chain
        (1e5/100) a catch-up refit after step 250, from 1161 solved points, cut the median error of the guess from
        0.0067 to 0.00024 dex and the model runs per solve (forward derivative) from a mean of about 10.5 to 5.6.
        An error is a warning, and the active table stays: the table changes only the path of each solve, not its
        root, so a failed refit must not end the run (nor, redone at every resume, make it impossible to resume);
        refit_done marks it handled, so that a resume does not redo it from a longer chain. An interrupt is not
        caught: the step is then checkpointed with the old table, and the resume redoes the refit (restore_table).

        Inputs:
            - why: the reason of a redone refit, for the log line [str]

        Output: True when the table was replaced [Bool]
        '''
        start = time.perf_counter()
        b, sv, step = self.backend, self.cfg['solve'], self.step
        before = np.median(np.abs(self.table['loo'])) if self.table is not None else None # the old table's error
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
        '''
        Logs a progress line: the step, the time per step, the mean acceptance, the max and median lp, the stuck and
        behind walkers (from step 50 on), the evaluations since the last line by status, in solve mode the model runs
        per solve, the isocalc time per model run, and the time left (ETA)
        '''
        b = self.backend
        now = time.perf_counter()
        t0, step0, counts0, runtime0, evals0 = self.mark
        per_step = (now - t0)/(self.step - step0) if self.step > step0 else nan # [s]
        counts = self.pool.counts['sampling'] - counts0 # since the last line
        n_eval = int(counts.sum())
        evals = self.pool.evals['sampling'] - evals0
        runs = int(np.sum(np.arange(evals.size)*evals)) # model runs: bin k holds the evaluations with k runs
        per_eval = (self.pool.runtime['sampling'] - runtime0)/runs if runs else nan # [s] isocalc only, per run
        acceptance = b.accepted/b.iteration # of each walker, over the current chain
        lp = b.log_prob[b.iteration - 1] # of each walker at this step
        stuck = '-'
        if b.iteration >= STUCK_AFTER: # the acceptance means little before
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
        Runs one convergence check: estimates tau (autocorr_info) and its noise (tau_noise), decides the stop
        (tau_rule), adds the check to the record and logs it (check_text)
        The rule: n >= min_steps, n > n_tau*max(tau), and tau stable, i.e. every walker in the estimate and tau
        changed by less than max(tau_rtol, its statistical noise), at tau_checks checks in a row (n: steps in the
        current chain).

        Output: True when the run stops (converged, with sampler.stop_on_convergence) [Bool]
        '''
        s, b = self.s, self.backend
        n = b.iteration
        acceptance = b.accepted/n
        tau, used, n_good = autocorr_info(b.chain[:n], acceptance) # the filled steps only
        noise = tau_noise(tau, used, n_good)
        previous = self.tau_hist[-1] if self.tau_hist else None
        stable, self.converged, change, tol = tau_rule(tau, previous, noise, n, s, self.tau_stable,
                                                       complete = n_good == self.nwalkers)
        if self.converged and not self.converged_at: # the first converged check
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
        Does the stuck-walker reset (sampler.reset_stuck_at = k > 0: once, at step k)
        Saves the chain so far as burnin.npz, replaces the walkers of stuck_walkers (stuck by acceptance, lp more
        than 20 below the median, or lp not finite) by jittered copies of random good ones (safe_jitter), and
        restarts with a fresh backend and sampler seeded from seed + 1. The steps go on counting from k (offset),
        and the record of the convergence checks starts over with the new chain. No walker is replaced when all of
        them look stuck, or more than half: copying the few others onto them would collapse the ensemble onto a
        few points. So many stuck walkers point to a posterior thinner than the moves can follow, e.g. a thin ridge
        (see "Solve mode" in the README).
        '''
        b, log, cfg = self.backend, self.log, self.cfg
        step = self.step
        self.checkpoint() # a resume from here redoes the same reset
        shutil.copyfile(os.path.join(self.run_dir, CHAIN), os.path.join(self.run_dir, BURNIN)) # steps 1-k
        it = b.iteration
        coords, lp, blobs = b.chain[it - 1].copy(), b.log_prob[it - 1].copy(), b.blobs[it - 1].copy() # at step k
        bad = np.flatnonzero(stuck_walkers(b.accepted/it, lp))
        good = np.setdiff1d(np.arange(self.nwalkers), bad)
        rng = np.random.default_rng(self.seed + 1) # seeded: a resume redoes the same reset
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
            source = rng.choice(good, bad.size) # a random good walker for each replaced one (with replacement)
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
        # the record of the checks belongs to the old chain
        self.tau_steps, self.tau_hist, self.converged, self.converged_at = [], [], False, 0
        self.tau_stable, self.tau_change, self.tau_tol, self.tau_walkers = [], [], [], []
        np.random.seed(self.seed + 1) # the new sampler's RNG, as in initialize
        self.sampler = self.make_sampler()
        self.initial = emcee.State(coords, log_prob = lp, blobs = blobs,
                                   random_state = np.random.RandomState(self.seed + 1).get_state())
        # the walkers of a running chain (as in restore): no start check, which a nearly degenerate ensemble on a
        # thin ridge could fail
        self.skip_check = True

    ###_____end of the session_____###

    def finish(self, wp):
        '''
        Ends a session that sampled to its end: the final checkpoint, the summary (logged, with its warnings), and
        the posterior tracks when sampler.tracks > 0 and the chain grew in this session (or tracks.npz is missing)
        A failure of the tracks is logged, not raised: the chain is safe, and `tracks RUN_DIR` retries.

        Inputs:
            - wp: the worker pool, which also runs the tracks [WorkerPool]
        '''
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
        '''
        Draws the figures of the run (plots.plot_run); a failure is logged, not raised (the chain is safe, and
        `plot RUN_DIR` retries).
        '''
        try:
            from isofate.mcmc import plots
            paths = plots.plot_run(self.run_dir)
            self.log(f'plots: {", ".join(os.path.basename(p) for p in paths or [])}')
        except Exception as e:
            self.log(f'plots failed ({type(e).__name__}: {e}); the chain is safe. Retry with: python -m '
                     f'isofate.mcmc plot {self.run_dir}')
            self.log(traceback.format_exc(), echo = False)

    def interrupted(self, phase):
        '''
        Ends a session that an interrupt cut short: checkpoints the chain, writes the summary (state
        "interrupted") and logs the resume command, with --max-steps when this session's max_steps was given or
        differs from the config's
        An interrupt after sampling (phase 'finish') only completes the final checkpoint and points to the tracks
        and plot commands; one before the first step of a fresh run leaves nothing to resume.

        Inputs:
            - phase: where it struck: 'start', 'init', 'sampling' or 'finish' [str]
        '''
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
        except Exception as e: # the chain is saved; the summary can wait for the `summary` command
            log(f'(no summary: {type(e).__name__}: {e})', echo = False)
        # an explicit --max-steps also keeps the extension of a converged run free of the stop (restore)
        given = self.max_steps_given or self.max_steps != self.cfg['sampler']['max_steps']
        more = f' --max-steps {self.max_steps}' if given else ''
        log(f'\ninterrupted at step {self.step} of {self.max_steps}; the chain is saved in '
            f'{os.path.join(self.run_dir, CHAIN)}. Resume with:\n'
            f'    python -m isofate.mcmc run --resume {self.run_dir}{more}')


def run_mcmc(cfg, workers = None, run_dir = None, resume = False, max_steps = None, make_plots = True):
    '''
    Runs the MCMC of a config and returns its run directory (see the module docstring)
    A resume gives the same chain as an uninterrupted run. Ctrl-C (and SIGTERM or SIGHUP, through __main__)
    checkpoints and prints the resume command. The lock of the run directory is released however the session ends.

    Inputs:
        - cfg: the config (config.load_config); None with resume = True reads the run's config.json [dict]
        - workers: the worker count, default parallel.workers, reduced to fit the memory [int]
        - run_dir: the directory of a new run (missing or empty; default a new <output_root>/<name>/mcmc_<stamp>),
          or the run to resume [str]
        - resume: True continues run_dir from its chain.npz [Bool]
        - max_steps: the total steps, counted from the start of the run; default sampler.max_steps [int]
        - make_plots: False skips plots.plot_run at the end [Bool]

    Output: the run directory [str]
    '''
    run = _Run(cfg, workers, run_dir, resume, max_steps)
    try:
        run.execute(make_plots)
    finally:
        release_lock(run.lock) # also after an interrupt or an error
    return run.run_dir


###_____Reading a run_____###

def load_run(run_dir):
    '''
    Reads a run directory: its config, its chain with the bookkeeping, and its summary

    Inputs:
        - run_dir: the run directory [str]

    Output: dict
        - run_dir: its absolute path [str]
        - cfg: the run's config.json [dict]
        - free: the physical free parameters [list]
        - chain: the walker positions (nsteps, nwalkers, ndim), in the sampler's coordinates [array]
        - log_prob: (nsteps, nwalkers) [array]
        - blobs: (nsteps, nwalkers) [structured array], or None
        - accepted: accepted proposals of each walker (nwalkers,) [array]
        - iteration: the steps of the chain [int]
        - extra: the bookkeeping of chain.npz [dict]
        - summary: summary.json, or None [dict]
    Raises FileNotFoundError when the run has no chain.npz yet.
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
    Returns the burn-in and thinning of a run: the summary's when it was written for this chain length, else
    the tau rule of write_summary (burn_and_thin)

    Inputs:
        - run: a load_run dict [dict]

    Output: tuple
        - discard: steps dropped at the start [steps]
        - thin: every thin-th step is kept after that [steps]
    '''
    n = run['chain'].shape[0]
    summary = run.get('summary') or {}
    if summary.get('steps') == n and summary.get('discard') is not None: # the summary is up to date
        return int(summary['discard']), max(1, int(summary.get('thin') or 1))
    tau = autocorr_time(run['chain'], run['accepted']/max(n, 1))
    return burn_and_thin(tau, n, run['cfg']['sampler']['burn_factor'])


def posterior(run, discard = None, thin = None):
    '''
    Returns the posterior samples of a run, after burn-in and thinning, every walker's together

    Inputs:
        - run: a load_run dict [dict]
        - discard, thin: the burn-in and thinning [steps]; each one None comes from discard_thin (the summary,
          else the tau rule)

    Output: tuple
        - samples: the positions (N, ndim), in the sampler's coordinates (physical_samples converts them) [array]
        - blobs: their blobs (N,) [structured array], or None when the run has none
    '''
    chain = run['chain']
    if discard is None or thin is None:
        auto = discard_thin(run)
        discard = auto[0] if discard is None else discard
        thin = auto[1] if thin is None else thin
    start = discard + thin - 1 # emcee's get_chain(discard, thin) convention
    samples = chain[start::thin].reshape(-1, chain.shape[2]) # (steps kept x nwalkers, ndim)
    blobs = run['blobs'][start::thin].reshape(-1) if run['blobs'] is not None else None
    return samples, blobs


###_____Summary_____###

def _stats(x, with_nan = False):
    '''
    Summarizes the posterior of one quantity, ignoring NaN

    Inputs:
        - x: the values, any shape [array]
        - with_nan: True adds frac_nan, the fraction of NaN values [Bool]

    Output: dict
        - p2.5, p16, p50, p84, p97.5: the percentiles; with inf values they are sample values (no interpolation)
        - mean, std: of the finite values
        - frac_nan (with_nan = True)
        Everything is NaN when x has no value that is not NaN.
    '''
    x = np.asarray(x, dtype = float).ravel()
    defined = x[~np.isnan(x)]
    out = {}
    if defined.size:
        method = 'nearest' if np.isinf(defined).any() else 'linear' # interpolating next to an inf gives inf or NaN
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
    Explains why check j of a tau_bookkeeping record did not count as stable ('' if it did).
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
    Explains why a chain whose final N/tau is above n_tau has not converged, for the summary's warning
    The answer is the condition of the stopping rule (tau_rule) that its last check failed, from that check's
    own record: e.g. N/tau, which the final chain passes, may have been below n_tau at that check (the chain
    grew past it after the check). The conditions are tried in order: tau estimated, N >= min_steps,
    N > n_tau*max(tau), tau stable at the last check (a check of the older rule reports its change against
    tau_rtol), then at the tau_checks - 1 checks before it.

    Inputs:
        - checks: the record of the checks (tau_bookkeeping) [dict]
        - s: the sampler config [dict]
        - reset_step: the step of the stuck-walker reset, 0 for none [steps]
        - n: steps of the current chain [steps]
        - nwalkers: the walkers of the ensemble [int]

    Output: the reason [str]
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
    if checks['walkers'][i] < 0 and not math.isfinite(checks['change'][i]): # padded: a check of the older rule,
        # which needed only a change below tau_rtol at one check (tau_bookkeeping)
        previous = checks['tau'][i - 1] if i else None
        if previous is not None and np.isfinite(previous).all():
            with np.errstate(all = 'ignore'):
                change = float(np.max(np.abs(tau - previous)/tau))
            return (f'tau changed by {100*change:.3g}% {at}, a check of the rule before 2026-10-29 (a change below '
                    f'tau_rtol = {100*s["tau_rtol"]:g}% at one check); a resume checks with the current rule')
        return f'the last check (step {step}) predates the noise-aware rule of 2026-10-29'
    failure = _check_failure(checks, i, nwalkers) # tau not stable at the last check
    if failure:
        return f'{failure} {at}'
    earlier = [j for j in range(max(0, i - (k - 1)), i) if not checks['stable'][j]] # of the k - 1 before it
    if earlier:
        j = earlier[-1] # the latest one that was not stable
        return (f'tau was stable at the last check (step {step}) but not at step {steps[j]} '
                f'({_check_failure(checks, j, nwalkers)}), and the rule needs {k} stable checks in a row')
    if i < k - 1: # fewer than k checks so far
        return f'tau was stable at the last check (step {step}), but the rule needs {k} stable checks in a row'
    return f'no check confirmed it (the last, step {step}, met every condition but did not count)'


def write_summary(run_dir, state = 'finished'):
    '''
    Writes <run_dir>/summary.json from chain.npz, config.json and provenance.json, and returns it
    The posterior is the final chain after the burn-in and thinning of its own tau (burn_and_thin). converged is
    the stopping rule's verdict at the last check; when it says no, a warning gives the reason (not_converged_text
    when the final chain is long enough by N/tau, else N/tau itself). More warnings flag poor mixing, stuck
    walkers, walkers that end far behind, and in solve mode stripped and atmosphere-bearing planets between which
    the walkers crossed fewer than PLATEAU_CROSSINGS times.

    Inputs:
        - run_dir: the run directory [str]
        - state: 'finished', 'interrupted' or 'partial' (the summary command, for a run that may be going) [str]

    Output: the summary [dict]
        - state, name, description, run_dir, provenance
        - settings: nwalkers, ndim, free, n_steps, n_atmodeller, backend, workers, moves, init, seed, max_steps,
          tau_checks, observables
        - steps, total_steps: steps of the current chain and of the whole run [steps]
        - reset_step, replaced_walkers: the step of the stuck-walker reset (None without one) and the walkers it
          replaced
        - timing: wall_s, warmup_s, init_s, sampling_s, s_per_step, isocalc_s_per_eval, isocalc_s_per_run [s]
        - evaluations: total, and the counts per status for init and sampling
        - acceptance: mean, min, n_stuck, behind (the walkers that end far behind), per_walker
        - tau (per coordinate), tau_max [steps], n_over_tau, tau_history (the record of the checks)
        - converged (at the last check), converged_at (the step of the first converged check, or None)
        - discard, thin [steps], n_samples
        - params: _stats of every physical parameter over the posterior (in solve mode the solved one from
          solve_value); sampled: the same for the sampler's coordinates (solve mode, else None)
        - solve: in solve mode its settings and the model runs per evaluation (solve.evals_stats; evals_init,
          evals_sampling, evals_posterior), plateau (solve.plateau) and branches (after the burn-in: the
          stripped_fraction of the samples and crossings_median, the walkers' median number of crossings between
          stripped and not), else None
        - blobs: _stats of every blob field over the posterior, with frac_nan
        - map: the sample with the highest lp in the current chain (burn-in included; in solve mode lp includes
          log_jac): step (over the whole run), walker, lp, theta (physical), sampled (solve mode), log_prior,
          log_like, log_jac, terms and obs
        - warnings: the config's, the run's and the summary's own [list of str]
    '''
    run = load_run(run_dir)
    cfg, chain, lp, blobs, extra = run['cfg'], run['chain'], run['log_prob'], run['blobs'], run['extra']
    n, nwalkers, ndim = chain.shape
    s = cfg['sampler']
    acceptance = run['accepted']/n # of each walker, over the current chain
    tau = autocorr_time(chain, acceptance) # of the final chain (the checks' estimates are in the record)
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

    # warnings: the config's and the run's, then convergence, mixing, stuck walkers and walkers behind
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
    has_reset = s['reset_stuck_at'] > 0 # the advice to set reset_stuck_at only when it is not set
    if n < STUCK_AFTER:
        warnings.append(f'only {n} steps: the acceptance and tau are not meaningful yet')
    # barely mixing: a low mean acceptance, and the chain not long against tau (N/tau <= n_tau, or no tau)
    elif acceptance.mean() < STUCK_ACCEPTANCE and not (finite_tau and n_over > s['n_tau']):
        warnings.append(f'mean acceptance {acceptance.mean():.3g} < {STUCK_ACCEPTANCE:g} ({n_stuck} of {nwalkers} '
                        'walkers stuck) and no converged tau: the moves barely mix, so the samples may still be close '
                        'to the starting points, not a posterior. A posterior much thinner than the spread of the '
                        'walkers does this, e.g. a thin curved ridge such as the LHS 1140 b outflow island; sample it '
                        'in solve mode ([solve] in the config; see "Solve mode" in the README), or run far longer')
    elif n_stuck:
        warnings.append(f'{n_stuck} walkers with acceptance < min({STUCK_ACCEPTANCE:g}, {STUCK_RELATIVE:g} x the '
                        'median) (stuck)' + ('' if has_reset else '; consider sampler.reset_stuck_at'))
    behind = np.flatnonzero(behind_walkers(lp[-1])).tolist() # at the last step
    if behind:
        one = len(behind) == 1
        warnings.append(f'{len(behind)} walker{"" if one else "s"} {behind} end{"s" if one else ""} with lp more '
                        f'than {RESET_DLP:g} below the median: lost in another region (e.g. a stripped-core '
                        f'plateau), {"it is" if one else "they are"} part of the posterior samples; run longer'
                        + ('' if has_reset else ' or set sampler.reset_stuck_at'))

    # Solve mode with the plateau branch: stripped planets and planets that keep an atmosphere lie in separate
    # stretches of u, usually with a gap between them (the jump of the observable at the stripping edge), which the
    # walkers cross rarely. tau of the sampled coordinates does not see that, so the crossings are counted.
    plateau = None
    if config.solve_on(cfg) and blobs is not None and 'exhausted' in blobs.dtype.names and discard < n:
        stripped = blobs['exhausted'][discard:] == 1 # (steps, walkers) after the burn-in
        crossings = np.sum(stripped[1:] != stripped[:-1], axis = 0) # of each walker
        plateau = {'stripped_fraction': float(stripped.mean()), 'crossings_median': float(np.median(crossings))}
        if 0.01 <= plateau['stripped_fraction'] <= 0.99 and plateau['crossings_median'] < PLATEAU_CROSSINGS:
            warnings.append(f"{100*plateau['stripped_fraction']:.0f}% of the posterior after the burn-in are stripped "
                            'planets (the plateau branch of solve mode) and the rest keep an atmosphere. In u they lie '
                            'in separate stretches, which each walker crossed only '
                            f"{plateau['crossings_median']:.0f} times (median; {PLATEAU_CROSSINGS} wanted), so the "
                            'split between them is less certain than tau says: run longer, or check it with plain '
                            "sampling (--set 'solve.param=\"\"')")

    # the MAP sample, then the posterior statistics
    i_step, i_walker = np.unravel_index(np.argmax(lp), lp.shape) # the highest lp, burn-in included
    best = blobs[i_step, i_walker]
    stored = blobs.dtype.names # a run of an older version may lack newer fields: NaN
    obs = {f: float(best[f]) if f in stored else nan for f in model.OBS_FIELDS}
    log_like, terms = likelihood.log_like(obs, cfg) # the blobs hold the observables, not the terms
    names = config.sampled_names(cfg)
    physical = physical_samples(samples, post_blobs, cfg)
    # the MAP's physical parameters (solve mode: P* from its blob)
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
                   'evals_posterior': _stats(post_blobs['solve_evals']) if 'solve_evals' in stored else None,
                   'plateau': bool(sv.get('plateau', False)), 'branches': plateau}
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
        # [s]; isocalc time per evaluation and per model run (a solve-mode evaluation makes several runs)
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
        # the MAP's step counted over the whole run, from 1; log_jac is 0 without solve mode (and without the field)
        'map': {'step': int(i_step) + 1 + reset_step, 'walker': int(i_walker), 'lp': float(lp[i_step, i_walker]),
                'theta': dict(zip(cfg['free'], map_theta.tolist())),
                'sampled': dict(zip(names, chain[i_step, i_walker].tolist())) if solving else None,
                'log_prior': float(best['log_prior']), 'log_like': float(log_like),
                'log_jac': float(best['log_jac']) if 'log_jac' in stored else 0.0, 'terms': terms, 'obs': obs},
        'warnings': warnings,
    }
    _dump_json(summary, os.path.join(run['run_dir'], SUMMARY))
    return summary
