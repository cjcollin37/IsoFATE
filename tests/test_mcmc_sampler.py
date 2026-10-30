'''
Tests of isofate.mcmc.sampler and isofate.mcmc.tracks on the toy backend with 2 spawned workers: a full run
(checkpoints, summary, tracks, plots), bit-identical resumes after a normal end and after Ctrl-C, the stuck-walker
reset, the convergence stop, the initializations and the preflight. A run may take several sessions: the first
run, then each resume.
The stopping rule rests on the integrated autocorrelation time tau, roughly the number of steps a walker needs to
forget where it was (Sokal 1997; estimated as in emcee, Foreman-Mackey et al 2013): N steps of a walker hold about
N/tau independent samples. Its tests check the noise of the estimate on AR(1) chains, the rule itself and its
bookkeeping (a check lost to an interrupt, old checkpoints, extensions of a converged run). Those run the sampler
with an in-process stand-in for the worker pool (in_process), which gives the same chains, faster.
'''

from isofate.mcmc import env

env.set_single_thread_env() # before anything imports jax

import json
import math
import os
import shutil
import subprocess
import sys
import types
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

from isofate.mcmc import config, model, sampler, solve, tracks
from isofate.mcmc import params as prm

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
STEPS = 30 # steps of the toy runs
# the example reduced to the radius and mass fit (3 free parameters, prescan init, no reset): the toy has no
# He-dominated outflow island, and these tests check the sampler, not the physics
RP_ONLY = ['params.eps.vary=false', 'params.albedo.vary=false',
           'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false',
           'solve.param=""', 'sampler.init="prescan"', 'sampler.reset_stuck_at=0']
# toy at n_steps = 2e4: its resolution bias (1e3/n_steps) still lets Rp match the data near the centre
BASE = ['model.backend="toy"', 'parallel.cache=false', 'parallel.warmup=false', 'parallel.workers=2',
        'model.n_steps=20000', 'model.n_atmodeller=100', 'sampler.nwalkers=16', f'sampler.max_steps={STEPS}',
        'sampler.checkpoint_every=10', 'sampler.print_every=10', 'sampler.check_every=10', 'sampler.tracks=0',
        *RP_ONLY]
# the arrays of chain.npz that a bit-identical resume must reproduce: the chain and the sampler's RNG state
STATE_KEYS = ('iteration', 'chain', 'log_prob', 'accepted', 'rs_name', 'rs_key', 'rs_pos', 'rs_has_gauss',
              'rs_cached')


def toy_cfg(root, *overrides):
    '''
    The toy config of BASE, writing to root, with more --set overrides
    '''
    return config.load_config(EXAMPLE, BASE + [f'output_root="{root}"', *overrides])


def chain_npz(run_dir, name = 'chain.npz'):
    '''
    The arrays of a checkpoint file of run_dir (chain.npz or burnin.npz), as a dict
    '''
    with np.load(os.path.join(run_dir, name)) as d:
        return {key: d[key] for key in d.files}


def assert_same_chain(a, b):
    '''
    Asserts that two checkpoints (chain_npz dicts) hold bit-identical chains: STATE_KEYS, and the blobs field by
    field, except the timing
    '''
    for key in STATE_KEYS:
        np.testing.assert_array_equal(a[key], b[key], err_msg = key)
    for field in model.BLOB_FIELDS:
        if field != 'runtime':
            np.testing.assert_array_equal(a['blobs'][field], b['blobs'][field], err_msg = field)


@pytest.fixture(scope = 'module')
def straight(tmp_path_factory):
    '''
    An uninterrupted 30-step run with 6 tracks and the plots, shared by the module: the reference of the resume
    tests. Returns (cfg, run_dir).
    '''
    root = tmp_path_factory.mktemp('straight')
    cfg = toy_cfg(root, 'sampler.tracks=6', 'sampler.track_points=40')
    return cfg, sampler.run_mcmc(cfg)


@pytest.fixture
def worker_globals():
    '''
    Restores the worker globals of model after a test that runs init_worker in this process
    '''
    saved = model._CFG, model._RUN_DIR, model._N_UNEXPECTED
    yield
    model._CFG, model._RUN_DIR, model._N_UNEXPECTED = saved


class SerialPool:
    '''
    An in-process stand-in for a WorkerPool: map calls f here, in order (after model.init_worker(cfg) in this
    process).
    '''

    def map(self, f, iterable):
        '''
        [f(x) for x in iterable], in this process
        '''
        return [f(x) for x in iterable]


class InProcessPool(SerialPool):
    '''
    An in-process stand-in for the WorkerPool of a sampler session (fixture in_process), with its signature and the
    attributes the sampler reads (n, warmup_s, _pool). It runs model.init_worker here and empties the warm-start
    table cache, as the fresh workers of each session start without one.
    '''

    def __init__(self, cfg, workers = None, n_steps_max = None, run_dir = None, verbose = True):
        '''
        Sets up the worker globals in this process; the arguments are those of WorkerPool.
        '''
        model.init_worker(cfg, run_dir)
        solve._TABLES.clear()
        self.n, self.warmup_s, self._pool = 1, 0.0, [] # _pool: no worker processes (worker_rss_gb)

    def __enter__(self):
        '''
        The pool itself, as WorkerPool's context manager
        '''
        return self

    def __exit__(self, *exc):
        '''
        Nothing to stop; never suppresses an exception
        '''
        return False


@pytest.fixture
def in_process(monkeypatch, worker_globals):
    '''
    Runs the sampler of one test with InProcessPool in place of WorkerPool
    '''
    monkeypatch.setattr(sampler, 'WorkerPool', InProcessPool)


def quiet(*args, **kwargs):
    '''
    A log function that prints nothing
    '''
    pass


###_____A full run_____###

def test_run_outputs(straight):
    '''
    Checks the files of a finished run and what they hold: config.json equal to the config, config.toml with the
    overrides, a chain of 30 steps x 16 walkers x 3 parameters at finite points only, lp = log_prior + log_like,
    and the summary (state, settings, provenance, one evaluation per walker and step, acceptance, discard and thin,
    the tau record, ordered percentiles, the MAP), the prescan file, and the lines of run.log.
    '''
    cfg, run_dir = straight
    assert os.path.dirname(run_dir) == os.path.join(cfg['output_root'], 'lhs1140b')
    assert os.path.basename(run_dir).startswith('mcmc_')
    for name in ('config.json', 'config.toml', 'provenance.json', 'run.log', 'prescan.npz', 'chain.npz',
                 'summary.json', 'tracks.npz'):
        assert os.path.exists(os.path.join(run_dir, name)), name
    assert config.load_resolved(os.path.join(run_dir, 'config.json')) == cfg
    assert '--set sampler.tracks=6' in open(os.path.join(run_dir, 'config.toml')).read()

    run = sampler.load_run(run_dir)
    assert run['chain'].shape == (STEPS, 16, 3) and run['iteration'] == STEPS and run['free'] == cfg['free']
    assert run['blobs'].shape == (STEPS, 16) and run['blobs'].dtype.names == model.BLOB_FIELDS
    blobs, lp = run['blobs'], run['log_prob']
    assert np.isfinite(lp).all() and (blobs['status'] == 0).all() # walkers only sit at finite points
    np.testing.assert_allclose(lp, blobs['log_prior'] + blobs['log_like'], rtol = 1e-12)
    assert np.isfinite(blobs['Rp']).all()

    s = run['summary']
    assert s['state'] == 'finished' and s['steps'] == STEPS and s['settings']['nwalkers'] == 16
    assert s['settings']['workers'] == 2 and s['settings']['free'] == cfg['free']
    assert s['provenance']['created']['versions']['emcee'] and s['provenance']['created']['git']['repo']
    assert sum(s['evaluations']['sampling'].values()) == 16*STEPS # one proposal per walker and step
    assert s['evaluations']['init']['ok'] >= 1 + 8*16 # preflight, prescan, jitter
    assert s['timing']['s_per_step'] > 0 and 0 < s['acceptance']['mean'] < 1
    assert 0 <= s['discard'] <= STEPS//2 and s['thin'] >= 1 and len(s['tau']) == 3
    assert s['tau_history']['steps'] == [10, 20, 30]
    for name in cfg['free']:
        p = s['params'][name]
        assert p['p2.5'] <= p['p16'] <= p['p50'] <= p['p84'] <= p['p97.5']
    assert s['blobs']['Rp']['frac_nan'] == 0 and set(s['blobs']) == set(model.BLOB_FIELDS)
    assert s['map']['lp'] == lp.max() and s['map']['log_like'] == pytest.approx(sum(s['map']['terms'].values()))
    samples, post_blobs = sampler.posterior(run)
    assert samples.shape == (s['n_samples'], 3) and post_blobs.shape == (s['n_samples'],)

    with np.load(os.path.join(run_dir, 'prescan.npz')) as d: # prescan_factor x nwalkers = 8 x 16 prior points
        assert d['theta'].shape == (8*16, 3) and d['lp'].shape == (8*16,) and d['start'].shape == (16,)
        lo, hi = prm.prior_box(cfg)
        assert ((d['theta'] >= lo) & (d['theta'] <= hi)).all()
    text = open(os.path.join(run_dir, 'run.log')).read()
    for line in ('preflight at the centre', 'prescan: 128', f'step {STEPS}/{STEPS}', 'tau = ', 'posterior of',
                 'tracks: 7 runs'):
        assert line in text, line


def test_tracks_file(straight):
    '''
    Checks tracks.npz: the MAP and 6 samples on one time grid (40 log-spaced times from t0 plus each sample's age,
    so that each track ends on the grid at the state the chain compared with the data), the MAP's re-run repeating
    its lp, float64 for the atom numbers N_X, NaN after each sample's age, and a file below 20 MB.
    '''
    cfg, run_dir = straight
    t = tracks.load_tracks(run_dir)
    n = 7 # the MAP + 6 samples
    ages = np.array([prm.theta_to_params(x, cfg)['age'] for x in t['theta']])
    points = np.unique(np.concatenate([np.geomspace(cfg['model']['t0'], ages.max(), 40), ages])).size
    assert t['grid_yr'].shape == (points,) and t['grid_yr'][0] == cfg['model']['t0']
    assert points > 40 and (np.diff(t['grid_yr']) > 0).all() # the 40 log-spaced times plus each sample's age
    assert t['theta'].shape == (n, 3) and t['is_map'].tolist() == [True] + [False]*6
    assert t['grid_yr'][-1] == pytest.approx(ages.max())
    s = sampler.load_run(run_dir)['summary']
    assert t['theta'][0].tolist() == list(s['map']['theta'].values())
    assert (t['status'] == 0).all() and t['final'].dtype.names == model.BLOB_FIELDS
    assert t['lp'][0] == pytest.approx(s['map']['lp'], rel = 1e-12) # the re-run repeats the chain
    for key in model.TRACK_KEYS:
        assert t[key].shape == (n, points), key
        assert t[key].dtype == (np.float64 if key.startswith('N_') else np.float32), key
    for i in range(n): # isocalc's last time is age + dt, dt = (age - t0)/n_steps
        age = prm.theta_to_params(t['theta'][i], cfg)['age']
        grid = t['grid_yr']
        assert np.isfinite(t['Rp'][i][grid <= age]).all() and np.isnan(t['Rp'][i][grid > age*1.001]).all()
        # each track ends on the grid, at its own age, with the final state the chain compared with the data
        assert age in grid and t['Rp'][i][grid == age][0] == pytest.approx(t['final']['Rp'][i], rel = 1e-3)
    assert np.isfinite(t['VMR_H2'][0][:5]).all() # save_molecules on
    assert os.path.getsize(os.path.join(run_dir, 'tracks.npz')) < 20e6


def test_plots(straight):
    '''
    Checks that the plots of the run did not fail, and that the corner plot, the traces and the tracks are there.
    '''
    pytest.importorskip('isofate.mcmc.plots') # skipped where the plotting module cannot be imported
    cfg, run_dir = straight
    log = open(os.path.join(run_dir, 'run.log')).read()
    assert 'plots failed' not in log, log[-3000:]
    for name in ('corner_params.png', 'traces.png', 'tracks.png'):
        assert os.path.exists(os.path.join(run_dir, name)), name


###_____Resume_____###

def test_resume_is_bit_identical(straight, tmp_path):
    '''
    Checks that a run stopped at step 20 and resumed to 30 gives the chain of the uninterrupted run bit for bit
    (the checkpoint holds emcee's RNG state), with the evaluations and the tau record counted over both sessions.
    A new run never overwrites an existing one, and a resume with nothing left to do adds no steps.
    '''
    cfg = toy_cfg(tmp_path)
    run_dir = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'run'), max_steps = 20, make_plots = False)
    assert chain_npz(run_dir)['iteration'] == 20
    with pytest.raises(FileExistsError): # never overwrites a run
        sampler.run_mcmc(cfg, run_dir = run_dir, make_plots = False)
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = STEPS, make_plots = False)
    assert_same_chain(chain_npz(run_dir), chain_npz(straight[1]))
    s = sampler.load_run(run_dir)['summary']
    assert s['steps'] == STEPS and s['tau_history']['steps'] == [10, 20, 30]
    assert sum(s['evaluations']['sampling'].values()) == 16*STEPS # counted over both sessions
    assert len(s['provenance']['resumed']) == 1
    # nothing left to do: no pool, no new steps
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = STEPS, make_plots = False)
    assert chain_npz(run_dir)['iteration'] == STEPS


# a run (argv: the example, its run directory, then --set overrides) that sends itself SIGINT at its 40th pool map,
# as Ctrl-C does while emcee waits for the pool
INTERRUPT_SCRIPT = '''
import os, signal, sys
from isofate.mcmc import env
env.set_single_thread_env()
from isofate.mcmc import config, pool, sampler

def interrupting(original, at_call):
    calls = [0]
    def map(self, f, iterable):
        calls[0] += 1
        if calls[0] == at_call:
            os.kill(os.getpid(), signal.SIGINT) # Ctrl-C while emcee waits for the pool
        return original(self, f, iterable)
    return map

if __name__ == '__main__':
    # Python's Ctrl-C handler even when started with SIGINT ignored (a background job of a non-interactive shell
    # inherits that, e.g. pytest run with &), else the SIGINT below would be lost
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cfg = config.load_config(sys.argv[1], sys.argv[3:])
    pool.WorkerPool.map = interrupting(pool.WorkerPool.map, 40)
    sampler.run_mcmc(cfg, run_dir = sys.argv[2], make_plots = False)
'''


def test_interrupt_and_resume(straight, tmp_path):
    '''
    Checks Ctrl-C during a run: the run checkpoints what it has, marks the summary 'interrupted' and prints the
    resume command; the resume then gives the chain of the uninterrupted run bit for bit.
    '''
    script = tmp_path/'interrupt.py'
    script.write_text(INTERRUPT_SCRIPT)
    run_dir = str(tmp_path/'run')
    proc = subprocess.run([sys.executable, str(script), EXAMPLE, run_dir] + BASE + [f'output_root="{tmp_path}"'],
                          capture_output = True, text = True, timeout = 300, cwd = tmp_path)
    assert proc.returncode != 0 and 'KeyboardInterrupt' in proc.stderr, proc.stdout + proc.stderr
    assert f'python -m isofate.mcmc run --resume {run_dir}' in proc.stdout, proc.stdout
    interrupted = chain_npz(run_dir)['iteration']
    assert 0 < interrupted < STEPS
    assert json.load(open(os.path.join(run_dir, 'summary.json')))['state'] == 'interrupted'
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, make_plots = False)
    assert_same_chain(chain_npz(run_dir), chain_npz(straight[1]))


###_____Stuck walkers, convergence_____###

def test_stuck_walkers():
    '''
    Checks the walkers a reset replaces: stuck by acceptance, lp more than 20 below the median, or lp not finite.
    behind_walkers looks at lp alone: a walker lost on a plateau still accepts its moves.
    '''
    acceptance = np.array([0.3, 0.01, 0.3, 0.3, 0.2])
    lp = np.array([-1.0, -1.5, -30.0, -np.inf, -2.0]) # finite median -1.75
    assert sampler.stuck_walkers(acceptance, lp).tolist() == [False, True, True, True, False]
    assert sampler.behind_walkers(lp).tolist() == [False, False, True, True, False] # by lp only
    assert not sampler.behind_walkers([-1.0, -2.0]).any() and sampler.behind_walkers([-np.inf]).all()


def test_stuck_warnings(straight, tmp_path):
    '''
    Checks the summary's warnings about stuck walkers. Most walkers stuck: the moves do not mix, with no reset
    advice (a reset copies good walkers) but a pointer to solve mode. A few stuck without a reset configured: the
    advice to set sampler.reset_stuck_at. A short chain: no verdict yet.
    '''
    # the 30-step chain is repeated to 60 steps (stuck walkers are judged from STUCK_AFTER = 50 steps on)
    _, run_dir = straight
    for n_stuck, mixing, advice in ((16, True, False), (2, False, True)):
        copy = str(tmp_path/f'stuck{n_stuck}')
        shutil.copytree(run_dir, copy)
        d = chain_npz(copy)
        for key in ('chain', 'log_prob', 'blobs'):
            d[key] = np.concatenate([d[key], d[key]])
        d['iteration'] = 2*d['iteration']
        # the first n_stuck walkers never accepted a move, the others every one
        d['accepted'] = np.where(np.arange(d['accepted'].size) < n_stuck, 0, d['iteration'])
        np.savez(os.path.join(copy, 'chain.npz'), **d)
        text = ' '.join(sampler.write_summary(copy)['warnings'])
        assert f'{n_stuck} of 16 walkers' in text if mixing else f'{n_stuck} walkers with acceptance' in text
        assert ('barely mix' in text) == mixing and ('consider sampler.reset_stuck_at' in text) == advice
        assert 'solve mode' in text if mixing else True
    short = sampler.write_summary(run_dir)['warnings'] # 30 steps
    assert any('only 30 steps' in w for w in short) and not any('barely mix' in w for w in short)


def test_stuck_mask_is_relative():
    '''
    Checks that the stuck threshold is relative, min(0.05, 0.2 x the median acceptance): when every walker accepts
    about 2% of its proposals they all mix (slowly) and none is stuck, while one at 0 is. tau is still estimated
    from such walkers, not NaN.
    '''
    acc = np.array([0.02, 0.018, 0.025, 0.021, 0.0])
    assert sampler.stuck_mask(acc).tolist() == [False, False, False, False, True]
    assert sampler.stuck_mask([0.3, 0.01, 0.25]).tolist() == [False, True, False] # < 0.05 and < 0.2 x median
    assert sampler.stuck_mask(np.zeros(4)).all() and sampler.stuck_mask([]).size == 0
    # tau is estimated from such walkers (an AR(1) chain of 400 steps, 8 walkers), not NaN
    rng = np.random.default_rng(0)
    x = np.zeros((400, 8, 2))
    for i in range(1, 400):
        x[i] = 0.9*x[i - 1] + rng.standard_normal((8, 2))
    tau = sampler.autocorr_time(x, np.full(8, 0.02))
    assert np.isfinite(tau).all() and np.all((tau > 5) & (tau < 60)) # tau = 19 for rho = 0.9


def test_reset_stuck(tmp_path):
    '''
    Checks the stuck-walker reset at step 10: burnin.npz keeps the first 10 steps and chain.npz the rest, the
    replaced walkers are those stuck_walkers flags, and the summary counts the steps of both. A run stopped exactly
    at the reset step and resumed does the reset at the start of the resumed session, bit-identically.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.reset_stuck_at=10')
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    burnin, chain = chain_npz(straight, 'burnin.npz'), chain_npz(straight)
    assert burnin['iteration'] == 10 and chain['iteration'] == STEPS - 10 and chain['reset_step'] == 10
    bad = sampler.stuck_walkers(burnin['accepted']/10, burnin['log_prob'][-1])
    assert chain['replaced'].tolist() == np.flatnonzero(bad).tolist()
    s = sampler.load_run(straight)['summary']
    assert s['reset_step'] == 10 and s['total_steps'] == STEPS and s['steps'] == STEPS - 10
    assert s['map']['step'] > 10 and s['replaced_walkers'] == chain['replaced'].tolist()

    # stop exactly at the reset step, then resume: the reset happens at the start of the resumed session
    part = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'part'), max_steps = 10, make_plots = False)
    assert chain_npz(part)['iteration'] == 10 and not os.path.exists(os.path.join(part, 'burnin.npz'))
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    assert_same_chain(chain_npz(part), chain)
    assert_same_chain(chain_npz(part, 'burnin.npz'), burnin)


def test_convergence_stop(tmp_path):
    '''
    Checks the convergence stop with a loose rule (n_tau = 1, tau_rtol = 100, min_steps = 20): the run stops at a
    check, a multiple of 10 steps, before max_steps = 300, with N/tau > n_tau and 'converged' in run.log.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.max_steps=300', 'sampler.n_tau=1.0', 'sampler.tau_rtol=100.0',
                  'sampler.min_steps=20')
    run_dir = sampler.run_mcmc(cfg, make_plots = False)
    s = sampler.load_run(run_dir)['summary']
    assert s['converged'] and s['steps'] < 300 and s['steps'] % 10 == 0, s['tau_history']
    assert s['n_over_tau'] > 1 and 'converged' in open(os.path.join(run_dir, 'run.log')).read()


def test_burn_and_thin():
    '''
    Checks the burn-in and thinning of the summary: discard = ceil(burn_factor x max tau), capped at half the
    chain, and thin = floor(min tau/2), at least 1 and leaving at least one step; without a finite tau, half the
    chain and no thinning.
    '''
    assert sampler.burn_and_thin([10.0, 20.0], 1000, 3.0) == (60, 5)
    assert sampler.burn_and_thin([10.0, np.nan], 1000, 3.0) == (500, 1)
    assert sampler.burn_and_thin([400.0, 400.0], 1000, 3.0) == (500, 200) # discard capped at n/2
    discard, thin = sampler.burn_and_thin([50.0, 50.0], 30, 3.0)
    assert discard == 15 and len(range(discard + thin - 1, 30, thin)) >= 1 # at least one step left


def test_stats_with_inf_and_nan():
    '''
    Checks the statistics of a blob field with inf and NaN: NaN is counted (frac_nan) and left out, the percentiles
    take sample values when there are infinities, the mean covers the finite values, and an all-NaN field gives NaN.
    '''
    st = sampler._stats([1.0, 2.0, np.inf, np.inf, np.nan], with_nan = True)
    assert st['frac_nan'] == 0.2 and st['p97.5'] == np.inf and st['p2.5'] == 1.0 and st['mean'] == 1.5
    assert all(math.isnan(v) for v in sampler._stats([np.nan, np.nan]).values())


###_____Initialization and preflight_____###

def test_initializations(tmp_path, worker_globals):
    '''
    Checks the starting positions of the walkers. init_prescan: 16 distinct walkers with a finite lp, jittered by
    less than 1% of the box from the best 2 x nwalkers of 128 prescan points. init_prior and init_ball: 16 finite
    starts. A config where nothing but a single point has a finite posterior is an error that says so.
    '''
    cfg = toy_cfg(tmp_path)
    model.init_worker(cfg)
    pool = SerialPool()
    coords, lp, blobs = sampler.init_prescan(cfg, pool, 16, np.random.default_rng(1), str(tmp_path), quiet)
    assert coords.shape == (16, 3) and np.isfinite(lp).all() and len(np.unique(coords, axis = 0)) == 16
    assert all(math.isfinite(prm.log_prior(x, cfg)) for x in coords) and (blobs['status'] == 0).all()
    with np.load(tmp_path/'prescan.npz') as d:
        lo, hi = prm.prior_box(cfg)
        assert (np.abs(coords - d['theta'][d['start']]) < 0.01*(hi - lo)).all() # jittered prescan points
        assert np.sort(d['lp'])[::-1][31] <= lp.min() + 1 # from the best 2*nwalkers
    for init in (sampler.init_prior, sampler.init_ball):
        coords, lp, blobs = init(cfg, pool, 16, np.random.default_rng(2), quiet)
        assert coords.shape == (16, 3) and np.isfinite(lp).all() and blobs.shape == (16,)

    hopeless = toy_cfg(tmp_path, 'observables.Mp.sigma=1e-200') # -inf except at Mp = 5.6 exactly
    model.init_worker(hopeless)
    with pytest.raises(RuntimeError, match = 'finite posterior'):
        sampler.init_prescan(hopeless, pool, 16, np.random.default_rng(1), str(tmp_path), quiet)


def fake_log_prob(status):
    '''
    A log_prob tuple of the given status with a runtime of 0.5 s: lp -1 for status 0, else -inf; the other blob
    fields NaN
    '''
    out = [-math.inf if status else -1.0] + [math.nan]*len(model.BLOB_FIELDS)
    out[1 + model.BLOB_FIELDS.index('status')], out[1 + model.BLOB_FIELDS.index('runtime')] = status, 0.5
    return tuple(out)


def test_counting_pool():
    '''
    Checks the tally of the sampler's counting pool: the status of every evaluation per phase (init, sampling),
    rejected proposals included, the isocalc time, and the number of unexpected errors (status 5) that
    sampler.max_unexpected_errors is checked against.
    '''
    pool = sampler._CountingPool(SerialPool()) # what max_unexpected_errors is checked against
    pool.map(fake_log_prob, [0.0, 5.0, 1.0])
    pool.phase = 'sampling'
    pool.map(fake_log_prob, [5.0, 5.0, 2.0, 0.0])
    # counts by status 0-5
    assert pool.counts['init'].tolist() == [1, 1, 0, 0, 0, 1] and pool.counts['sampling'].tolist() == [1, 0, 1, 0, 0, 2]
    assert pool.unexpected() == 3 and pool.runtime['sampling'] == 2.0
    assert sampler.status_text(pool.counts['sampling']) == 'ok 1, model failure 1, unexpected error 2'


def test_preflight_unexpected_error(tmp_path):
    '''
    Checks the preflight at the centre: a bug there (a KeyError from planet_inputs, status 5) stops the run before
    any sampling, with its traceback in worker_errors/ and no chain.npz.
    '''
    cfg = toy_cfg(tmp_path)
    del cfg['model']['rho_rcb'] # a KeyError in planet_inputs: status 5 at the centre
    run_dir = str(tmp_path/'broken')
    with pytest.raises(RuntimeError, match = 'unexpected error at the centre'):
        sampler.run_mcmc(cfg, run_dir = run_dir, make_plots = False)
    assert os.listdir(os.path.join(run_dir, 'worker_errors')) and not os.path.exists(os.path.join(run_dir, 'chain.npz'))


def test_parent_imports_stay_light():
    '''
    Checks that importing sampler and tracks loads neither jax, atmodeller, numba nor the coupled model: only the
    pool's workers do.
    '''
    code = ('import sys, isofate.mcmc.sampler, isofate.mcmc.tracks; '
            "bad = [m for m in ('jax', 'atmodeller', 'numba', 'isofate.isofate_coupler') if m in sys.modules]; "
            'assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], check = True)


def ar1_ensembles(rho, walkers, n, replicates, seed):
    '''
    Independent ensembles of independent AR(1) walkers, x[i] = rho x[i - 1] + e[i], with unit stationary variance
    and an integrated autocorrelation time tau = (1 + rho)/(1 - rho)

    Inputs:
        - rho: lag-1 autocorrelation of each walker [ndim]
        - walkers: walkers per ensemble
        - n: steps
        - replicates: number of ensembles
        - seed: seed of the noise e

    Output: chains (replicates, n, walkers, 1) [array]
    '''
    e = np.random.default_rng(seed).normal(size = (replicates, n, walkers))*math.sqrt(1 - rho**2)
    return lfilter([1.0], [1.0, -rho], e, axis = 1)[..., None] # x[i] = rho x[i - 1] + e[i]


def test_tau_noise_matches_the_scatter_of_tau():
    '''
    Checks tau_noise, the relative 1-sigma error of a tau estimate: sqrt(2 (2M + 1)/(L W)) with emcee's window
    M = 5 tau, L steps and W walkers (Sokal 1997). Over 100 replicate ensembles of independent AR(1) walkers the
    spread of emcee's estimate is a fixed fraction of it, and the estimate lies within 20% of the true tau.
    '''
    # one exact value: sqrt(2 (2 M + 1)/(L W)) with M = 5 tau
    assert sampler.tau_noise(np.array([19.0]), 1000, 24)[0] == pytest.approx(math.sqrt(2*(2*5*19 + 1)/24000))
    assert np.isnan(sampler.tau_noise(np.array([19.0]), 0, 24)).all()
    # the spread of emcee's estimate over 100 replicate ensembles of independent AR(1) walkers is a fixed fraction
    # of tau_noise (0.84 and 0.735 here; seeded, so the bands can be narrow). The bands catch a noise off by
    # sqrt(2) or 1.3x, a window of 2.5 or 10 tau instead of 5, L taken 1.5x too long, M = tau, and ignoring the
    # walkers (measured, their ratios were 0.52-0.65, 1.03-1.19, 0.90-1.03, 1.6-1.9 and 0.17-0.18)
    for rho, walkers, n, expected in ((0.9, 24, 1500, 0.84), (0.95, 16, 3000, 0.735)):
        taus = []
        for x in ar1_ensembles(rho, walkers, n, 100, seed = 1):
            tau, used, good = sampler.autocorr_info(x, np.full(walkers, 0.3))
            taus.append(tau[0])
        taus = np.array(taus)
        assert used == n - n//3 and good == walkers # the last 2/3 of the chain, every walker
        assert abs(taus.mean()/((1 + rho)/(1 - rho)) - 1) < 0.2 # the windowed estimator is biased low by ~15%
        ratio = (taus.std()/taus.mean())/sampler.tau_noise(np.array([taus.mean()]), used, good)[0]
        assert abs(ratio - expected) < 0.08, (rho, ratio)


def test_tau_rule():
    '''
    Checks one convergence check of the stopping rule (tau_rule). tau is stable when every walker is in the
    estimate and every coordinate's tau changed by less than max(tau_rtol, its noise) since the previous check;
    converged when the last tau_checks checks were stable, n >= min_steps and n > n_tau x max(tau). Also a config
    saved before tau_checks existed (2 checks), tau_rtol as a floor of the tolerance, and no previous estimate.
    '''
    s = dict(config.SAMPLER_DEFAULTS, n_tau = 50.0, tau_rtol = 0.01, tau_checks = 2, min_steps = 200)
    tau, noise = np.array([10.0, 20.0]), np.array([0.1, 0.12])
    # a change within the noise is stable, but convergence needs the previous check stable too
    stable, conv, change, tol = sampler.tau_rule(tau, tau*1.08, noise, 2000, s, [])
    assert stable and not conv and abs(change - 0.08) < 1e-12 and tol == 0.1
    assert sampler.tau_rule(tau, tau*1.08, noise, 2000, s, [False])[:2] == (True, False)
    assert sampler.tau_rule(tau, tau*1.08, noise, 2000, s, [False, True])[:2] == (True, True)
    # beyond the noise: not stable; too short (n <= n_tau tau): not converged
    assert sampler.tau_rule(tau, tau*1.25, noise, 2000, s, [True]) [:2] == (False, False)
    assert sampler.tau_rule(tau, tau, noise, 999, s, [True])[:2] == (True, False)
    assert sampler.tau_rule(tau, tau, noise, 1000, s, [True])[:2] == (True, False) # n = n_tau max(tau): not >
    # every coordinate must be stable, not just one: the first within its noise, the second 30 % off
    assert sampler.tau_rule(tau, tau*np.array([1.05, 0.7]), noise, 2000, s, [True])[0] is False
    # a change equal to the tolerance is not below it (from 11 to 10: change exactly 0.1, the noise 0.1)
    stable, conv, change, tol = sampler.tau_rule(np.array([10.0]), np.array([11.0]), np.array([0.1]), 2000, s, [True])
    assert change == tol == 0.1 and not stable and not conv
    # n < min_steps: not converged, however long the chain is compared with tau
    assert sampler.tau_rule(tau/10, tau/10, noise, 150, s, [True])[:2] == (True, False)
    # walkers left out of the estimate (stuck or frozen): never stable
    assert sampler.tau_rule(tau, tau, noise, 2000, s, [True], complete = False)[:2] == (False, False)
    # tau_checks = 3: two stable checks before this one
    s3 = dict(s, tau_checks = 3)
    assert sampler.tau_rule(tau, tau, noise, 2000, s3, [False, True])[:2] == (True, False)
    assert sampler.tau_rule(tau, tau, noise, 2000, s3, [True])[:2] == (True, False)
    assert sampler.tau_rule(tau, tau, noise, 2000, s3, [True, True])[:2] == (True, True)
    assert sampler.tau_rule(tau, tau, noise, 2000, s3, [False, True, True])[:2] == (True, True)
    # a config saved before tau_checks existed gets 2
    s_old = {k: v for k, v in s.items() if k != 'tau_checks'}
    assert sampler.tau_rule(tau, tau, noise, 2000, s_old, [True])[:2] == (True, True)
    assert sampler.tau_rule(tau, tau, noise, 2000, s_old, [False])[:2] == (True, False)
    # tau_rtol is a floor on the tolerance; one check suffices with tau_checks = 1
    s1 = dict(s, tau_rtol = 0.3, tau_checks = 1)
    assert sampler.tau_rule(tau, tau*1.25, noise, 2000, s1, [])[:2] == (True, True)
    # no previous check or no estimate: never stable
    assert sampler.tau_rule(tau, None, noise, 2000, s, [True])[:2] == (False, False)
    assert sampler.tau_rule(np.array([np.nan, 20.0]), tau, noise, 2000, s, [True])[:2] == (False, False)


###_____The stopping rule's bookkeeping_____###

class CheckRun(types.SimpleNamespace):
    '''
    A stand-in for a _Run with what _Run.check_convergence uses: the [sampler] settings (s), nwalkers, the tau
    record and the convergence flags, a log, and a backend holding the chain
    '''
    step = property(lambda self: self.offset + self.backend.iteration) # as in _Run: counted over the whole run


def check_run(nwalkers = 24):
    '''
    A CheckRun with an empty tau record and the default stopping rule (n_tau 50, tau_rtol 0.01, tau_checks 2,
    min_steps 200); its log lines collect in run.lines
    '''
    s = dict(config.SAMPLER_DEFAULTS, n_tau = 50.0, tau_rtol = 0.01, tau_checks = 2, min_steps = 200)
    run = CheckRun(s = s, nwalkers = nwalkers, offset = 0, tau_steps = [], tau_hist = [], tau_stable = [],
                   tau_change = [], tau_tol = [], tau_walkers = [], converged = False, converged_at = 0, stop = False,
                   lines = [])
    run.log = run.lines.append
    return run


def checks_on(run, chain, accepted, every = 100):
    '''
    Runs check_convergence of run every `every` steps along a chain, as the sampler would

    Inputs:
        - run: a CheckRun [CheckRun]
        - chain: (steps, walkers, ndim) [array]
        - accepted: accepted moves of each walker over the whole chain, scaled to the length of each check [array]
        - every: steps between two checks

    Output: the step of the stop, or None
    '''
    for n in range(every, len(chain) + 1, every):
        run.backend = types.SimpleNamespace(iteration = n, chain = chain[:n], accepted = accepted*n/len(chain))
        if sampler._Run.check_convergence(run):
            return n
    return None


def test_frozen_walkers_do_not_converge():
    '''
    Checks that a check counts only with every walker in the tau estimate. With 20 of 24 walkers frozen at their
    start and 4 mixing (AR(1), tau 19), the noise of tau from 4 walkers is about 40%, which every change passes,
    so the rule would stop on the 4 alone. The same chain with every walker mixing stops.
    '''
    x = ar1_ensembles(0.9, 24, 1400, 4, seed = 2)[..., 0].transpose(1, 2, 0) # (steps, walkers, 4 coordinates)
    frozen = x.copy()
    frozen[:, :20] = frozen[0, :20]
    accepted = np.where(np.arange(24) < 20, 0.0, 0.3*len(x))
    run = check_run()
    assert checks_on(run, frozen, accepted) is None and not run.converged
    assert all('20 of 24 walkers left out of tau' in line for line in run.lines[1:]), run.lines[-1]
    assert run.tau_walkers == [4]*14 and not any(run.tau_stable)
    # the same chain with every walker mixing stops
    run = check_run()
    assert checks_on(run, x, np.full(24, 0.3*len(x))) is not None and run.converged and run.converged_at
    assert run.tau_walkers[-1] == 24


def test_check_line_without_estimates():
    '''
    Checks the run.log line of a check without estimates: the first check has no change to show, and an ensemble
    without a usable walker no noise. The line says so, without numpy's All-NaN RuntimeWarning (warnings are
    errors here).
    '''
    x = ar1_ensembles(0.9, 24, 1000, 4, seed = 3)[..., 0].transpose(1, 2, 0)
    for accepted, expected in ((np.full(24, 300.0), 'change - (first check: not stable)'),
                               (np.zeros(24), '(no estimate; 0/24 walkers)')):
        run = check_run()
        run.backend = types.SimpleNamespace(iteration = 1000, chain = x, accepted = accepted)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            sampler._Run.check_convergence(run)
        assert expected in run.lines[-1] and 'nan%' not in run.lines[-1], run.lines[-1]
    assert 'change - (no estimate of tau: not stable)' in run.lines[-1]


def record(steps, stable, change, tol, walkers, tau = 10.0):
    '''
    A record of convergence checks, as tau_bookkeeping returns it, with the same tau for 3 coordinates at every check
    '''
    return {'steps': steps, 'tau': [np.full(3, tau) for _ in steps], 'stable': stable, 'change': change, 'tol': tol,
            'walkers': walkers}


def test_not_converged_text():
    '''
    Checks the summary's reason for "not converged": it names the condition of the rule that the last check failed
    (the change of tau, the checks in a row, N/tau, min_steps, walkers left out), not just the change of tau. A
    checkpoint from before the noise-aware rule has its checks padded and says so.
    '''
    s = dict(config.SAMPLER_DEFAULTS, n_tau = 50.0, tau_rtol = 0.01, tau_checks = 2, min_steps = 200,
             check_every = 100)
    why = lambda rec, n = 1250, **kw: sampler.not_converged_text(rec, dict(s, **kw), 0, n, 16)
    # the change failed
    text = why(record([1000, 1100, 1200], [True, True, False], [0.05, 0.04, 0.2], [0.1]*3, [16]*3, tau = 20.0))
    assert text.startswith('tau changed by 20% (the rule needs < 10%') and 'step 1200' in text
    # the last check passed, the one before did not
    text = why(record([1000, 1100, 1200], [True, False, True], [0.05, 0.2, 0.04], [0.1]*3, [16]*3, tau = 20.0))
    assert 'stable at the last check (step 1200) but not at step 1100 (tau changed by 20%' in text
    assert '2 stable checks in a row' in text and 'tau changed by 4' not in text
    # N/tau below n_tau at the last check (24 < 25), above it for the longer final chain
    text = why(record([1000, 1100, 1200], [True, True, True], [0.05]*3, [0.1]*3, [16]*3, tau = 50.0), n_tau = 25.0)
    assert 'N/tau = 24 at the last check (step 1200), not above n_tau = 25' in text and 'step 1300' in text
    # min_steps
    text = why(record([20, 40, 60], [False, True, True], [math.nan, 0.6, 0.4], [math.nan, 1.0, 1.0], [16]*3, tau = 1.0),
               n = 60)
    assert text.startswith('N = 60 < min_steps = 200 at the last check (step 60)')
    # walkers left out
    text = why(record([1000, 1100, 1200], [True, True, False], [0.05, 0.04, 0.04], [0.1]*3, [16, 16, 13], tau = 20.0))
    assert '3 of 16 walkers were left out of tau' in text and 'reset_stuck_at' in text
    # tau_checks = 3 with one stable check before
    text = why(record([1000, 1100, 1200], [False, True, True], [0.2, 0.04, 0.04], [0.1]*3, [16]*3, tau = 20.0),
               tau_checks = 3)
    assert 'but not at step 1000' in text and '3 stable checks in a row' in text
    # a checkpoint from before the noise-aware rule: its checks are padded (tau_bookkeeping)
    old = sampler.tau_bookkeeping({'tau_steps': np.array([1300, 1400]), 'tau': np.array([[18.0]*3, [17.0]*3])}, 3)
    assert old['stable'] == [False, False] and np.isnan(old['change']).all() and old['walkers'] == [-1, -1]
    text = sampler.not_converged_text(old, dict(s, tau_rtol = 0.05), 0, 1409, 24)
    assert 'tau changed by 5.88% at the last check (step 1400), a check of the rule before 2026-10-29' in text
    assert 'in a row' not in text


# a loose stopping rule that a toy run of up to 300 steps meets
CONVERGING = ['sampler.max_steps=300', 'sampler.n_tau=1.0', 'sampler.tau_rtol=100.0', 'sampler.min_steps=20']


def test_a_check_lost_at_the_checkpoint_is_redone(tmp_path, in_process, monkeypatch):
    '''
    Checks that a convergence check lost to an interrupt is redone. Ctrl-C during the check of step 20 checkpoints
    step 20 without it; the resume runs that check first, so the tau record, the stop and the chain are those of
    the uninterrupted run. A periodic checkpoint always holds the check of its own step.
    '''
    monkeypatch.setattr(sampler, 'CHECKPOINT_MIN_S', 0.0) # a periodic checkpoint every checkpoint_every steps
    cfg = toy_cfg(tmp_path, *CONVERGING)
    checkpoint, saved = sampler._Run.checkpoint, []

    # records (step, step of the last check) at every checkpoint
    def recording(self, name = sampler.CHAIN):
        saved.append((self.step, self.tau_steps[-1] if self.tau_steps else None))
        return checkpoint(self, name)
    monkeypatch.setattr(sampler._Run, 'checkpoint', recording)
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    assert len(saved) >= 3 and all(step == last for step, last in saved) # a periodic checkpoint holds its check
    monkeypatch.setattr(sampler._Run, 'checkpoint', checkpoint)
    estimate, calls = sampler.autocorr_info, []

    # Ctrl-C in the tau estimate of the check at step 20, once
    def interrupted(chain, acceptance):
        if chain.shape[0] == 20 and not calls:
            calls.append(1)
            raise KeyboardInterrupt
        return estimate(chain, acceptance)
    monkeypatch.setattr(sampler, 'autocorr_info', interrupted)
    part = str(tmp_path/'part')
    with pytest.raises(KeyboardInterrupt):
        sampler.run_mcmc(cfg, run_dir = part, make_plots = False)
    d = chain_npz(part)
    assert d['iteration'] == 20 and d['tau_steps'].tolist() == [10]
    monkeypatch.setattr(sampler, 'autocorr_info', estimate)
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    a, b = chain_npz(straight), chain_npz(part)
    assert_same_chain(a, b)
    for key in ('tau_steps', 'tau', 'tau_stable', 'tau_change', 'tau_tol', 'tau_walkers', 'converged', 'converged_at'):
        np.testing.assert_array_equal(a[key], b[key], err_msg = key)
    assert a['converged'] and a['iteration'] < 300
    assert 'the convergence check of step 20 did not run in the previous session' in open(
        os.path.join(part, 'run.log')).read()


def test_extension_of_a_converged_run(tmp_path, in_process):
    '''
    Checks the extension of a converged run: a bare resume has nothing to sample and says how to go on (give
    --max-steps N), while a resume with --max-steps samples on to it, without the convergence stop. The same
    extension in two sessions, the first ending at a check that said "not converged", gives the same chain.
    '''
    cfg = toy_cfg(tmp_path, *CONVERGING)
    run_dir = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'run'), make_plots = False)
    c = int(chain_npz(run_dir)['iteration'])
    s = sampler.load_run(run_dir)['summary']
    assert s['converged'] and s['converged_at'] == c < 300
    # a bare resume of a converged run has nothing to sample, and says how to go on
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, make_plots = False)
    assert f'nothing to sample: the run converged (at step {c}; {c} steps): give --max-steps N' in open(
        os.path.join(run_dir, 'run.log')).read()
    straight = str(tmp_path/'straight')
    shutil.copytree(run_dir, straight)
    sampler.run_mcmc(None, run_dir = straight, resume = True, max_steps = c + 40, make_plots = False)
    assert chain_npz(straight)['iteration'] == c + 40 # without the convergence stop
    # the same extension in two sessions, the first ending at a check that said "not converged": the second, with
    # --max-steps again, also samples on without the stop
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = c + 20, make_plots = False)
    d = chain_npz(run_dir)
    d['converged'] = np.array(False)
    np.savez(os.path.join(run_dir, 'chain.npz'), **d)
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = c + 40, make_plots = False)
    assert_same_chain(chain_npz(run_dir), chain_npz(straight))


def test_resume_of_a_checkpoint_from_before_the_noise_aware_rule(tmp_path, in_process):
    '''
    Checks the resume of a checkpoint and config.json written before the noise-aware stopping rule: tau_steps and
    tau without the per-check record, and no sampler.tau_checks. The resume pads the record, so that the summary's
    tau_history stays aligned by check, uses tau_checks = 2, and says in run.log that the option was missing.
    '''
    run_dir = sampler.run_mcmc(toy_cfg(tmp_path), run_dir = str(tmp_path/'run'), max_steps = 20, make_plots = False)
    # make it old: drop the per-check record and sampler.tau_checks
    d = chain_npz(run_dir)
    for key in ('tau_stable', 'tau_change', 'tau_tol', 'tau_walkers', 'converged_at'):
        d.pop(key)
    np.savez(os.path.join(run_dir, 'chain.npz'), **d)
    path = os.path.join(run_dir, 'config.json')
    cfg = json.load(open(path))
    del cfg['sampler']['tau_checks']
    json.dump(cfg, open(path, 'w'))
    sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = 40, make_plots = False)
    s = sampler.load_run(run_dir)['summary']
    h = s['tau_history']
    assert h['steps'] == [10, 20, 30, 40] and {len(h[k]) for k in ('tau', 'stable', 'change', 'tol', 'walkers')} == {4}
    assert h['stable'][:2] == [False, False] and np.isnan(h['change'][:2]).all() and np.isnan(h['tol'][:2]).all()
    assert h['walkers'][:2] == [-1, -1] and min(h['walkers'][2:]) > 0 and np.isfinite(h['change'][2:]).all()
    assert s['settings']['tau_checks'] == 2
    assert "sampler.tau_checks is not in this run's config.json" in open(os.path.join(run_dir, 'run.log')).read()


def test_an_interrupt_during_the_final_save(tmp_path, in_process, monkeypatch):
    '''
    Checks Ctrl-C while finish writes the last checkpoint (here the only one: the run is shorter than a minute):
    the interrupt handler writes it again, so that "the chain is complete" holds.
    '''
    save, calls = sampler.NpzBackend.save, []

    # Ctrl-C in the save of the full chain, once
    def interrupted(self, path, **extra):
        if self.iteration == STEPS and path.endswith(sampler.CHAIN) and not calls:
            calls.append(1)
            raise KeyboardInterrupt
        return save(self, path, **extra)
    monkeypatch.setattr(sampler.NpzBackend, 'save', interrupted)
    run_dir = str(tmp_path/'run')
    with pytest.raises(KeyboardInterrupt):
        sampler.run_mcmc(toy_cfg(tmp_path), run_dir = run_dir, make_plots = False)
    assert calls and chain_npz(run_dir)['iteration'] == STEPS
    assert 'interrupted after sampling: the chain is complete' in open(os.path.join(run_dir, 'run.log')).read()
