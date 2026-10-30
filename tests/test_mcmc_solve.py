'''
Tests of solve mode (SPEC amendment 2: isofate.mcmc.solve, the root finder of isofate.mcmc.search, and the solve
paths of the sampler, resolution, scan, check and the figures) on the toy backend, in one process or with pools
of at most 2 spawn workers; one real-isocalc solve (slow). The key test is test_thin_ridge_posterior: the
solve-mode marginal of the solved parameter against a brute-force grid posterior, which only holds with the right
Jacobian.
'''

from isofate.mcmc import env

env.set_single_thread_env() # before anything imports jax

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
import types

import numpy as np
import pytest

from isofate.mcmc import config, likelihood, model, plots, sampler, scan, search, solve
from isofate.mcmc import params as prm
from isofate.mcmc.__main__ import main

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
TOY = ['model.backend="toy"', 'parallel.cache=false', 'parallel.warmup=false', 'model.n_steps=20000',
       'parallel.workers=2']
# the example reduced to Rp and Mp, solved on Rp (3 free parameters: log_f_atm, Mp, age)
RP = ['params.eps.vary=false', 'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false',
      'solve.observable="Rp"', 'sampler.reset_stuck_at=0']
STATE_KEYS = ('iteration', 'chain', 'log_prob', 'accepted', 'rs_name', 'rs_key', 'rs_pos', 'rs_has_gauss',
              'rs_cached')


def toy_cfg(tmp_path, *overrides):
    return config.load_config(EXAMPLE, TOY + RP + [f'output_root="{tmp_path}"', *overrides])


def quiet(*args, **kwargs):
    pass


def chain_npz(run_dir, name = 'chain.npz'):
    with np.load(os.path.join(run_dir, name)) as d:
        return {key: d[key] for key in d.files}


def same(a, b):
    '''
    Equal arrays, NaN equal to NaN for floats (a table's lslope is NaN for a solve without a usable derivative).
    '''
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan = a.dtype.kind == 'f')


def same_table(one, two):
    return one.keys() == two.keys() and all(same(one[k], two[k]) for k in one)


def assert_same_run(a, b):
    '''
    Bit-identical runs a and b (run directories): the chain, its RNG state, every blob field but the runtime, and
    the warm-start table at the end.
    '''
    x, y = chain_npz(a), chain_npz(b)
    for key in STATE_KEYS:
        np.testing.assert_array_equal(x[key], y[key], err_msg = key)
    for field in model.BLOB_FIELDS:
        if field != 'runtime':
            np.testing.assert_array_equal(x['blobs'][field], y['blobs'][field], err_msg = field)
    assert x['table_version'] == y['table_version']
    paths = [os.path.join(run, solve.TABLE) for run in (a, b)]
    assert os.path.exists(paths[0]) == os.path.exists(paths[1])
    if os.path.exists(paths[0]):
        assert same_table(*(solve.load_table(path) for path in paths))


class InProcessPool:
    '''
    In-process stand-in for the WorkerPool of a sampler session (in_process): model.init_worker here, and an empty
    table cache, as the fresh workers of each session have.
    '''

    def __init__(self, cfg, workers = None, n_steps_max = None, run_dir = None, verbose = True):
        model.init_worker(cfg, run_dir)
        solve._TABLES.clear()
        self.n, self.warmup_s, self._pool = 1, 0.0, [] # _pool: no worker processes (worker_rss_gb)

    def map(self, f, iterable):
        return [f(x) for x in iterable]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def in_process(monkeypatch):
    saved = model._CFG, model._RUN_DIR, model._N_UNEXPECTED
    monkeypatch.setattr(sampler, 'WorkerPool', InProcessPool)
    yield
    model._CFG, model._RUN_DIR, model._N_UNEXPECTED = saved


def toy_bias(cfg):
    '''
    The toy's resolution bias of Rp (toy.isocalc).
    '''
    m = cfg['model']
    return 1 + 0.05*m['n_atmodeller']/m['n_steps'] + 1e3/m['n_steps']


###_____One solve_____###

def test_solve_reproduces_u(tmp_path):
    cfg = toy_cfg(tmp_path)
    assert cfg['sampled'] == ['u_Rp', 'Mp', 'age'] and cfg['solve']['bounds'] == pytest.approx([1.58, 1.88])
    thetas = prm.sample_solve(cfg, 30, np.random.default_rng(1))
    solved = 0
    for theta in thetas:
        res = solve.evaluate(theta, cfg)
        if res['status'] != model.OK:
            assert res['status'] == model.OUTSIDE_PRIOR and 'no log_f_atm in its prior' in res['error']
            continue
        solved += 1
        P = res['solve_value']
        assert -5 <= P <= -1.3 and abs(res['obs']['Rp'] - theta[0]) <= 0.01*0.025 # within solve.tol sigma
        assert res['lp'] == pytest.approx(res['log_prior'] + res['log_jac'] + res['log_like'], rel = 1e-12)
        assert 3 <= res['solve_evals'] <= 42 and res['runtime'] > 0
        # the blobs belong to the physical point (P*, phi): the same as evaluating it directly
        direct = model.evaluate(solve.physical(theta, P, cfg), cfg)
        assert direct['obs']['Rp'] == res['obs']['Rp'] and direct['log_like'] == res['log_like']
        blob = dict(zip(model.BLOB_FIELDS, model.blob(res)))
        assert (blob['solve_value'], blob['solve_evals'], blob['log_jac']) == (P, res['solve_evals'], res['log_jac'])
    assert solved >= 20
    # without solve mode the blob fields say so: no solved value, one model run, an identity Jacobian
    plain = model.blob(model.evaluate([-2.5, 5.6, 5e9], config.load_config(EXAMPLE, TOY + RP[:3] + [
        'solve.param=""', 'sampler.init="prescan"'])))
    assert math.isnan(plain[-3]) and plain[-2:] == (1.0, 0.0)


def test_jacobian_matches_the_analytic_derivative(tmp_path):
    # the toy's Rp = R_core b (1 + 4 sqrt(fatm)) with fatm = 10**log_f_atm x (a decay): dRp/dlog_f_atm =
    # ln(10)/2 (Rp - R_core b), so ln|dP*/du| = -ln(ln(10)/2 (Rp* - R_core b)), R_core = Mp**0.25 R_earth
    cfg = toy_cfg(tmp_path)
    b = toy_bias(cfg)
    for u, Mp, age in ((1.73, 5.6, 5e9), (1.65, 5.0, 8e9), (1.80, 6.2, 4e9)):
        res = solve.evaluate([u, Mp, age], cfg)
        assert res['status'] == model.OK
        exact = -math.log(math.log(10)/2*(res['obs']['Rp'] - Mp**0.25*b))
        assert abs(res['log_jac'] - exact) < 1e-3 # the accuracy target is 0.01
        # and the change of P* with u itself
        du = 1e-3
        up, down = (solve.evaluate([u + s*du, Mp, age], cfg)['solve_value'] for s in (1, -1))
        assert (up - down)/(2*du) == pytest.approx(math.exp(res['log_jac']), rel = 0.01)


def test_no_root_is_status_1(tmp_path):
    cfg = toy_cfg(tmp_path)
    # at Mp = 6.7 the smallest radius with an atmosphere is R_core b = 1.69 R_earth and the bare core 1.609: no
    # log_f_atm gives Rp = 1.6
    res = solve.evaluate([1.6, 6.7, 5e9], cfg)
    assert res['status'] == model.OUTSIDE_PRIOR and res['lp'] == -math.inf and math.isnan(res['solve_value'])
    assert 'no log_f_atm in its prior gives u_Rp = 1.6' in res['error'] and res['solve_evals'] == 2 # the two bounds
    outside = solve.evaluate([1.5, 5.6, 5e9], cfg) # u below its support (6 sigma)
    assert outside['status'] == model.OUTSIDE_PRIOR and outside['solve_evals'] == 0
    lp, blobs = model.evaluate_many(model.LocalPool(cfg), [[1.6, 6.7, 5e9], [1.73, 5.6, 5e9]])
    assert lp[0] == -math.inf and blobs['status'].tolist() == [1.0, 0.0] and np.isfinite(lp[1])
    # physical evaluations through the same pool do not solve
    lp, blobs = model.evaluate_many(model.LocalPool(cfg), [[-2.5, 5.6, 5e9]], physical = True)
    assert blobs['solve_evals'][0] == 1 and np.isnan(blobs['solve_value'][0])


###_____The root finder_____###

def synthetic(f, lo = -5.0, hi = 0.0, target = 0.0, tol = 0.01, sigma = 1.0, **kwargs):
    '''
    search.find_root on g = f(x) - target in [lo, hi] (no model): returns (root, history).
    '''
    st = {'lo': lo, 'hi': hi, 'log': False, 'index': 0, 'ocfg': {'sigma': sigma, 'sigma_model': 0.0}}

    def run(theta):
        value = f(theta[0])
        return value - target, True, value
    return search.find_root(run, [math.nan], st, target, tol, kwargs.pop('max_evals', 60), **kwargs)


def test_root_finder():
    # smooth and convex: Illinois needs far fewer steps than bisection (12 halvings for 5/2**12 ~ 1e-3)
    root, hist = synthetic(lambda x: math.exp(2*x) - 0.05)
    assert root.outcome == search.CONVERGED and abs(math.exp(2*root.x) - 0.05) <= 0.01 and root.evals <= 12
    assert hist[0][0] == hist[1][0] == 'ends' and root.sign_a == -1
    # undefined (NaN) below -2: the NaN end takes the sign opposite to the defined end
    root, _ = synthetic(lambda x: math.nan if x < -2 else x + 1.5)
    assert root.outcome == search.CONVERGED and abs(root.x + 1.5) <= 0.01 and root.nan_sign == -1
    # a step across the target: the bracket closes without a root
    root, _ = synthetic(lambda x: -1.0 if x < -1.234 else 1.0)
    assert root.outcome == search.JUMP and root.b - root.a < 1e-9*5 and root.a < -1.234 <= root.b
    # undefined inside a bracket whose ends are defined
    root, _ = synthetic(lambda x: math.nan if -3 < x < -2 else x + 2.5, lo = -5, hi = 0)
    assert root.outcome == search.HOLE
    # no sign change, and undefined at both ends
    assert synthetic(lambda x: x + 10)[0].outcome == search.ABOVE
    assert synthetic(lambda x: x - 10)[0].outcome == search.BELOW
    assert synthetic(lambda x: math.nan)[0].outcome == search.UNDEFINED
    # the evaluation limit
    assert synthetic(lambda x: math.exp(2*x) - 0.05, max_evals = 3)[0].outcome == search.UNCONVERGED
    # warm starts: from a guess, toward the root (orient = sign of dg/dx)
    root, hist = synthetic(lambda x: x + 1.5, guess = -1.45, step = 0.01, orient = 1)
    assert root.outcome == search.CONVERGED and abs(root.x + 1.5) <= 0.01 and [h[0] for h in hist][:2] == ['guess',
                                                                                                         'expand']
    assert root.evals <= 6
    # the wrong orientation: the first step takes g away from the target, so the expansion turns around
    root, hist = synthetic(lambda x: x + 1.5, guess = -1.45, step = 0.01, orient = -1)
    assert root.outcome == search.CONVERGED and root.reversed and root.evals <= 6
    # a Newton first step from the expected slope lands next to the root
    root, hist = synthetic(lambda x: 3*(x + 1.5), guess = -1.4, step = 0.01, orient = 1, slope = 3.0)
    assert root.outcome == search.CONVERGED and root.evals <= 4
    # a guess between an undefined region and the root, on the wrong side (as on the LHS 1140 b stripping edge):
    # stepping into the undefined region does not make a bracket there; the far bound decides, as in the cold search
    f = lambda x: math.nan if x < -2.0 else x + 1.5 # undefined below -2, the root at -1.5, defined g < 0 between
    root, hist = synthetic(f, guess = -1.9, step = 0.2, orient = -1)
    assert root.outcome == search.CONVERGED and abs(root.x + 1.5) <= 0.01 and 'far' in [h[0] for h in hist]
    cold, _ = synthetic(f)
    assert abs(cold.x - root.x) <= 0.02 # the same root
    # g > 0 wherever it is defined: the far bound has the guess's sign, so the undefined region takes the opposite
    # one (the NaN-end rule) and the bracket closes on its edge, a jump, as in the cold search
    h = lambda x: math.nan if x < -2.0 else x + 2.5
    root, hist = synthetic(h, guess = -1.0, step = 0.2, orient = 1)
    assert root.outcome == search.JUMP == synthetic(h)[0].outcome and 'far' in [step[0] for step in hist]
    # no root on either side of the guess
    assert synthetic(lambda x: x + 10, guess = -2.0, step = 0.1, orient = 1)[0].outcome == search.ABOVE
    # a guess where g is undefined: the cold search from both bounds
    root, hist = synthetic(lambda x: math.nan if x < -2 else x + 1.5, guess = -3.0, step = 0.1, orient = 1)
    assert root.outcome == search.CONVERGED and [h[0] for h in hist][:3] == ['guess', 'ends', 'ends']


def test_wind_fractions_need_an_h_he_wind():
    # a metal-dominated wind (H + He < half of the escaping mass) counts as no H/He wind for the root finder
    o = {'space': 'log10', 'floor': 1e-15, 'ceiling': 1e15, 'sigma': 0.25, 'sigma_model': 0.0}
    metal = {'outflow_H_fraction': 1e-3, 'Mdot_HHe': 1.0, 'Mdot': 10.0}
    assert np.isnan(search.g_of(metal, 'outflow_H_fraction', o, -3.0)[0])
    helium = dict(metal, Mdot_HHe = 9.0)
    assert search.g_of(helium, 'outflow_H_fraction', o, -3.0)[0] == pytest.approx(0.0, abs = 1e-12)
    blobs = np.array([tuple(metal.values()), tuple(helium.values())],
                     dtype = [('outflow_H_fraction', 'f8'), ('Mdot_HHe', 'f8'), ('Mdot', 'f8')])
    assert np.isnan(search.search_values(blobs, 'outflow_H_fraction')).tolist() == [True, False]
    assert search.search_values(blobs, 'Mdot').tolist() == [10.0, 10.0] # other observables as they are


def wind_model(p, metal = True):
    '''
    A stand-in for log_f_atm -> final outflow (as at the LHS 1140 b stripping edge with dynamic_phi): no wind below
    -3.3; a metal-dominated wind up to -3.0 whose H fraction swings across the data (x = 1, then 1e-7); then the
    H/He island, log10 x = -7 + 20 (p + 3), up to x = 1. metal = False gives the metal band an H/He wind instead.
    '''
    if p < -3.3:
        return {'outflow_H_fraction': math.nan, 'Mdot_HHe': 0.0, 'Mdot': 0.0}
    if p < -3.0:
        return {'outflow_H_fraction': 1.0 if p < -3.1 else 1e-7, 'Mdot_HHe': 1e-3 if metal else 1e5, 'Mdot': 1e5}
    return {'outflow_H_fraction': min(10**(-7 + 20*(p + 3)), 1.0), 'Mdot_HHe': 1e5, 'Mdot': 1e5}


def test_metal_wind_edge_is_the_stripped_side():
    # the root finder skips the metal-wind band at the stripping edge and converges on the H/He island root
    o = {'space': 'log10', 'floor': 1e-15, 'ceiling': 1e15, 'sigma': (0.261, 0.240), 'sigma_model': 0.0}
    target = math.log10(0.00115)
    island = -3 + (7 + target)/20
    st = {'lo': -5.0, 'hi': -1.3, 'log': False, 'index': 0, 'ocfg': o}

    def search_with(metal, **warm):
        def run(theta):
            obs = wind_model(theta[0], metal)
            return float(search.g_of(obs, 'outflow_H_fraction', o, target)[0]), True, obs
        return search.find_root(run, [math.nan], st, target, 0.01, 60, **warm)[0]
    root = search_with(True)
    assert root.outcome == search.CONVERGED and abs(root.x - island) < 1e-3 and root.nan_sign == -1
    # warm starts from either side of the band land on the same root
    for guess in (-3.2, -3.05, -2.7):
        warm = search_with(True, guess = guess, step = 0.05, orient = 1)
        assert warm.outcome == search.CONVERGED and abs(warm.x - island) < 1e-3, guess
    # the test has power: read as an H/He wind, the band's swing ends the same search at its edge (no root)
    assert search_with(False).outcome != search.CONVERGED


###_____Initialization, the warm-start table, resume_____###

def test_init_solve_and_table(tmp_path):
    cfg = toy_cfg(tmp_path)
    pool = model.LocalPool(cfg)
    coords, lp, blobs = sampler.init_solve(cfg, pool, 16, np.random.default_rng(1), str(tmp_path), quiet)
    assert coords.shape == (16, 3) and len(np.unique(coords, axis = 0)) == 16 and np.isfinite(lp).all()
    assert all(len(np.unique(coords[:, i])) == 16 for i in range(3)) # distinct in every coordinate, u included
    assert np.all((coords[:, 0] >= 1.58) & (coords[:, 0] <= 1.88)) and (blobs['status'] == 0).all()
    with np.load(tmp_path/'solve_init.npz') as d:
        assert d['theta'].shape[0] >= 48 and list(d['free']) == ['u_Rp', 'Mp', 'age']
    table = solve.load_table(str(tmp_path/solve.TABLE))
    assert table['x'].shape[1] == 3 and table['k'] == 8 and table['param'] == 'log_f_atm' and table['step'] > 0
    guess, step, orient, slope = solve.predict(table, coords[0])
    assert -5 <= guess <= -1.3 and orient == 1 # Rp grows with log_f_atm
    assert 0.05 < slope < 5 # dRp/dlog_f_atm [R_earth/dex], from the neighbours' log_jac
    # warm starts need fewer model runs and land within the tolerance of the cold roots
    cold = [solve.evaluate(theta, cfg) for theta in coords]
    warm = [solve.evaluate(theta, cfg, table = table) for theta in coords]
    assert np.mean([w['solve_evals'] for w in warm]) < np.mean([c['solve_evals'] for c in cold])
    assert all(abs(w['obs']['Rp'] - theta[0]) <= 0.01*0.025 for w, theta in zip(warm, coords))
    # without warm starts the table is still written, and the log says that it is unused
    cold_cfg, lines = toy_cfg(tmp_path, 'solve.warm_start=false'), []
    (tmp_path/'cold').mkdir()
    sampler.init_solve(cold_cfg, model.LocalPool(cold_cfg), 4, np.random.default_rng(1), str(tmp_path/'cold'),
                       lines.append)
    assert 'table (unused: solve.warm_start = false) of' in lines[-1] and 'warm-start' not in lines[-1]


def test_solve_table_determinism(tmp_path):
    cfg = toy_cfg(tmp_path)
    thetas = prm.sample_solve(cfg, 40, np.random.default_rng(3))
    solved = [solve.evaluate(theta, cfg) for theta in thetas]
    values = np.array([r['solve_value'] for r in solved])
    log_jac = np.array([r['log_jac'] for r in solved])
    one, two = (solve.fit_table(cfg, thetas, values, log_jac) for _ in range(2))
    for key in one:
        a, b = np.asarray(one[key]), np.asarray(two[key])
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan = a.dtype.kind == 'f'), key
    path = solve.save_table(one, str(tmp_path/solve.TABLE))
    loaded = solve.load_table(path)
    point = [1.7, 5.4, 6e9]
    assert solve.predict(loaded, point) == solve.predict(one, point)
    a, b = (model.blob(solve.evaluate(point, cfg, table = loaded)) for _ in range(2))
    runtime = model.BLOB_FIELDS.index('runtime')
    assert a[:runtime] + a[runtime + 1:] == b[:runtime] + b[runtime + 1:]
    # the same in another process (a worker of a resumed run)
    code = ('from isofate.mcmc import env; env.set_single_thread_env()\n'
            'import json, sys\nfrom isofate.mcmc import config, model, solve\n'
            f'cfg = config.load_resolved({str(tmp_path/"cfg.json")!r})\n'
            f'res = solve.evaluate({point!r}, cfg, table = solve.load_table({path!r}))\n'
            'print(json.dumps([res["solve_value"], res["log_jac"], res["lp"], res["solve_evals"]]))')
    config.save_resolved(cfg, str(tmp_path/'cfg.json'))
    out = subprocess.run([sys.executable, '-c', code], capture_output = True, text = True, timeout = 120, check = True)
    there = json.loads(out.stdout.strip().splitlines()[-1])
    blob = dict(zip(model.BLOB_FIELDS, a))
    assert there == [blob['solve_value'], blob['log_jac'], blob['log_prior'] + blob['log_jac'] + blob['log_like'],
                     blob['solve_evals']]


def test_resume_is_bit_identical_in_solve_mode(tmp_path):
    # table refits before (5, 15) and after (25) the interruption at step 20
    cfg = toy_cfg(tmp_path, 'sampler.nwalkers=16', 'sampler.max_steps=30', 'sampler.tracks=0',
                  'sampler.check_every=10', 'sampler.print_every=10', 'solve.refit_steps=[5, 15, 25]')
    straight = sampler.run_mcmc(cfg, workers = 2, run_dir = str(tmp_path/'straight'), make_plots = False)
    part = sampler.run_mcmc(cfg, workers = 2, run_dir = str(tmp_path/'part'), max_steps = 20, make_plots = False)
    assert chain_npz(part)['iteration'] == 20
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    a, b = chain_npz(straight), chain_npz(part)
    for key in STATE_KEYS:
        np.testing.assert_array_equal(a[key], b[key], err_msg = key)
    for field in model.BLOB_FIELDS:
        if field != 'runtime':
            np.testing.assert_array_equal(a['blobs'][field], b['blobs'][field], err_msg = field)
    t1, t2 = solve.load_table(os.path.join(straight, solve.TABLE)), solve.load_table(os.path.join(part, solve.TABLE))
    assert same_table(t1, t2)
    # the sampling evaluations were warm starts: fewer model runs than the cold solves of the initialization
    s = sampler.load_run(part)['summary']
    assert s['solve']['evals_sampling']['mean'] < s['solve']['evals_init']['mean']
    # at least the root and the forward derivative
    assert np.isfinite(a['blobs']['solve_value']).all() and np.all(a['blobs']['solve_evals'] >= 2)
    # each checkpoint holds the table active at its step, here the refit after step 25
    table, version = solve.table_from_extras(sampler.NpzBackend.load(os.path.join(part, 'chain.npz')).extra)
    assert version == 25 and same_table(table, t1)
    log = open(os.path.join(part, 'run.log')).read()
    assert log.count('warm-start table refitted after step') == 3


def test_refit_table_and_forward_derivative(tmp_path):
    cfg = toy_cfg(tmp_path, 'sampler.nwalkers=16', 'sampler.max_steps=40', 'sampler.tracks=0',
                  'sampler.check_every=20', 'sampler.print_every=20', 'solve.refit_steps=[10, 20]')
    run = sampler.run_mcmc(cfg, workers = 2, run_dir = str(tmp_path/'run'), make_plots = False)
    c = chain_npz(run)
    init = solve.build_table(cfg, run)
    refit = solve.build_table(cfg, run, c['chain'][:20], c['blobs'][:20])
    final = solve.load_table(os.path.join(run, solve.TABLE))
    assert same_table(refit, final) # deterministic
    assert len(refit['p']) > len(init['p']) and np.median(np.abs(refit['loo'])) < np.median(np.abs(init['loo']))
    # forward and central Jacobians agree on the smooth toy
    thetas = prm.sample_solve(cfg, 6, np.random.default_rng(4))
    central = dict(cfg, solve = dict(cfg['solve'], derivative = 'central'))
    for theta in thetas:
        f, c2 = solve.evaluate(theta, cfg, table = final), solve.evaluate(theta, central, table = final)
        if f['status'] == model.OK:
            assert abs(f['log_jac'] - c2['log_jac']) < 1e-3
            assert c2['solve_evals'] - f['solve_evals'] in (0, 1) # 0: forward fell back to P* - h (2 runs)
    with pytest.raises(config.ConfigError, match = 'derivative'):
        toy_cfg(tmp_path, 'solve.derivative="backward"')
    assert toy_cfg(tmp_path, 'solve.refit_steps=[]')['solve']['refit_steps'] == []


REFITS = ['sampler.nwalkers=16', 'sampler.max_steps=30', 'sampler.tracks=0', 'sampler.check_every=10',
          'sampler.print_every=10', 'solve.refit_steps=[5, 15, 25]']


def test_an_interrupted_refit_is_redone(tmp_path, in_process, monkeypatch):
    # Ctrl-C (or SIGTERM, SIGHUP) while the table is refitted after step 15 checkpoints step 15 with the table of
    # step 5: the resume redoes that refit before step 16, as the uninterrupted run did it
    cfg = toy_cfg(tmp_path, *REFITS)
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    build = solve.build_table

    def interrupted(cfg, run_dir, chain = None, blobs = None):
        if chain is not None and len(chain) == 15:
            raise KeyboardInterrupt
        return build(cfg, run_dir, chain, blobs)
    monkeypatch.setattr(solve, 'build_table', interrupted)
    part = str(tmp_path/'part')
    with pytest.raises(KeyboardInterrupt):
        sampler.run_mcmc(cfg, run_dir = part, make_plots = False)
    d = chain_npz(part)
    assert d['iteration'] == 15 and d['table_version'] == 5
    monkeypatch.setattr(solve, 'build_table', build)
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    assert_same_run(straight, part)
    assert 'refitted after step 15 (redone' in open(os.path.join(part, 'run.log')).read()


def test_a_failed_refit_keeps_the_table(tmp_path, in_process, monkeypatch):
    # an error in a refit is a warning: the run goes on with the table it has, and a resume does not redo the refit
    # from its longer chain
    build = solve.build_table

    def failing(cfg, run_dir, chain = None, blobs = None):
        if chain is not None and len(chain) == 15:
            raise np.linalg.LinAlgError('singular matrix')
        return build(cfg, run_dir, chain, blobs)
    monkeypatch.setattr(solve, 'build_table', failing)
    cfg = toy_cfg(tmp_path, *REFITS)
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    part = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'part'), max_steps = 20, make_plots = False)
    d = chain_npz(part)
    assert d['table_version'] == 5 and d['refit_step'] == 15
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    assert_same_run(straight, part)
    s = sampler.load_run(straight)['summary']
    assert s['steps'] == 30 and chain_npz(straight)['table_version'] == 25
    assert any('refit after step 15 failed (LinAlgError: singular matrix)' in w for w in s['warnings'])


@pytest.mark.parametrize('init', ['prescan', 'solve'])
def test_resume_after_a_hard_kill(tmp_path, in_process, init):
    # A session killed (SIGKILL, out of memory, power loss) after the refit at step 5 but before its next checkpoint
    # leaves the table of step 5 on disk next to a checkpoint of step 4. Before step 5 the workers had the
    # initialization's table (init = "solve", written back from the checkpoint), or none at all (init = "prescan":
    # table_version -1, so the file is removed and they run cold): the resume repeats steps 5 on as they were
    cfg = toy_cfg(tmp_path, f'sampler.init="{init}"', *REFITS)
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    part = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'part'), max_steps = 4, make_plots = False)
    assert chain_npz(part)['table_version'] == (0 if init == 'solve' else solve.NO_TABLE)
    shutil.copyfile(os.path.join(straight, solve.TABLE), os.path.join(part, solve.TABLE)) # a later refit's table
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    assert_same_run(straight, part)
    log = open(os.path.join(straight, 'run.log')).read()
    if init == 'prescan': # without solve_init.npz the refits use the chain alone, and there was no table before
        assert 'warm-start table refitted after step 5: ' in log and '(chain only)' in log
        assert not any('(was' in line for line in log.splitlines() if 'refitted after step 5:' in line)


def test_configs_from_before_the_options(tmp_path):
    # configs saved before solve.derivative and solve.refit_steps existed (2026-10-29) were sampled with the central
    # difference; a resume keeps the scheme of the run's last session (sampler.run_derivative)
    cfg = toy_cfg(tmp_path)
    old = copy.deepcopy(cfg)
    del old['solve']['derivative'], old['solve']['refit_steps']
    assert solve.settings(cfg)['derivative'] == 'forward' and solve.settings(old)['derivative'] == 'central'
    assert solve.refit_steps(old) == [25, 50, 100, 200, 400] == config.SOLVE_DEFAULTS['refit_steps']
    assert sampler.run_derivative(cfg, {}) == 'forward'
    assert sampler.run_derivative(old, {}) == 'central' # no table in the checkpoint: sampled before refits existed
    assert sampler.run_derivative(old, {'table_version': np.array(400)}) == 'forward' # e.g. the final LHS 1140 b chain
    assert sampler.run_derivative(old, {'table_version': np.array(400), 'derivative': np.array('central')}) == 'central'


def test_an_old_central_run_resumes_central(tmp_path, in_process):
    # a run of 2026-10-28: sampled with the central difference, its config.json without solve.derivative and
    # solve.refit_steps, its checkpoint without the table. Resumed, it goes on with the central difference (and the
    # default refits), as the same run in one session with the central difference
    cfg = toy_cfg(tmp_path, 'solve.derivative="central"', 'sampler.nwalkers=16', 'sampler.max_steps=30',
                  'sampler.tracks=0', 'sampler.check_every=10', 'sampler.print_every=10')
    straight = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'straight'), make_plots = False)
    part = sampler.run_mcmc(cfg, run_dir = str(tmp_path/'part'), max_steps = 20, make_plots = False)
    path = os.path.join(part, 'config.json')
    old = json.load(open(path))
    del old['solve']['derivative'], old['solve']['refit_steps']
    json.dump(old, open(path, 'w'))
    d = chain_npz(part)
    for key in [k for k in d if k.startswith('table_')] + ['derivative', 'refit_step']:
        d.pop(key)
    np.savez(os.path.join(part, 'chain.npz'), **d)
    sampler.run_mcmc(None, run_dir = part, resume = True, make_plots = False)
    assert model._CFG['solve']['derivative'] == 'central' # what the workers used
    x, y = chain_npz(straight), chain_npz(part)
    for key in STATE_KEYS:
        np.testing.assert_array_equal(x[key], y[key], err_msg = key)
    for field in ('log_jac', 'solve_value', 'solve_evals', 'log_like'):
        np.testing.assert_array_equal(x['blobs'][field], y['blobs'][field], err_msg = field)
    log = open(os.path.join(part, 'run.log')).read()
    assert 'the Jacobian uses the "central" difference, as in its last session' in log
    assert "solve.refit_steps is not in this run's config.json" in log and 'refitted after step 25' in log
    assert y['derivative'] == 'central' and sampler.load_run(part)['summary']['solve']['derivative'] == 'central'


def test_forward_derivative_sides():
    # the forward difference runs the model at P* + h; at the upper prior bound, or where T is undefined at P* + h,
    # at P* - h (2 runs in the second case); the central difference runs both sides
    u, h = 0.0, 1e-3
    st = {'h': h, 'index': 0, 'lo': -5.0, 'hi': 0.0, 'derivative': 'forward', 'observable': 'Rp', 'param': 'P'}

    def derivative(T, x, settings = st):
        calls = []

        def run(p):
            calls.append(float(p[0]))
            return T(p[0]) - T(x), True, None
        root = types.SimpleNamespace(x = x, g = 0.0, a = None, sign_a = 0.0)
        deriv, n, why = solve._derivative(run, np.array([x]), settings, root, u)
        return deriv, n, why, calls
    line = lambda p: 2*p + 1
    deriv, n, why, calls = derivative(line, -0.5)
    assert deriv == pytest.approx(2) and n == 1 and why is None and calls == [pytest.approx(-0.5 + h)]
    deriv, n, why, calls = derivative(line, 0.0) # at the upper bound
    assert deriv == pytest.approx(2) and n == 1 and calls == [pytest.approx(-h)]
    deriv, n, why, calls = derivative(lambda p: math.nan if p > -0.5 else line(p), -0.5) # undefined above
    assert deriv == pytest.approx(2) and n == 2 and calls == [pytest.approx(-0.5 + h), pytest.approx(-0.5 - h)]
    deriv, n, why, calls = derivative(lambda p: math.nan, -0.5)
    assert math.isnan(deriv) and n == 2 and 'undefined on both sides' in why
    deriv, n, why, calls = derivative(line, -0.5, dict(st, derivative = 'central'))
    assert deriv == pytest.approx(2) and n == 2 and sorted(calls) == [pytest.approx(-0.5 - h), pytest.approx(-0.5 + h)]
    old = {k: v for k, v in st.items() if k != 'derivative'} # settings() always sets it; central without it
    assert derivative(line, -0.5, old)[1] == 2


###_____The key test: the posterior on a thin curved ridge_____###

RIDGE_RP, RIDGE_SIGMA = 2.75, 2e-4 # [R_earth]


def ridge_cfg(root, *overrides):
    '''
    The toy with log_f_atm and Mp free and a radius known to 2e-4 R_earth: a ridge 0.0003-0.0008 dex thin in
    log_f_atm that curves from -1.4 (Mp = 4) to -3.1 (Mp = 8); the slope dRp/dlog_f_atm, and with it the Jacobian,
    changes by 2.7x along it.
    '''
    raw = {'name': 'ridge', 'output_root': str(root), 'star': {'mass': 0.18, 'luminosity': 0.0038},
           'planet': {'period': 24.74},
           'model': {'backend': 'toy', 'n_steps': 2000, 'n_atmodeller': 0, 'step_fn': True},
           'params': {'log_f_atm': {'vary': True, 'bounds': [-4.0, -0.5]}, 'Mp': {'vary': True, 'bounds': [4.0, 8.0]},
                      'age': 2e9, 'F_final': 0.17, 't_sat': 3.128e9},
           'observables': {'Rp': {'use': True, 'value': RIDGE_RP, 'sigma': RIDGE_SIGMA}},
           'parallel': {'workers': 2, 'cache': False, 'warmup': False},
           'sampler': {'nwalkers': 16, 'check_every': 100, 'tracks': 0, 'print_every': 100, 'checkpoint_every': 100}}
    return config.load_config(raw, list(overrides))


def grid_posterior(cfg, n_mp = 160, n_lf = 81, k = 7.0):
    '''
    The posterior over (log_f_atm, Mp) on a grid, by brute force: Mp at n_mp cell midpoints, and at each a grid of
    n_lf points across the ridge (+-k sigma of Rp around its root, found once), each weighted by exp(lp) x its
    cell. No Jacobian is involved. Returns (log_f_atm, weight) of every grid point.
    '''
    st = dict(search.bracket(cfg, 'log_f_atm'), ocfg = cfg['observables']['Rp'])
    lfs, weights = [], []
    for Mp in 4.0 + (np.arange(n_mp) + 0.5)*4.0/n_mp:
        def run(theta):
            res = model.evaluate(theta, cfg)
            return res['obs']['Rp'] - RIDGE_RP, res['status'] in search.RAN, res
        root, _ = search.find_root(run, [math.nan, Mp], st, 0.0, 1e-3, 80)
        assert root.outcome == search.CONVERGED
        h = 1e-6
        up, down = (model.evaluate([root.x + s*h, Mp], cfg)['obs']['Rp'] for s in (1, -1))
        slope = (up - down)/(2*h)
        grid = root.x + np.linspace(-k, k, n_lf)*RIDGE_SIGMA/abs(slope)
        lp = np.array([model.evaluate([x, Mp], cfg)['lp'] for x in grid])
        lfs.append(grid)
        weights.append(np.exp(lp)*(grid[1] - grid[0]))
    return np.concatenate(lfs), np.concatenate(weights)


def weighted_quantiles(x, w, q):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5*w)/np.sum(w) # midpoint rule
    return np.interp(q, cdf, x)


def test_thin_ridge_posterior(tmp_path):
    Q = np.array([0.16, 0.5, 0.84])
    # plain sampling in log_f_atm, started on the ridge (init = "bisect"): the moves stall
    plain = ridge_cfg(tmp_path, 'sampler.init="bisect"', 'sampler.bisect_observable="Rp"', 'sampler.max_steps=150')
    acc_plain = sampler.load_run(sampler.run_mcmc(plain, workers = 2, make_plots = False))['summary']['acceptance']
    assert acc_plain['mean'] < 0.05
    # solve mode: the sampler moves in (u = Rp, Mp) and mixes
    solved = ridge_cfg(tmp_path, 'solve.param="log_f_atm"', 'sampler.max_steps=1000')
    assert solved['sampled'] == ['u_Rp', 'Mp'] and solved['sampler']['init'] == 'solve'
    run = sampler.load_run(sampler.run_mcmc(solved, workers = 2, make_plots = False))
    s = run['summary']
    assert s['acceptance']['mean'] > 0.15
    samples, blobs = sampler.posterior(run)
    P = blobs['solve_value']
    assert np.isfinite(P).all() and np.allclose(samples[:, 0], blobs['Rp'], atol = 0.01*RIDGE_SIGMA)
    # the marginal of log_f_atm against the brute-force grid posterior, within the Monte Carlo error
    lf, w = grid_posterior(solved)
    expected = weighted_quantiles(lf, w, Q)
    tau = max(s['tau'].values())
    assert np.isfinite(tau)
    ess = len(P)*s['thin']/tau # effective sample size

    def density(x0, h = 0.03): # of the grid posterior at x0 [1/dex]
        return np.sum(w[np.abs(lf - x0) < h])/np.sum(w)/(2*h)
    se = np.sqrt(Q*(1 - Q)/ess)/np.array([density(x) for x in expected]) # [dex]
    found = np.percentile(P, 100*Q)
    assert np.all(np.abs(found - expected) < 4*se), (found, expected, se)
    # the test has power: without the Jacobian (each sample reweighted by |dT/dP| = exp(-log_jac)) the marginal is
    # several standard errors off
    wrong = weighted_quantiles(P, np.exp(-blobs['log_jac']), Q)
    assert np.max(np.abs(wrong - expected)/se) > 5, (wrong, expected, se)
    # the summary reports the physical parameter from the solves
    assert s['params']['log_f_atm']['p50'] == pytest.approx(np.median(P)) and 'u_Rp' in s['sampled']


###_____Commands and outputs in solve mode_____###

def toy_file(tmp_path, name = 'toy', solve_on = True, **sampler_options):
    '''
    A toy-backend copy of the LHS 1140 b example in tmp_path with Rp and Mp only, solved on Rp (solve_on).
    '''
    with open(EXAMPLE, 'rb') as f:
        raw = tomllib.load(f)
    raw['name'] = name
    raw['output_root'] = str(tmp_path)
    raw['model'].update(backend = 'toy', n_steps = 20000)
    raw['params']['eps'] = {'value': 0.15}
    raw['observables']['outflow_H_fraction']['use'] = raw['observables']['Mdot_HHe']['use'] = False
    raw['solve'] = {'param': 'log_f_atm' if solve_on else '', 'observable': 'Rp'}
    raw['parallel'] = {'workers': 2, 'cache': False, 'warmup': False}
    raw['sampler'].update(max_steps = 20, checkpoint_every = 10, print_every = 10, check_every = 10, tracks = 4,
                          track_points = 40, nwalkers = 16, reset_stuck_at = 0,
                          init = 'solve' if solve_on else 'prescan', **sampler_options)
    path = tmp_path/f'{name}.toml'
    path.write_text(config.dump_toml(raw))
    return str(path)


def test_bare_core_plateau(tmp_path, capsys):
    # a stripped planet has Rp = the bare core for every log_f_atm below the stripping edge: a plateau that solve
    # mode cannot represent. The config warns when it lies inside the u support, and check points it out.
    toi = os.path.join(config.CONFIG_DIR, 'toi406b.toml')
    cfg = config.load_config(toi, ['solve.param="log_f_atm"', f'output_root="{tmp_path}"'])
    warn = [w for w in cfg['warnings'] if 'bare-core' in w]
    assert len(warn) == 1 and '(1-1.682 R_earth over the Mp prior)' in warn[0] and '-1.5 sigma' in warn[0]
    assert not [w for w in config.load_config(toi, [f'output_root="{tmp_path}"'])['warnings'] if 'bare-core' in w]
    assert not [w for w in config.load_config(EXAMPLE)['warnings'] if 'bare-core' in w] # solved on the outflow
    # solve.plateau on a synthetic search: equal T at three log_f_atm (the plateau), then the rising branch
    st = solve.settings(cfg)
    flat = {'status': model.OK, 'terms': {'Rp': -1.07}}
    rising = {'status': model.OK, 'terms': {'Rp': -0.01}}
    hist = [('ends', -5.0, 1.4621 - 1.55, flat), ('ends', -1.3, 2.9461 - 1.55, rising),
            ('refine', -4.2, 1.4621 - 1.55, flat), ('refine', -2.92, 1.4621 - 1.55, flat),
            ('refine', -2.77, 1.5526 - 1.55, rising)]
    assert solve.plateau(hist, st, 1.55) == pytest.approx((1.4621, -5.0, -2.92, -1.07))
    assert solve.plateau(hist[1:3] + hist[4:], st, 1.55) is None # one evaluation on it: not seen as a plateau
    assert solve.plateau(hist, dict(st, bounds = (1.5, 1.9)), 1.55) is None # outside the u support
    # check on the toy (Rp datum 1.6 +- 0.06, so the bare core 1.538 at Mp = 5.6 lies inside the u support; at
    # 10 Gyr the toy is stripped below log_f_atm -4.83, where its radius jumps from 1.618 to the bare core)
    path = toy_file(tmp_path, 'plateau')
    assert main(['check', path, '--theta', 'age=1e10', '--set', 'observables.Rp.value=1.6', '--set',
                 'observables.Rp.sigma=0.06']) == 1 # no root: the jump
    out = capsys.readouterr().out
    assert 'note: a plateau: Rp gives T = 1.53832 at every log_f_atm evaluated from -5 to -4.8' in out
    assert 'bare-core radius (1.456-1.609 R_earth over the Mp prior)' in out # the config warning


def test_check_solve_cli(tmp_path, capsys):
    path = toy_file(tmp_path)
    assert main(['check', path, '--theta', 'age=5e9']) == 0
    out = capsys.readouterr().out
    assert 'Solve mode: log_f_atm in [-5, -1.3] solved for u_Rp = 1.73' in out and 'solved: log_f_atm* =' in out
    assert 'ln|dlog_f_atm*/du|' in out and 'lo ' in out and 'model runs per solve' in out
    assert main(['check', path, '--theta', 'u_Rp=1.75', '--theta', 'age=5e9']) == 0
    assert 'solved for u_Rp = 1.75' in capsys.readouterr().out
    assert main(['check', path, '--theta', 'u_Rp=1.6', '--theta', 'Mp=6.7']) == 1 # no root
    assert 'no solve: status 1' in capsys.readouterr().out
    assert main(['check', path, '--theta', 'u_Rpp=1.7']) == 2 # not the sampled coordinate
    assert main(['check', toy_file(tmp_path, 'plain', solve_on = False), '--theta', 'u_Rp=1.7']) == 2


def test_run_outputs_in_solve_mode(tmp_path, capsys):
    path = toy_file(tmp_path)
    assert main(['run', path]) == 0
    (run_dir,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    for name in ('solve_init.npz', solve.TABLE, 'summary.json', 'tracks.npz', 'corner_params.png', 'corner_sampled.png',
                 'traces.png', 'predictive.png', 'tracks.png'):
        assert os.path.exists(os.path.join(run_dir, name)), name
    run = sampler.load_run(run_dir)
    s = run['summary']
    assert list(s['params']) == ['log_f_atm', 'Mp', 'age'] and list(s['sampled']) == ['u_Rp', 'Mp', 'age']
    assert s['solve']['param'] == 'log_f_atm' and s['solve']['evals_sampling']['n'] > 0
    assert s['map']['theta']['log_f_atm'] == run['blobs']['solve_value'].ravel()[np.argmax(run['log_prob'])]
    np.testing.assert_allclose(run['log_prob'], run['blobs']['log_prior'] + run['blobs']['log_jac']
                               + run['blobs']['log_like'], rtol = 1e-12)
    log = open(os.path.join(run_dir, 'run.log')).read()
    assert 'solve init (batch 1)' in log and 'warm-start table' in log and 'model runs/solve mean' in log
    assert 'tracks: 5 runs' in log and '5/5 reproduce' in log # re-run at (P*, phi) without solving
    with np.load(os.path.join(run_dir, 'tracks.npz')) as t:
        assert np.allclose(t['theta'][0, 1:], run['chain'].reshape(-1, 3)[np.argmax(run['log_prob'])][1:])
        assert t['theta'][0, 0] == s['map']['theta']['log_f_atm'] # physical
    capsys.readouterr()
    assert main(['summary', run_dir]) == 0
    out = capsys.readouterr().out
    assert 'solve mode: log_f_atm solved from u = T(Rp)' in out and 'u_Rp (sampled)' in out
    # compare a solve-mode run with a plain run of the same planet: the physical parameters line up
    plain = toy_file(tmp_path, 'plain', solve_on = False)
    assert main(['run', plain, '--no-plots']) == 0
    (plain_dir,) = [os.path.join(tmp_path, 'plain', d) for d in os.listdir(tmp_path/'plain')]
    out_dir, paths = plots.compare_runs([run_dir, plain_dir], out_dir = str(tmp_path/'cmp'))
    text = open(os.path.join(out_dir, 'compare.txt')).read()
    row = next(line for line in text.splitlines() if line.strip().startswith('log_f_atm'))
    assert 'fixed' not in row and row.count('+') == 2 # a median for both runs
    assert os.path.exists(os.path.join(out_dir, 'compare_corner.png'))


GRID = {'n_steps': [20000, 50000], 'n_atmodeller': [100, 10]} # the toy's Rp bias leaves roots for u at >= 2e4


def test_resolution_solve_mode(tmp_path, capsys):
    from isofate.mcmc import resolution
    cfg = toy_cfg(tmp_path)
    assert config.resolution_source(cfg) == 'solve'
    out = resolution.run_resolution(cfg, grid = GRID, points = 2, workers = 2)
    printed = capsys.readouterr().out
    assert 'solve mode: each cell solves for log_f_atm' in printed and 'model runs per solve measured' in printed
    rec = json.load(open(os.path.join(out, 'recommendation.json')))
    assert rec['solve'] and rec['source'] == 'solve' and rec['recommended'] is not None
    with np.load(os.path.join(out, 'results.npz')) as r:
        d_rp = r['d'][:, :, list(r['obs_names']).index('Rp')]
        assert np.nanmax(d_rp[r['used']]) <= 0.03 # solved at every cell: about 0 by construction
        assert r['solve_shift'].shape == r['d'].shape[:2] and np.isfinite(r['shift_blobs']['Rp']).all()
        assert np.all(r['blobs']['solve_evals'][..., 0] >= 3) and bool(r['solve'])
    with np.load(os.path.join(out, 'points.npz')) as p:
        assert list(p['free']) == ['u_Rp', 'Mp', 'age'] and list(p['physical_free']) == ['log_f_atm', 'Mp', 'age']
        assert np.all(np.abs(p['theta_perturbed'][:, 1:] - p['theta'][:, 1:]) == 0) # u is perturbed
    table = open(os.path.join(out, 'table.txt')).read()
    assert 'Solve mode: every cell solves for log_f_atm again' in table and 'worst shift of log_f_atm*' in table
    # without solve mode on the same config (--no-solve): physical test points, bisection on Rp
    plain = resolution.run_resolution(toy_cfg(tmp_path, 'sampler.bisect_observable="Rp"'), grid = GRID, points = 2,
                                      workers = 2, out_dir = str(tmp_path/'plain'), solve_mode = False)
    rec = json.load(open(os.path.join(plain, 'recommendation.json')))
    assert rec['source'] == 'bisect' and not rec['solve']


def test_scan_reports_monotonicity(tmp_path, capsys):
    path = toy_file(tmp_path)
    assert main(['scan', path, '--x', 'log_f_atm=-4:-1.5:26', '--theta', 'age=5e9', '--workers', '1']) == 0
    out = capsys.readouterr().out
    assert 'solve check: Rp along log_f_atm inside the u range' in out and 'monotonic on every line' in out
    (one,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    assert 'monotonic' in str(scan.load_scan(one)['solve_check'][0])
    # a line that turns inside the u range, and one with two stretches inside it
    p = np.linspace(0, 1, 11)
    stretches, turns = scan._monotonic_lines(p, np.array([1, 1.5, 2, 2.5, 3, 2.5, 2, 1.5, 2, 2.5, 3]), 0.5, 3.5)
    assert stretches == [(p[0], p[10])] and turns == [p[4], p[7]]
    stretches, _ = scan._monotonic_lines(p, np.array([0, 1, 2, 9, 9, 9, 9, 1, 2, 9, 9.0]), 0.5, 3.5)
    assert stretches == [(p[1], p[2]), (p[7], p[8])]
    # plateaus: the bare core inside the u range is reported, x = 1 outside it is not
    line = np.array([1.46, 1.46, 1.46, 1.5, 1.6, 1.7, 1.8, 2.0, 2.0, 2.0, 2.0])
    assert scan._plateau_lines(p, line, 1.2, 1.9, 1e-9) == [(p[0], p[2], 1.46)]
    assert scan._plateau_lines(p, np.array([np.nan]*3 + list(line[3:])), 1.2, 1.9, 1e-9) == []
    # at 10 Gyr the toy is stripped below log_f_atm -4.83 (Rp = the bare core 1.538, inside [1.24, 1.96])
    assert main(['scan', path, '--x', 'log_f_atm=-5:-4:11', '--theta', 'age=1e10', '--workers', '1', '--set',
                 'observables.Rp.value=1.6', '--set', 'observables.Rp.sigma=0.06']) == 0
    out = capsys.readouterr().out
    assert 'but Rp is constant (a plateau) over part of log_f_atm on 1 of 1 line(s)' in out
    assert 'plateau: Rp = 1.53832 for log_f_atm -5 to -4.9' in out


###_____Real isocalc (slow)_____###

@pytest.mark.slow
def test_real_solve_lhs1140b(tmp_path):
    # one solve on the outflow H fraction of LHS 1140 b at n_steps = 2000, in this process
    cfg = config.load_config(EXAMPLE, ['model.n_steps=2000', 'model.n_atmodeller=20', 'parallel.cache=false',
                                       'parallel.warmup=false', f'output_root="{tmp_path}"'])
    u = math.log10(0.00115)
    res = solve.evaluate([u, 5.6, 7.6e9, 0.15], cfg, history = True)
    assert res['status'] == model.OK and -3.2 < res['solve_value'] < -2.8
    assert abs(math.log10(res['obs']['outflow_H_fraction']) - u) <= 0.01*0.261
    assert math.isfinite(res['log_jac']) and res['log_jac'] < -3 # dT/dlog_f_atm of order 100 or more
    assert res['solve_evals'] <= 41 and len(res['history']) == res['solve_evals'] - 1 # the forward derivative
    assert res['obs']['Mdot_HHe'] >= 0.5*res['obs']['Mdot'] # an H/He wind
