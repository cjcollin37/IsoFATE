'''
Tests of isofate.mcmc.resolution, the study that finds the cheapest n_steps and n_atmodeller whose observables are
converged, on the toy backend (pools of at most 2 spawned workers). The toy's radius carries a resolution bias of
about 1e3/n_steps (relative), so the study has something to find.
Covered: the quick grid end to end, the pass/stable/recommendation rules on hand-made blobs, the metric
conventions, the grid and the plan, the test-point sources (given points, a chain, the bisection), the noise floor
and the equivalent shift.
'''

import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from isofate.mcmc import config, likelihood, model, resolution
from isofate.mcmc.resolution import distance, relative_change

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
# the files of a study's output folder
FILES = ('config.json', 'points.npz', 'results.npz', 'table.txt', 'recommendation.json', 'resolution_heatmap.png',
         'resolution_convergence.png')
TINY = {'n_steps': [2000, 5000], 'n_atmodeller': [100, 10]} # a 2 x 2 grid


# the example reduced to the radius and mass fit of 3 free parameters
RP_ONLY = ['params.eps.vary=false', 'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false',
           'solve.param=""', 'sampler.init="prescan"', 'sampler.reset_stuck_at=0'] # without solve mode


def toy_cfg(tmp_path, *overrides):
    '''
    The LHS 1140 b example on the toy, reduced to its radius and mass fit (RP_ONLY), writing to tmp_path, with
    more --set overrides
    '''
    return config.load_config(EXAMPLE, ['model.backend=toy', 'parallel.cache=false', 'parallel.warmup=false',
                                        f'output_root="{tmp_path}"', *RP_ONLY, *overrides])


def test_quick_grid_toy(tmp_path, capsys):
    '''
    Checks a whole study on the quick grid (4 x 3 cells, 2 test points): every output file, the printed
    recommendation, the verdicts and the arrays of results.npz and points.npz. The toy's radius bias makes every
    cell below 3e5 steps fail, the fiducial 1e5/100 included, and the recommendation a cell at 3e5 steps.
    '''
    cfg = toy_cfg(tmp_path)
    out = resolution.run_resolution(cfg, grid = 'quick', points = 2, workers = 2)
    assert os.path.dirname(out) == os.path.join(str(tmp_path), 'lhs1140b')
    assert os.path.basename(out).startswith('resolution_')
    for name in FILES:
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    printed = capsys.readouterr().out
    assert 'RECOMMENDATION:' in printed and 'Recommended: --set model.n_steps=' in printed

    rec = json.load(open(os.path.join(out, 'recommendation.json')))
    assert rec['reference']['n_steps'] == 300000 and rec['reference']['n_atmodeller'] == 10
    # the toy's Rp bias 1e3/n_steps is 0.4-7 sigma below 3e5 steps, 0.01 sigma at 3e5
    best = rec['recommended']
    assert best['verdict'] == 'stable' and best['n_steps'] == 300000 and best['n_atmodeller'] in (1000, 100)
    assert best['worst_d'] <= 0.1 and best['speedup_vs_reference'] > 1
    assert rec['fiducial']['verdict'] == 'fail'
    verdicts = {(c['n_steps'], c['n_atmodeller']): c['verdict'] for c in rec['cells']}
    assert all(verdicts[ns, na] == 'fail' for ns in (10000, 30000, 100000) for na in (1000, 100, 10))

    table = open(os.path.join(out, 'table.txt')).read()
    assert best['options'] in table and best['options'] == (f"--set model.n_steps={best['n_steps']} --set "
                                                           f"model.n_atmodeller={best['n_atmodeller']}")
    assert 'How to read this' in table and 'Noise floor' in table and 'smooth' in table

    with np.load(os.path.join(out, 'results.npz')) as r:
        # blobs (points, cells, 2): each run and its perturbed twin of the noise floor
        assert r['blobs'].shape == (2, 12, 2) and r['blobs'].dtype == np.dtype(model.blob_dtype())
        assert np.all(r['blobs']['status'] == model.OK)
        assert r['d'].shape == (2, 12, 1) and list(r['obs_names']) == ['Rp'] # Mp is an input, not compared
        assert int(r['reference']) == 11 and r['stable'][int(r['recommended'])] # the reference: 3e5/10
    with np.load(os.path.join(out, 'points.npz')) as p:
        assert p['theta'].shape == (2, 3) and p['candidates'].shape == (40, 3) # 20 prescan points per test point
        # the noise floor moves log_f_atm alone, by 1e-6 of its prior width (3.7 dex)
        step = p['theta_perturbed'] - p['theta']
        assert np.all(step[:, 1:] == 0) and np.allclose(np.abs(step[:, 0]), 1e-6*3.7)
    assert config.load_resolved(os.path.join(out, 'config.json'))['resolution']['points'] == 2


def _blobs(cfg, cells, Rp, Rp_perturbed = None, runtime = None):
    '''
    Hand-made blobs of a study, as resolution.analyse reads them: status ok, Mp = 5.6 and the log-likelihood of
    each run

    Inputs:
        - cfg: the config, whose observables give the log-likelihood [dict]
        - cells: the cells of the grid (not used)
        - Rp: the radius at each (point, cell) [R_earth]
        - Rp_perturbed: the radius of the perturbed runs (default Rp) [R_earth]
        - runtime: time per evaluation, a number or one per cell (default 1) [s]

    Output: blobs (points, cells, 2) of model.blob_dtype(); [..., 1] is the perturbed run [array]
    '''
    Rp = np.asarray(Rp, dtype = float)
    blobs = np.empty(Rp.shape + (2,), dtype = model.blob_dtype())
    blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
    for k, values in enumerate((Rp, Rp if Rp_perturbed is None else Rp_perturbed)):
        blobs['status'][..., k] = model.OK
        blobs['Rp'][..., k] = values
        blobs['Mp'][..., k] = 5.6
        blobs['log_like'][..., k] = [[likelihood.log_like({'Rp': x, 'Mp': 5.6}, cfg)[0] for x in row]
                                     for row in values]
        blobs['runtime'][..., k] = 1.0 if runtime is None else runtime
    return blobs


def test_verdicts_and_recommendation(tmp_path):
    '''
    Checks the verdicts on hand-made blobs of a 2 x 2 grid: a cell passes within tol (0.1 sigma), is stable when
    every finer cell passes too, and the recommendation is the fastest stable cell, not the fastest passing one,
    which may be close to the reference by chance. An Rp undefined at one resolution only is an infinite distance;
    a test point whose reference run failed is left out.
    '''
    cfg = toy_cfg(tmp_path) # Rp = 1.730 +- 0.025, Mp (input)
    cells = resolution.plan_cells(cfg, [1000, 10000], [100, 10])
    ref = resolution.reference_index(cells)
    assert (cells[ref]['n_steps'], cells[ref]['n_atmodeller']) == (10000, 10)
    # cells: (1e3, 100) close by chance, (1e3, 10) off by 0.4 sigma, (1e4, 100) within 0.04 sigma, the reference
    Rp = [[1.7305, 1.740, 1.731, 1.730], [1.7302, 1.7301, 1.7305, 1.730]]
    runtime = np.array([0.1, 0.5, 0.3, 2.0]) # [s]
    m = resolution.analyse(cfg, cells, _blobs(cfg, cells, Rp, runtime = runtime), ref, 0.1, 0.5)
    assert m['verdict'] == ['pass', 'fail', 'stable', 'reference']
    assert m['recommended'] == 2 # the fastest passing cell is not stable: a finer cell fails
    assert math.isclose(m['worst_d'][1], 0.01/0.025) and m['worst_obs'][1] == 0 and m['worst_point'][1] == 0
    assert np.allclose(m['mean_runtime'], runtime)
    assert m['worst_noise'][ref] == 0

    # undefined at one resolution only (NaN Rp at a coarse cell): inf, flagged; a failed reference run: point dropped
    Rp[1][0] = math.nan
    blobs = _blobs(cfg, cells, Rp, runtime = runtime)
    blobs['status'][1, 0, 0] = model.MODEL_FAILED
    m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5)
    assert np.isinf(m['d'][1, 0, 0]) and m['changed'][1, 0, 0] and m['verdict'][0] == 'fail'
    assert m['n_failed'][0] == 1
    blobs['status'][0, ref, 0] = model.MODEL_FAILED
    m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5)
    assert list(m['used']) == [False, True]


def test_metric_conventions():
    '''
    Checks the conventions of the metrics: distance is 0 between two NaN or two equal infinities and inf where only
    one value is finite; relative_change is NaN between two NaN, and inf against 0 or where only one value is
    finite. Also the farthest-point choice of test points, which starts from the first (best) one, and the short
    form of n_steps in the tables (1e4, 3e5).
    '''
    nan, inf = math.nan, math.inf
    assert list(distance([nan, nan, 1.0, inf, inf, 3.0], [nan, 2.0, nan, inf, 5.0, 1.0], 2.0)) == [0, inf, inf, 0,
                                                                                                  inf, 1.0]
    r = relative_change([nan, 1.0, 0.0, 2.0, 1.0], [nan, nan, 0.0, 1.0, 0.0])
    assert math.isnan(r[0]) and list(r[1:]) == [inf, 0.0, 1.0, inf]
    assert resolution.farthest_points([[0.5], [0.0], [1.0], [0.55], [0.45]], 3) == [0, 1, 2]
    assert [resolution.fmt_n(n) for n in (10000, 300000, 1000000, 2000, 12345)] == ['1e4', '3e5', '1e6', '2000',
                                                                                    '12345']


def test_grid_plan_and_dry_run(tmp_path, capsys):
    '''
    Checks the grid and the plan: the quick and default grids, an explicit n_steps list (sorted, without
    duplicates), the errors for n_steps above 1e6 and for an unknown grid, and n_atmodeller = [0] when uncoupled.
    Cells with n_atmodeller > n_steps, or estimated above max_eval_seconds, are skipped; the reference is the
    finest computed cell, an uncoupled one only when alone. A dry run prints the plan and writes nothing; a study
    needs an active observable.
    '''
    cfg = toy_cfg(tmp_path)
    assert resolution.grid_values(cfg, 'quick') == ([10000, 30000, 100000, 300000], [1000, 100, 10])
    assert resolution.grid_values(cfg) == ([10000, 30000, 100000, 300000, 1000000], [1000, 300, 100, 30, 10, 3, 1])
    assert resolution.grid_values(cfg, {'n_steps': [5e3, 1e3, 5e3]}) == ([1000, 5000], [1000, 300, 100, 30, 10, 3, 1])
    with pytest.raises(config.ConfigError, match = 'allow_large_n_steps'):
        resolution.grid_values(cfg, {'n_steps': [2e6]})
    with pytest.raises(config.ConfigError):
        resolution.grid_values(cfg, 'fine')
    uncoupled = toy_cfg(tmp_path, 'model.n_atmodeller=0')
    assert resolution.grid_values(uncoupled, 'quick')[1] == [0]

    cells = resolution.plan_cells(cfg, [1000, 1000000], [3000, 100, 1])
    skip = {(c['n_steps'], c['n_atmodeller']): c['skip'] for c in cells}
    assert skip[1000, 3000] == 'n_atmodeller > n_steps' and 'max_eval_seconds' not in (skip[1000000, 1] or '')
    # the estimate is twice the cost model: 120 us per Atmodeller call plus 0.2 us per step
    assert math.isclose(cells[1]['est_s'], 2*(1.2e-4*11 + 2e-4))
    assert resolution.est_eval_seconds(1e6, 1) < 300 < resolution.est_eval_seconds(1e6, 1)*2 # 240 s < the default
    cells = resolution.plan_cells(toy_cfg(tmp_path, 'resolution.max_eval_seconds=10'), [1000, 1000000], [100, 1, 0])
    assert 'max_eval_seconds' in cells[4]['skip'] # (1e6, 1)
    ref = cells[resolution.reference_index(cells)]
    assert (ref['n_steps'], ref['n_atmodeller']) == (1000000, 100) # n_atmodeller = 0 only when alone
    # b resolves at least as finely as a; an uncoupled cell compares with uncoupled ones only
    a, b, zero = {'n_steps': 1000, 'n_atmodeller': 100}, {'n_steps': 5000, 'n_atmodeller': 10}, \
        {'n_steps': 5000, 'n_atmodeller': 0}
    assert resolution.at_least_as_fine(a, b) and not resolution.at_least_as_fine(b, a)
    assert not resolution.at_least_as_fine(a, zero)

    assert resolution.run_resolution(cfg, grid = 'quick', points = 3, workers = 2, dry_run = True) is None
    plan = capsys.readouterr().out
    assert '12 cells x 3 test points x 2 runs + 3 = 75 evaluations' in plan and 'estimated time' in plan
    assert not os.path.exists(os.path.join(str(tmp_path), 'lhs1140b'))
    with pytest.raises(config.ConfigError, match = 'none is active'):
        resolution.run_resolution(toy_cfg(tmp_path, 'observables.Rp.use=false'), dry_run = True)


def test_explicit_thetas_rough_and_no_overwrite(tmp_path, capsys):
    '''
    Checks a study at given test points and the warning of a rough likelihood. The noise floor removes the smooth
    (linear) change of a perturbation, measured by the derivative runs: what is left at a perturbation of 20% of the
    log_f_atm width (0.74 dex) is the toy's curvature, several sigma of Rp, which the table reports as roughness. A
    study never overwrites its output folder, and a test point outside the prior is an error.
    '''
    cfg = toy_cfg(tmp_path, 'resolution.perturb=0.2')
    thetas = [[-2.8, 5.6, 5e9], [-2.0, 5.0, 8e9]]
    out = resolution.run_resolution(cfg, grid = TINY, thetas = thetas, workers = 1, out_dir = str(tmp_path/'res'))
    table = open(os.path.join(out, 'table.txt')).read()
    assert '2 given test points' in table and 'ROUGH' in table
    assert 'likelihood is rough at the 0.2 level; MCMC will see numerical noise' in table
    assert 'convergence is ill-defined at point(s) 0, 1' in table
    with np.load(os.path.join(out, 'points.npz')) as p:
        assert np.array_equal(p['theta'], np.array(thetas))
    with pytest.raises(FileExistsError):
        resolution.run_resolution(cfg, grid = TINY, thetas = thetas, workers = 1, out_dir = out)
    with pytest.raises(config.ConfigError, match = 'outside the prior'):
        resolution.run_resolution(cfg, grid = TINY, thetas = [[-0.5, 5.6, 5e9]], dry_run = True)


def test_chain_source(tmp_path):
    '''
    Checks test points from an MCMC run (source = "chain:RUN_DIR"): they are posterior samples of the chain after
    its burn-in (summary.json's discard), and the table says so.
    '''
    cfg = toy_cfg(tmp_path)
    run_dir = tmp_path/'lhs1140b'/'mcmc_20260101-000000'
    os.makedirs(run_dir)
    # a fake chain of 40 steps x 4 walkers around the centre, with the burn-in in its summary
    rng = np.random.default_rng(3)
    chain = np.array([-2.8, 5.6, 5e9]) + np.array([0.1, 0.1, 2e8])*rng.standard_normal((40, 4, 3))
    log_prob = np.zeros((40, 4))
    log_prob[25, 0] = -np.inf # inside the burn-in, which is discarded anyway
    np.savez_compressed(run_dir/'chain.npz', chain = chain, log_prob = log_prob, accepted = np.ones(4))
    config.save_resolved(cfg, str(run_dir/'config.json'))
    with open(run_dir/'summary.json', 'w') as f:
        json.dump({'discard': 30, 'thin': 1}, f)
    out = resolution.run_resolution(cfg, grid = TINY, points = 3, source = f'chain:{run_dir}', workers = 1)
    with np.load(os.path.join(out, 'points.npz')) as p:
        theta = p['theta']
    later = chain[30:].reshape(-1, 3)
    assert theta.shape == (3, 3) and all(any(np.array_equal(t, s) for s in later) for t in theta)
    assert 'random posterior samples of' in open(os.path.join(out, 'table.txt')).read()
    assert glob.glob(os.path.join(out, '*.png'))


def test_bisect_source_and_equivalent_shift(tmp_path, capsys):
    '''
    Checks the bisection source and the equivalent shift. With init = "bisect" the source "auto" is the bisection,
    so the test points sit where Rp fits its datum; each is also run at log_f_atm + h (h = 1e-4 of the prior
    width) for dRp/dlog_f_atm, which turns the error of a cell into a shift of log_f_atm. On hand-made blobs a cell
    0.5 sigma off fails at fixed theta, but it is a negligible shift where Rp is steep and a huge one where it is
    flat.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.init="bisect"', 'sampler.bisect_observable="Rp"')
    assert config.resolution_source(cfg) == 'bisect'
    out = resolution.run_resolution(cfg, grid = TINY, points = 3, workers = 2)
    printed = capsys.readouterr().out
    assert 'test points: bisect' in printed and 'bisection of log_f_atm on Rp (batch 1)' in printed
    rec = json.load(open(os.path.join(out, 'recommendation.json')))
    assert rec['source'] == 'bisect' and rec['equivalent_shift']['param'] == 'log_f_atm'
    assert rec['equivalent_shift']['shift_tol'] == 1e-3 and rec['equivalent_shift']['prior_width'] == 3.7
    table = open(os.path.join(out, 'table.txt')).read()
    assert 'where the bisection on log_f_atm brought Rp to its data' in table
    assert 'Equivalent shift of log_f_atm' in table and 'SHIFT-BASED RECOMMENDATION' in table
    with np.load(os.path.join(out, 'points.npz')) as p:
        theta = p['theta']
        assert theta.shape == (3, 3) and p['bisect_point'].shape[1] == 3 and p['bisect_converged'] >= 3
        assert all(any(np.array_equal(t, q) for q in p['bisect_point']) for t in theta)
        step = p['theta_shifted'] - theta # h along log_f_atm alone
        assert np.all(step[:, 1:] == 0) and np.allclose(np.abs(step[:, 0]), 1e-4*3.7)
    at_config = model.evaluate_many(model.LocalPool(cfg), theta)[1]
    assert np.all(np.abs(at_config['Rp'] - 1.73) <= 0.1*0.025) # bisect_tol sigma at the config resolution
    with np.load(os.path.join(out, 'results.npz')) as r:
        ref = int(r['reference'])
        assert r['deriv'].shape == (3, 1) and r['shift'].shape == (3, 4, 1) and str(r['shift_param']) == 'log_f_atm'
        # deriv = dRp/dlog_f_atm at the reference cell; shift = (T(cell) - T(reference))/deriv
        T = r['blobs']['Rp'][:, :, 0]
        assert np.allclose(r['deriv'][:, 0], (r['shift_blobs']['Rp'] - T[:, ref])/r['shift_h'])
        assert np.allclose(r['shift'][:, :, 0], (T - T[:, [ref]])/r['deriv'][:, [0]])
        assert np.allclose(r['shift_frac'], np.abs(r['shift'])/3.7)

    # hand-made: every coarse cell is 0.5 sigma off at fixed theta; whether that is a negligible shift of
    # log_f_atm depends on how steep Rp is there
    cells = resolution.plan_cells(cfg, [1000, 10000], [100, 10])
    ref = resolution.reference_index(cells)
    Rp = [[1.7425, 1.7425, 1.7425, 1.730]]*2
    blobs = _blobs(cfg, cells, Rp, runtime = np.array([0.1, 0.5, 0.3, 2.0]))
    h = np.full(2, 1e-4*3.7) # [dex]
    for dRp, steep in ((1.0, True), (1e-6, False)): # Rp(theta + h) - Rp(theta) at the reference [R_earth]
        shift_blobs = _blobs(cfg, cells, [[1.730 + dRp]]*2)[:, 0, 0]
        m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5, shift_blobs, h, 0, 1e-3)
        assert m['verdict'] == ['fail', 'fail', 'fail', 'reference'] and m['recommended'] == ref # conservative
        assert np.all(m['steep'] == steep)
        assert m['worst_shift'][0] == pytest.approx(0.0125/(dRp/h[0]), rel = 1e-6) # [dex]
        if steep: # 0.5 sigma = 4.6e-6 dex of log_f_atm, 1.3e-6 of the prior width: the cheapest cell
            assert m['verdict_shift'] == ['stable', 'stable', 'stable', 'reference'] and m['recommended_shift'] == 0
        else: # 0.5 sigma = 4.6 dex: no cell but the reference
            assert m['verdict_shift'] == ['fail', 'fail', 'fail', 'reference'] and m['recommended_shift'] == ref


def test_bisect_source_takes_the_better_half(tmp_path, monkeypatch):
    '''
    Checks that the bisection source takes its test points from the better half (by lp) of the converged draws,
    the best one first. A converged draw can still lie far off the posterior (on LHS 1140 b one matched the outflow
    H fraction at the stripping edge, with lp -1859), and farthest-point sampling alone would pick such draws from
    the corners of the cube. With fewer good draws than test points, the worse ones fill up.
    '''
    # 10 converged draws, the worse half far off (lp -1000 and below) and spread to the corners of the cube, where
    # farthest-point sampling would pick them; 2 draws not converged
    from isofate.mcmc import search
    cfg = toy_cfg(tmp_path, 'sampler.init="bisect"', 'sampler.bisect_observable="Rp"')
    rng = np.random.default_rng(3)
    lo = np.array([cfg['params'][p]['bounds'][0] for p in cfg['free']])
    hi = np.array([cfg['params'][p]['bounds'][1] for p in cfg['free']])
    good_u = 0.45 + 0.1*rng.random((5, len(lo)))     # good points: the middle of the cube
    bad_u = np.array([[0.01]*len(lo), [0.99]*len(lo), [0.01, 0.99, 0.01], [0.99, 0.01, 0.99], [0.01, 0.01, 0.99],
                      [0.5]*len(lo), [0.5]*len(lo)])
    point = lo + (hi - lo)*np.vstack([good_u, bad_u])
    lp = np.array([-20.0, -21, -22, -23, -24, -1000, -1500, -1859, -2000, -3000, -np.inf, -np.inf])
    outcome = np.array([search.CONVERGED]*10 + [search.UNCONVERGED]*2)
    blobs = np.empty(12, dtype = model.blob_dtype())
    blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
    blobs['status'] = 0
    # a stand-in result of search.bisect_points with these draws
    fake = {'point': point, 'theta': point, 'x': point[:, 0], 'g': np.zeros(12), 'outcome': outcome, 'lp': lp,
            'blobs': blobs, 'evaluations': 100, 'settings': {'param': 'log_f_atm', 'observable': 'Rp'}}
    monkeypatch.setattr(search, 'bisect_points', lambda *args, **kwargs: fake)
    warnings = []
    thetas, info = resolution._bisect_points(cfg, None, 4, rng, warnings)
    chosen = [int(np.flatnonzero((point == t).all(axis = 1))[0]) for t in thetas] # rows of point
    assert chosen[0] == 0 and sorted(chosen) == sorted(set(chosen)) and max(chosen) <= 4, chosen
    assert info['bisect_converged'] == 10 and not warnings
    # with fewer good points than test points, it still takes n (the worse ones fill up)
    thetas, _ = resolution._bisect_points(cfg, None, 7, rng, warnings)
    assert len(thetas) == 7


def test_parent_imports_stay_light():
    '''
    Checks that importing resolution and plots loads neither jax, atmodeller, numba nor the coupled model: the
    study runs its evaluations in spawned workers.
    '''
    code = ('import sys, isofate.mcmc.resolution, isofate.mcmc.plots; '
            "heavy = [m for m in ('jax', 'atmodeller', 'numba', 'isofate.isofate_coupler') if m in sys.modules]; "
            'assert not heavy, heavy')
    subprocess.run([sys.executable, '-c', code], check = True)


###_____Noise floor, equivalent shift and table details_____###

def test_noise_floor_removes_the_slope(tmp_path):
    '''
    Checks that the noise floor subtracts the smooth change over its step. For a steep but smooth observable the
    perturbed runs change Rp by 5 sigma, all of it the slope (100 R_earth/dex x 1.25e-3 dex): with the slope from
    the derivative runs the floor is 0, and real noise on top of it (0.2 sigma at one cell) is what is left.
    Without the step given, the raw 5 sigma remain.
    '''
    cfg = toy_cfg(tmp_path)
    cells = resolution.plan_cells(cfg, [1000, 10000], [100, 10])
    ref = resolution.reference_index(cells)
    slope, step, h = 100.0, 1.25e-3, 3.7e-4 # [R_earth/dex], [dex], [dex]
    Rp = np.full((1, 4), 1.73)
    perturbed = Rp + slope*step
    perturbed[0, 1] += 0.2*0.025 # noise of 0.2 sigma at cell 1
    blobs = _blobs(cfg, cells, Rp, perturbed)
    shift_blobs = _blobs(cfg, cells, [[1.73 + slope*h]*4])[:, 0, 0]
    m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5, shift_blobs, np.array([h]), 0, 1e-3,
                           noise_step = np.array([step]))
    assert m['noise'][0, :, 0] == pytest.approx([0, 0.2, 0, 0], abs = 1e-6)
    raw = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5, shift_blobs, np.array([h]), 0, 1e-3) # no step given
    assert raw['noise'][0, 0, 0] == pytest.approx(5.0)


def two_observables(tmp_path):
    '''
    A config fitting Rp (linear) and Mdot_HHe (log10), its 2 x 2 grid of cells and the reference: (cfg, cells, ref)
    '''
    cfg = toy_cfg(tmp_path, 'observables.Mdot_HHe.use=true') # Rp (linear) and Mdot_HHe (log10, 0.137/0.121 dex)
    cells = resolution.plan_cells(cfg, [1000, 10000], [100, 10])
    return cfg, cells, resolution.reference_index(cells)


def blobs_rp_mdot(cfg, Rp, Mdot):
    '''
    Hand-made blobs of a study with Rp and Mdot_HHe and their log-likelihood (status ok, Mp = 5.6, runtime 1 s)

    Inputs:
        - cfg: the config, whose observables give the log-likelihood [dict]
        - Rp: the radius at each (point, cell) [R_earth]
        - Mdot: Mdot_HHe at each (point, cell) [kg/s]

    Output: blobs (points, cells, 1) of model.blob_dtype(), without perturbed runs [array]
    '''
    Rp, Mdot = np.asarray(Rp, dtype = float), np.asarray(Mdot, dtype = float)
    blobs = np.empty(Rp.shape + (1,), dtype = model.blob_dtype())
    blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
    blobs['status'], blobs['runtime'], blobs['Mp'] = model.OK, 1.0, 5.6
    blobs['Rp'][..., 0], blobs['Mdot_HHe'][..., 0] = Rp, Mdot
    blobs['log_like'][..., 0] = np.vectorize(lambda r, x: likelihood.log_like({'Rp': r, 'Mdot_HHe': x, 'Mp': 5.6},
                                                                              cfg)[0])(Rp, Mdot)
    return blobs


def test_worst_shift_counts_only_the_observables_that_need_it(tmp_path):
    '''
    Checks that the worst shift of a cell counts only the observables that fail at fixed theta. At cell 0 Rp (steep,
    100 R_earth/dex) fails at fixed theta by a shift of 1e-3 dex. Mdot_HHe (not steep, 0.1 dex/dex) passes at fixed
    theta (d = 0.08), although its 0.01 dex error would be a 'shift' of 0.1 dex. So the worst shift is Rp's.
    '''
    cfg, cells, ref = two_observables(tmp_path)
    v = 2.03e5 # the Mdot_HHe datum [kg/s]
    blobs = blobs_rp_mdot(cfg, [[1.73 + 0.1, 1.73, 1.73, 1.73]]*2, [[v*10**0.01, v, v, v]]*2)
    shift_blobs = blobs_rp_mdot(cfg, [[1.73 + 100*3.7e-4]]*2, [[v*10**(0.1*3.7e-4)]]*2)[:, 0, 0]
    m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5, shift_blobs, np.full(2, 3.7e-4), 0, 1e-3)
    names = m['obs_names']
    k = names.index('Mdot_HHe')
    assert m['d'][0, 0, k] < 0.1 and m['shift'][0, 0, k] == pytest.approx(0.1)
    assert m['worst_shift'][0] == pytest.approx(1e-3) and names[m['worst_shift_obs'][0]] == 'Rp'
    assert m['verdict_shift'][0] == 'stable' and m['ridge_spread'][0] < 0.01 and names[m['ridge_obs']] == 'Rp'


def test_shift_criterion_keeps_the_ridge_weights(tmp_path):
    '''
    Checks that the shift criterion also keeps the weights along the ridge. Mdot_HHe is steep too (5 dex/dex) and 3
    sigma off its datum. At cell 0 the Rp error is a shift of +3.5e-3 dex at point 0 and -3.5e-3 at point 1, each
    below shift_tol x the width (3.7e-3), and Mdot_HHe is unchanged at fixed theta. Moved back along log_f_atm to
    where Rp matches, Mdot_HHe changes by -/+0.0175 dex and its term by -/+0.4: a spread of 0.8 > dlogl_tol, so the
    cell fails although each observable passes on its own.
    '''
    cfg, cells, ref = two_observables(tmp_path)
    off = 2.03e5*10**(3*0.137) # 3 sigma above
    rp = [[1.73 + 0.35, 1.73, 1.73, 1.73], [1.73 - 0.35, 1.73, 1.73, 1.73]]
    blobs = blobs_rp_mdot(cfg, rp, [[off]*4]*2)
    shift_blobs = blobs_rp_mdot(cfg, [[1.73 + 100*3.7e-4]]*2, [[off*10**(5*3.7e-4)]]*2)[:, 0, 0]
    m = resolution.analyse(cfg, cells, blobs, ref, 0.1, 0.5, shift_blobs, np.full(2, 3.7e-4), 0, 1e-3)
    assert np.all(m['shift_frac'][:, 0, 0] <= 1e-3) and np.all(m['steep'])
    assert m['ridge_spread'][0] > 0.5 and m['verdict_shift'][0] == 'fail'
    assert m['ridge_dlogl'][0, 0]*m['ridge_dlogl'][1, 0] < 0 # opposite changes at the two points


def test_table_prints_each_line_of_a_warning(tmp_path):
    '''
    Checks that a warning of several lines keeps its line breaks in table.txt. With an Rp datum of 0.5, below the
    bare core, the bisection finds no test point, and its failure counts are bullet lines, each on its own line.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.init="bisect"', 'sampler.bisect_observable="Rp"', 'observables.Rp.value=0.5')
    out = resolution.run_resolution(cfg, grid = TINY, points = 2, workers = 1)
    table = open(os.path.join(out, 'table.txt')).read()
    lines = table.splitlines()
    assert any(line.startswith('      - no sign change: g > 0 at both ends in') for line in lines), table
    assert any(line.startswith('      Try another sampler.bisect_observable') for line in lines)
