'''
Tests of the bisection initializer (isofate.mcmc.search; sampler.init = "bisect", check --bisect), the safe jitter
of the starting points, grid scans (isofate.mcmc.scan, the scan command) and run comparisons (the compare
command), on the toy backend: in one process, or with pools of at most 2 spawned workers.
The bisection moves each prior draw along one parameter (default log_f_atm) until one observable (default Rp)
matches its datum, to start the walkers on a thin surface that no prior draw hits by chance.
'''

from isofate.mcmc import env

env.set_single_thread_env() # before anything imports jax

import json
import os
import tomllib

import numpy as np
import pytest

from isofate.mcmc import config, model, plots, sampler, scan, search
from isofate.mcmc import params as prm
from isofate.mcmc.__main__ import main

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
# the example on the toy at 2e4 steps, on 2 workers
TOY = ['model.backend="toy"', 'parallel.cache=false', 'parallel.warmup=false', 'model.n_steps=20000',
       'parallel.workers=2', 'solve.param=""', 'sampler.init="bisect"',
       'params.albedo.vary=false'] # the bisection, without solve mode;
# albedo fixed at 0 (the example samples it)
# the radius and mass fit: eps fixed, the outflow observables off
RP_ONLY = ['params.eps.vary=false', 'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false']


def toy_cfg(tmp_path, *overrides):
    '''
    The toy config of TOY, writing to tmp_path, with more --set overrides
    '''
    return config.load_config(EXAMPLE, TOY + [f'output_root="{tmp_path}"', *overrides])


def quiet(*args, **kwargs):
    '''
    A log function that prints nothing
    '''
    pass


###_____Bisection_____###

def test_bisect_on_rp(tmp_path):
    '''
    Checks the bisection of log_f_atm on Rp: at least 8 draws converge to within bisect_tol (0.1 sigma) of the
    datum, in 1-3 batches with a log line each; only log_f_atm moves, the root lies inside its final bracket, and
    lp and blobs are those of the model at the point. Also the outcome counts, the number of evaluations, the saved
    file, and the choice of the walkers' starts: 8 distinct draws of the best 16.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.bisect_observable="Rp"') # the default observable of any planet
    pool = model.LocalPool(cfg)
    lines = []
    result = search.bisect_points(cfg, pool, 8, np.random.default_rng(1), lines.append)
    ok = search.good(result)
    assert ok.size >= 8 and 1 <= result['batches'] <= 3 and len(lines) == result['batches']
    assert 'bisection of log_f_atm on Rp' in lines[-1] and 'converged' in lines[-1]
    st = result['settings']
    assert (st['param'], st['observable'], st['lo'], st['hi'], st['iters']) == ('log_f_atm', 'Rp', -5.0, -1.3, 22)
    assert np.all(np.abs(result['blobs']['Rp'][ok] - 1.73) <= 0.1*0.025) # within bisect_tol sigma
    j = cfg['free'].index('log_f_atm') # only the bisection parameter moved
    assert np.array_equal(np.delete(result['point'][ok], j, 1), np.delete(result['theta'][ok], j, 1))
    assert np.array_equal(result['point'][ok, j], result['x'][ok])
    assert np.all((result['a'][ok] <= result['x'][ok]) & (result['x'][ok] <= result['b'][ok]))
    for i in ok[:3]: # lp and blobs are those of model.log_prob at the point
        res = model.evaluate(result['point'][i], cfg)
        assert res['lp'] == result['lp'][i] and res['obs']['Rp'] == result['blobs']['Rp'][i]
    # every search starts from both prior bounds
    assert np.all(result['hist_x'][:, 0] == -5.0) and np.all(result['hist_x'][:, 1] == -1.3)
    counts = search.counts(result)
    assert sum(counts[name] for name in search.OUTCOMES.values()) == len(result['outcome'])
    assert counts['converged'] == ok.size and counts['bracketed'] >= ok.size
    assert result['evaluations'] == np.isfinite(result['hist_x']).sum()

    path = search.save(result, str(tmp_path/'bisect.npz'))
    with np.load(path) as d:
        assert str(d['param']) == 'log_f_atm' and str(d['observable']) == 'Rp'
        assert d['point'].shape == result['point'].shape and d['blobs'].dtype == np.dtype(model.blob_dtype())
    pick = search.choose(result, 8, np.random.default_rng(2)) # 8 of the best 16, distinct
    assert len(set(pick.tolist())) == 8 and set(pick) <= set(ok)
    best16 = ok[np.argsort(-result['lp'][ok])][:16]
    assert set(pick) <= set(best16)


def test_bisect_another_observable_and_parameter(tmp_path):
    '''
    Checks the bisection of another observable along another parameter: the final atmospheric mass fraction
    (1e-4 +- 0.3 dex, in log10 space) along the age. The points converge to within 0.1 sigma (0.03 dex) and stay
    inside the age prior, with the age alone moved.
    '''
    cfg = toy_cfg(tmp_path, *RP_ONLY, 'observables.fatm={use = true, value = 1e-4, sigma = 0.3, space = "log10"}',
                  'sampler.bisect_observable="fatm"', 'sampler.bisect_param="age"')
    result = search.bisect_points(cfg, model.LocalPool(cfg), 6, np.random.default_rng(4), quiet)
    ok = search.good(result)
    assert ok.size >= 6
    assert np.all(np.abs(np.log10(result['blobs']['fatm'][ok]) + 4) <= 0.03)
    j = cfg['free'].index('age')
    lo, hi = cfg['params']['age']['bounds']
    assert np.all((result['point'][ok, j] >= lo) & (result['point'][ok, j] <= hi))
    assert np.array_equal(np.delete(result['point'][ok], j, 1), np.delete(result['theta'][ok], j, 1))


def test_bisect_check_single_draw(tmp_path):
    '''
    Checks the bisection of single draws, as check --bisect runs it, with its iteration table, and the rules for
    an undefined or infinite g. A NaN end takes the sign opposite to the finite end, so a bracket can close on the
    edge of the stripped region without a root; an infinite g (log10 of 0 with floor 0) is defined and has a sign.
    '''
    cfg = toy_cfg(tmp_path, *RP_ONLY, 'sampler.bisect_observable="Rp"')
    theta = [-2.5, 5.6, 5e9]
    result = search.run_bisection(cfg, [theta], model.LocalPool(cfg))
    result['settings'] = search.settings(cfg)
    assert result['outcome'][0] == search.CONVERGED and 3 <= result['rounds'][0] <= 22
    lines = search.iteration_table(result, 0, cfg)
    assert lines[1].split()[:2] == ['lo', '-5'] and lines[2].split()[:2] == ['hi', '-1.3']
    assert len(lines) == 2 + 2 + result['rounds'][0] # header, the two ends, the rounds, the footnote
    # a NaN end takes the sign opposite to the finite end: the outflow H fraction is undefined when stripped. At
    # 10 Gyr the toy is stripped at log_f_atm = -5 and has a pure H wind (x = 1, g = +2.94) wherever it is not, so
    # the bracket closes on the edge of the stripped region without converging
    out = toy_cfg(tmp_path, 'params.eps.vary=false')
    st = search.settings(out)
    res = search.run_bisection(out, [[-3.0, 5.6, 1e10]], model.LocalPool(out), st)
    assert np.isnan(res['hist_g'][0, 0]) and res['hist_g'][0, 1] == pytest.approx(np.log10(1/0.00115))
    assert res['outcome'][0] == search.UNCONVERGED and res['rounds'][0] == 22 # bracketed, never within tol
    a, b = res['a'][0], res['b'][0]
    assert 0 < b - a < 3.7/2**21
    ends = model.evaluate_many(model.LocalPool(out), [[a, 5.6, 1e10], [b, 5.6, 1e10]])[1]
    assert ends['exhausted'].tolist() == [1.0, 0.0] # a stripped (the NaN side), b not
    # at 5 Gyr nothing is stripped: g > 0 at both ends, no sign change
    res = search.run_bisection(out, [[-3.0, 5.6, 5e9]], model.LocalPool(out), st)
    assert res['outcome'][0] == search.ABOVE and res['rounds'][0] == 0
    # an infinite g is defined: with floor = 0 the log10 He fraction of an H-only wind is -inf (g < 0), so the
    # stripped (NaN) end takes the opposite sign and the draw is bracketed
    he = toy_cfg(tmp_path, 'params.eps.vary=false', 'observables.outflow_H_fraction.use=false',
                 'observables.outflow_He_fraction={use = true, value = 0.5, sigma = 0.3, space = "log10", floor = 0.0}',
                 'sampler.bisect_observable="outflow_He_fraction"')
    res = search.run_bisection(he, [[-3.0, 5.6, 1e10]], model.LocalPool(he))
    assert np.isnan(res['hist_g'][0, 0]) and res['hist_g'][0, 1] == -np.inf
    assert res['outcome'][0] != search.UNDEFINED and res['rounds'][0] > 0


@pytest.mark.parametrize('value, sign, why', [(0.5, '>', 'bare-core radius'), (5.0, '<', 'largest log_f_atm')])
def test_bisect_no_bracket_message(tmp_path, value, sign, why):
    '''
    Checks the message when no draw has a sign change, with the fallback off: an Rp datum below the bare-core
    radius (g > 0 at both ends) or above what the largest log_f_atm gives (g < 0). After 3 batches of 10 draws the
    text names the case, its likely cause and what to try; init_bisect raises it, after saving bisect.npz.
    '''
    cfg = toy_cfg(tmp_path, *RP_ONLY, f'observables.Rp.value={value}', 'sampler.bisect_observable="Rp"',
                  'sampler.bisect_fallback=false')
    pool = model.LocalPool(cfg)
    result = search.bisect_points(cfg, pool, 8, np.random.default_rng(1), quiet, draws = 10)
    assert search.good(result).size == 0 and result['batches'] == 3 and len(result['outcome']) == 30
    text = search.failure_text(result, 8)
    assert f'no sign change: g {sign} 0 at both ends in 30 draws' in text and why in text
    assert '--bisect-observable' in text and '--bisect-param' in text and 'check CONFIG --bisect' in text
    with pytest.raises(RuntimeError, match = f'g {sign} 0 at both ends'):
        sampler.init_bisect(cfg, pool, 8, np.random.default_rng(1), str(tmp_path), quiet)
    assert os.path.exists(tmp_path/'bisect.npz') # saved before the failure


def test_bisect_fallback(tmp_path):
    '''
    Checks the fallback of a failed bisection (sampler.bisect_fallback, on by default): init_bisect warns once,
    with the reason, and starts the walkers from a prescan instead.
    '''
    cfg = toy_cfg(tmp_path, *RP_ONLY, 'observables.Rp.value=0.5', 'sampler.bisect_observable="Rp"') # below the core
    assert cfg['sampler']['bisect_fallback'] # the default
    warnings = []
    coords, lp, blobs = sampler.init_bisect(cfg, model.LocalPool(cfg), 8, np.random.default_rng(1), str(tmp_path),
                                            quiet, warnings.append)
    assert len(warnings) == 1 and 'falling back to init = "prescan"' in warnings[0]
    assert 'g > 0 at both ends' in warnings[0]
    assert coords.shape == (8, 3) and np.isfinite(lp).all() and os.path.exists(tmp_path/'prescan.npz')


def test_run_with_bisect_init(tmp_path):
    '''
    Checks a short run with init = "bisect" on Rp: the log shows the bisection, bisect.npz holds at least 16
    converged draws, and the 16 walkers (4 free parameters, eps included) start from them. The evaluations of the
    initialization count the preflight, both ends of each draw and the rounds.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.bisect_observable="Rp"', 'sampler.nwalkers=16', 'sampler.max_steps=10',
                  'sampler.tracks=0', 'sampler.check_every=10', 'sampler.reset_stuck_at=0')
    run_dir = sampler.run_mcmc(cfg, workers = 2, make_plots = False)
    log = open(os.path.join(run_dir, 'run.log')).read()
    assert 'bisection of log_f_atm on Rp (batch 1)' in log and 'converged bisection points' in log
    with np.load(os.path.join(run_dir, 'bisect.npz')) as d:
        assert (d['outcome'] == search.CONVERGED).sum() >= 16
    run = sampler.load_run(run_dir)
    assert run['summary']['settings']['init'] == 'bisect' and run['chain'].shape == (10, 16, 4)
    # 48 draws per batch (3 x nwalkers)
    assert sum(run['summary']['evaluations']['init'].values()) > 2*48 # preflight, both ends, rounds


###_____Safe jitter_____###

def test_safe_jitter_keeps_walkers_on_a_ridge(tmp_path):
    '''
    Checks that safe_jitter keeps starting points on a thin ridge. With Rp known to 1e-4 R_earth the ridge in
    log_f_atm is so thin that a plain jitter of 1e-3 of the prior box costs about 20 in lp. safe_jitter halves the
    jitter until no point loses more than JITTER_DLP = 5, keeps the points distinct and re-evaluates them. With no
    halving left, distinct points keep their unjittered position, and copies take their best try.
    '''
    # Rp to 1e-4 R_earth: a thin ridge in log_f_atm, where a jitter of 1e-3 of the prior box costs ~20 in lp
    cfg = toy_cfg(tmp_path, *RP_ONLY, 'observables.Rp.sigma=1e-4', 'sampler.bisect_observable="Rp"')
    pool = model.LocalPool(cfg)
    result = search.bisect_points(cfg, pool, 6, np.random.default_rng(1), quiet)
    ok = search.good(result)[:6]
    coords, lp, blobs = result['point'][ok], result['lp'][ok], result['blobs'][ok]
    plain = prm.jitter(coords, cfg, np.random.default_rng(3))
    assert np.any(model.evaluate_many(pool, plain)[0] < lp - 5)
    new, lp_new, blobs_new, info = sampler.safe_jitter(cfg, pool, coords, lp, blobs, np.random.default_rng(3))
    assert info['halved'] > 0 and np.all(lp_new >= lp - sampler.JITTER_DLP)
    assert np.all(np.any(new != coords, axis = 1) | np.all(new == coords, axis = 1))
    assert len(np.unique(new, axis = 0)) == len(new)
    assert np.array_equal(lp_new, model.evaluate_many(pool, new)[0]) and blobs_new.dtype == blobs.dtype
    # with no halving left (a tiny tolerance in lp), distinct points stay unjittered; copies keep their best try
    cfg['observables']['Rp']['sigma'] = 1e-9
    lp_tight, blobs_tight = model.evaluate_many(pool, coords)
    new, lp_new, _, info = sampler.safe_jitter(cfg, pool, coords, lp_tight, blobs_tight, np.random.default_rng(3))
    assert info['unjittered'] + int(np.sum(lp_new >= lp_tight - 5)) >= len(coords) - info['unjittered']
    copies, lp_c, _, info_c = sampler.safe_jitter(cfg, pool, coords, lp_tight, blobs_tight, np.random.default_rng(3),
                                                  others = coords)
    assert info_c['unjittered'] == 0 and np.all(np.isfinite(lp_c))


###_____Scans_____###

def toy_file(tmp_path, name = 'toy', **sampler_options):
    '''
    Writes a toy-backend copy of the LHS 1140 b example to tmp_path/<name>.toml, for the commands: all observables
    as in the example, init = "bisect" on Rp without solve mode, 16 walkers, 10 steps and 4 tracks

    Inputs:
        - tmp_path: folder of the file, and its output_root [Path]
        - name: the config's name [str]
        - sampler_options: more [sampler] options [dict]

    Output: path of the config [str]
    '''
    with open(EXAMPLE, 'rb') as f:
        raw = tomllib.load(f)
    raw['name'] = name
    raw['output_root'] = str(tmp_path)
    raw['model'].update(backend = 'toy', n_steps = 20000)
    raw['parallel'] = {'workers': 2, 'cache': False, 'warmup': False}
    raw['params']['albedo'] = {'value': 0.0} # fixed (the example samples it)
    raw['solve'] = {'param': ''} # the bisection initializer, without solve mode
    raw['sampler'].update(max_steps = 10, checkpoint_every = 10, print_every = 10, check_every = 10, tracks = 4,
                          track_points = 40, nwalkers = 16, reset_stuck_at = 0, bisect_observable = 'Rp',
                          init = 'bisect', **sampler_options)
    path = tmp_path/f'{name}.toml'
    path.write_text(config.dump_toml(raw))
    return str(path)


def test_scan_axes():
    '''
    Checks the axes of a scan: name=lo:hi:n gives n values from lo to hi, both included; values outside the prior, a
    fixed or misspelt parameter and a malformed axis are errors. The grid of two axes runs x fastest, with the other
    parameters at their values, and by default a scan draws the active observables that the model computes.
    '''
    cfg = config.load_config(EXAMPLE, TOY)
    name, values = scan.parse_axis('log_f_atm=-3.1:-2.9:81', cfg)
    assert name == 'log_f_atm' and len(values) == 81 and values[0] == -3.1 and values[-1] == -2.9
    with pytest.raises(config.ConfigError, match = 'outside the prior'):
        scan.parse_axis('log_f_atm=-6:-2:5', cfg)
    with pytest.raises(config.ConfigError, match = 'must be a free parameter'):
        scan.parse_axis('t_sat=1e9:3e9:3', cfg)
    with pytest.raises(config.ConfigError, match = 'did you mean'):
        scan.parse_axis('log_fatm=-3:-2:3', cfg)
    with pytest.raises(config.ConfigError, match = 'name=lo:hi:n'):
        scan.parse_axis('log_f_atm=-3:-2', cfg)
    thetas, shape = scan.grid_thetas(cfg, ('log_f_atm', np.array([-3.0, -2.0])), ('age', np.array([4e9, 5e9, 6e9])))
    assert shape == (3, 2) and thetas.shape == (6, 4) # (ny, nx); theta in the order of log_f_atm, Mp, age, eps
    assert thetas[:, 0].tolist() == [-3.0, -2.0]*3 and thetas[:, 2].tolist() == [4e9, 4e9, 5e9, 5e9, 6e9, 6e9]
    assert np.all(thetas[:, 1] == 5.6) and np.all(thetas[:, 3] == 0.15) # the others at their values
    assert scan.scan_observables(cfg) == ['Rp', 'Mdot_HHe', 'outflow_H_fraction']


def test_scan_1d_and_2d(tmp_path, capsys):
    '''
    Checks the scan command in 1-D (16 values of log_f_atm at 5 Gyr) and in 2-D (6 x 4 over log_f_atm and age,
    with chosen observables and --out): scan.npz, the printed best point and scan.png, which plot_dir draws again.
    An axis outside the prior and an unknown observable are errors.
    '''
    path = toy_file(tmp_path)
    assert main(['scan', path, '--x', 'log_f_atm=-3:-1.5:16', '--theta', 'age=5e9', '--workers', '1']) == 0
    out = capsys.readouterr().out
    assert 'best point: lp =' in out and 'log_f_atm' in out and 'scan.png' in out
    (one,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    data = scan.load_scan(one)
    assert data['x'].shape == (16,) and data['blobs'].shape == (16,) and tuple(data['shape']) == (16,)
    assert np.all(data['theta'][:, 2] == 5e9) and data['blobs'].dtype == np.dtype(model.blob_dtype())
    assert list(data['obs']) == ['Rp', 'Mdot_HHe', 'outflow_H_fraction']
    assert os.path.getsize(os.path.join(one, 'scan.png')) > 10000
    assert main(['scan', path, '--x', 'log_f_atm=-3:-1.5:6', '--y', 'age=3.5e9:1e10:4', '--obs',
                 'Rp,fatm,exhausted', '--workers', '2', '--out', str(tmp_path/'two')]) == 0
    data = scan.load_scan(str(tmp_path/'two'))
    assert tuple(data['shape']) == (4, 6) and data['y'].shape == (4,) and list(data['obs']) == ['Rp', 'fatm',
                                                                                                'exhausted']
    assert np.array_equal(data['theta'][:6, 0], data['x']) and np.all(data['theta'][:6, 2] == 3.5e9)
    assert os.path.getsize(tmp_path/'two'/'scan.png') > 10000
    assert plots.plot_dir(str(tmp_path/'two')) == [str(tmp_path/'two'/'scan.png')]
    assert main(['scan', path, '--x', 'log_f_atm=-9:-1.5:6']) == 2 # outside the prior
    assert main(['scan', path, '--x', 'log_f_atm=-3:-1.5:6', '--obs', 'Rpp']) == 2


def test_scan_floor_cells(tmp_path):
    '''
    Checks the cells that a 2-D scan leaves out of its colour scale: those at the log10 floor of an observable (e.g.
    Mdot_HHe = 0 of stripped planets), which would otherwise set the scale and flatten everything else. Nothing is
    singled out when every value is at the floor, nor for a linear observable or one that is not fitted.
    '''
    cfg = toy_cfg(tmp_path)
    mdot, rp = cfg['observables']['Mdot_HHe'], cfg['observables']['Rp']
    values = np.array([[-3.0, 4.5], [4.6, np.nan]]) # log10 of 0 (floor 1e-3 kg/s), two rates, undefined
    assert plots._at_floor(values, mdot).tolist() == [[True, False], [False, False]]
    assert not plots._at_floor(np.full((2, 2), -3.0), mdot).any() # all at the floor: nothing to single out
    assert not plots._at_floor(values, rp).any() and not plots._at_floor(values, None).any() # linear, unfitted


###_____check --bisect, run --bisect-*, compare_____###

def test_check_bisect_cli(tmp_path, capsys):
    '''
    Checks check --bisect: one bisection at the given point, printed round by round. The toy never reaches the
    outflow H fraction of 0.00115, so there is no root (exit code 1); an input such as Mp cannot be bisected (exit
    code 2).
    '''
    path = toy_file(tmp_path)
    assert main(['check', path, '--bisect', '--theta', 'log_f_atm=-2.5', '--theta', 'age=5e9']) == 0
    out = capsys.readouterr().out
    assert 'Bisection of log_f_atm in [-5, -1.3] on Rp' in out and 'converged: log_f_atm =' in out
    assert 'g = T(m) - target' in out
    # the outflow H fraction: the toy never reaches 0.00115, so no root (exit code 1)
    assert main(['check', path, '--bisect', '--bisect-observable', 'outflow_H_fraction']) == 1
    assert 'no root' in capsys.readouterr().out
    assert main(['check', path, '--bisect', '--bisect-param', 'age', '--bisect-observable', 'Mp']) == 2
    assert 'computed by the model' in capsys.readouterr().err


def test_compare_runs(tmp_path, capsys):
    '''
    Checks the compare command on two toy runs (dynamic_phi on and off): compare.txt and its three figures in a new
    folder compare_<stamp> next to the first run, and the text, also printed, with each run's settings, statistics
    and posterior medians. One run, a wrong number of labels and an existing output folder are errors.
    '''
    a = toy_file(tmp_path, 'toy_a')
    b = toy_file(tmp_path, 'toy_b', seed = 7) # another seed
    assert main(['run', a, '--no-plots']) == 0
    assert main(['run', b, '--no-plots', '--set', 'model.dynamic_phi=false']) == 0
    (run_a,) = [os.path.join(tmp_path, 'toy_a', d) for d in os.listdir(tmp_path/'toy_a')]
    (run_b,) = [os.path.join(tmp_path, 'toy_b', d) for d in os.listdir(tmp_path/'toy_b')]
    capsys.readouterr()
    assert main(['compare', run_a, run_b, '--labels', 'dynamic,static']) == 0
    out = capsys.readouterr().out
    (folder,) = [d for d in os.listdir(tmp_path/'toy_a') if d.startswith('compare_')]
    folder = os.path.join(tmp_path, 'toy_a', folder)
    for name in ('compare.txt', 'compare_corner.png', 'compare_predictive.png', 'compare_tracks.png'):
        assert os.path.getsize(os.path.join(folder, name)) > 500, name
    text = open(os.path.join(folder, 'compare.txt')).read()
    assert text in out and 'dynamic_phi = true' in text and 'dynamic_phi = false' in text
    for line in ('acceptance (mean)', 'tau max', 's/step', 'evaluations of dynamic', 'outflow_H_fraction', 'eps'):
        assert line in text, line
    run = sampler.load_run(run_a)
    median = np.median(sampler.posterior(run)[0][:, 0])
    assert f'{median:.5g}' in text # the log_f_atm median of the first run
    assert main(['compare', run_a]) == 2
    assert main(['compare', run_a, run_b, '--labels', 'one']) == 2
    with pytest.raises(FileExistsError):
        plots.compare_runs([run_a, run_b], out_dir = folder)
    assert json.load(open(os.path.join(run_b, 'summary.json')))['settings']['init'] == 'bisect'
