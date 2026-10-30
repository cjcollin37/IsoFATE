'''
Tests of isofate.mcmc.model: the isocalc inputs of a parameter set, the observables of the final state, the status
codes and blobs of an evaluation, and the posterior tracks; also the toy backend and the worker pool.
Most tests run on the toy backend. Tests marked slow run the real isocalc in this process, at n_steps = 2000.
'''

from isofate.mcmc import env

env.set_single_thread_env() # before anything imports jax

import math
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from isofate import constants as c
from isofate.orbit_params import EqTemp
from isofate.mcmc import config, likelihood, model, toy
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool, choose_workers, est_worker_gb, worker_rss_gb

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
CENTRE = [-2.8, 5.6, 5e9] # log_f_atm, Mp [M_earth], age [yr]
STRIPPED = [-6.0, 5.6, 5e9] # an atmosphere this small is lost before 5 Gyr
# the composition observables: NaN once the atmosphere is gone
COMPOSITION = ('H_He', 'C_O', 'D_H', 'mmw', 'T_surf') + tuple(f'VMR_{mol}' for mol in model.MOLECULES)


# the example reduced to the radius and mass fit of 3 free parameters (log_f_atm, Mp, age)
RP_ONLY = ['params.eps.vary=false', 'params.albedo.vary=false',
           'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false',
           'solve.param=""', 'sampler.init="prescan"', 'sampler.reset_stuck_at=0'] # without solve mode


def example(*overrides, backend = 'toy'):
    '''
    The LHS 1140 b example reduced to its radius and mass fit (RP_ONLY), with a wider log_f_atm prior, [-7, -1],
    that reaches stripped planets, and without compile caches or warm-up

    Inputs:
        - overrides: more --set overrides [str]
        - backend: 'toy' or 'isocalc' [str]

    Output: the resolved config [dict]
    '''
    return config.load_config(EXAMPLE, [f'model.backend={backend}', 'parallel.cache=false', 'parallel.warmup=false',
                                        'params.log_f_atm.bounds=[-7, -1]', *RP_ONLY, *overrides])


@pytest.fixture
def worker_globals():
    '''
    Restores the worker globals of model (_CFG, _RUN_DIR, _N_UNEXPECTED), which init_worker and the status-5 tests
    set, so that the tests stay independent.
    '''
    saved = model._CFG, model._RUN_DIR, model._N_UNEXPECTED
    yield
    model._CFG, model._RUN_DIR, model._N_UNEXPECTED = saved


###_____isocalc inputs_____###

# the initial inventories N_X [atoms] and mean particle mass mu [kg] of the composition recipe of isofate_MCMC.py
# (lines 139-149) at Mp = 5.6 M_earth, for (log_f_atm, log_OtoH_enhancement)
LEGACY_REFERENCE = {
    (-2.8, 0.0): {'N_H': 2.2526852795748566e+49, 'N_He': 2.2189469206995597e+48, 'N_D': 4.433989440316896e+44,
                  'N_O': 1.3256257089607216e+46, 'N_C': 6.628128544803608e+45, 'N_N': 1.8284492537389264e+45,
                  'N_S': 4.114010820912584e+44, 'mu': 2.1400552706020595e-27},
    (-2.5, 2.0): {'N_H': 3.3857478700674325e+48, 'N_He': 4.427381169788069e+48, 'N_D': 8.846972035234255e+44,
                  'N_O': 2.644971020843231e+48, 'N_C': 1.3224855104216152e+46, 'N_N': 3.64823589081825e+45,
                  'N_S': 8.208530754341062e+44, 'mu': 1.0094823846760542e-26},
    (-4.0, -1.0): {'N_H': 1.4332963362540399e+48, 'N_He': 1.4000608566270958e+47, 'N_D': 2.797658202715566e+43,
                   'N_O': 8.364132771005301e+43, 'N_C': 4.18206638550265e+44, 'N_N': 1.1536734856559036e+44,
                   'N_S': 2.595765342725783e+43, 'mu': 2.124833644368505e-27},
}


def test_legacy_composition_reproduces_common():
    '''
    Checks that the legacy composition reproduces, bit for bit, the initial inventories of the recipe of
    isofate_MCMC.py (LEGACY_REFERENCE), and that the other isocalc arguments are the ones that script passes.
    '''
    cfg = example('params.t_pms=1e8', 'params.pms_factor=100', backend = 'isocalc') # that script's PMS boost
    for (log_f_atm, log_O), ref in LEGACY_REFERENCE.items():
        params = prm.theta_to_params([log_f_atm, 5.6, 5e9], cfg)
        params['log_OtoH_enhancement'] = log_O
        args, kwargs = model.planet_inputs(params, cfg)
        for key, value in ref.items():
            assert kwargs[key] == value, (log_f_atm, log_O, key) # bit for bit
    # the other inputs, as isofate_MCMC.py passes them: (f_atm, Mp [kg], Mstar [kg], F0 [W/m2], Fp [W/m2], T [K],
    # d [m], age [yr]) of the last point above, then the mechanism and rad_evol
    assert args[:8] == (1e-4, 5.6*c.Me, 0.18*c.Ms, 0.5876283244560114, 587.6283244560115, 225.61341420087453,
                        14035292576.12045, 5e9)
    assert args[8:] == ('XUV', True)
    expected = dict(melt_fraction_override = False, eps = 0.15, activity = 'medium', flux_model = 'power law',
                    stellar_type = 'M1', Rp_override = False, step_fn = True, F_final = 0.17, t_pms = 1e8,
                    pms_factor = 100.0, n_steps = 100000, t0 = 1e6, rho_rcb = 1.0, RR = True, thermal = True,
                    beta = -1.23, n_atmodeller = 100, save_molecules = False, mantle_iron_dict = False,
                    dynamic_phi = True)
    assert {key: kwargs[key] for key in expected} == expected
    # the saturation time of isofate_MCMC.py, (5.9 - 15.4 Mstar/M_sun) Gyr
    assert kwargs['t_sat'] == pytest.approx((5.9 - 15.4*0.18)*1e9, rel = 1e-15)


def test_ratios_composition():
    '''
    Checks the ratios composition: the initial inventories add up to the atmosphere's mass, mu is its mean particle
    mass, and each number ratio X/H is the one requested (protosolar He/H, solar D/H, and O, C, N, S protosolar
    times Z = 10**log_metallicity, with the extra O/H and C/H enhancements). Also the n_steps, n_atmodeller and
    save_molecules arguments of planet_inputs, and that isocalc gets its own copy of mantle_iron.
    '''
    cfg = example('model.composition="ratios"', 'params.log_metallicity=0.5', 'params.log_OtoH_enhancement=0.3',
                  'params.log_CtoH_enhancement=-0.2', 'model.mantle_iron={type = "static", Fe_mass_fraction = 0.1}')
    params = prm.theta_to_params(CENTRE, cfg)
    M_atm = 5.6*c.Me*10**-2.8 # [kg]
    N, mu = model.composition(params, cfg, M_atm)
    assert sum(N[X]*model.MU[X] for X in model.SPECIES) == pytest.approx(M_atm, rel = 1e-13)
    assert mu == pytest.approx(M_atm/sum(N.values()), rel = 1e-15)
    Z = 10**0.5
    requested = {'He': c.HetoH_protosolar, 'D': c.DtoH_solar, 'O': c.OtoH_protosolar*Z*10**0.3,
                 'C': c.CtoH_protosolar*Z*10**-0.2, 'N': c.NtoH_protosolar*Z, 'S': c.StoH_protosolar*Z}
    for X, r in requested.items():
        assert N[X]/N['H'] == pytest.approx(r, rel = 1e-13)
    args, kwargs = model.planet_inputs(params, cfg, n_steps = 3000, n_atmodeller = 30, save_molecules = True)
    assert (kwargs['n_steps'], kwargs['n_atmodeller'], kwargs['save_molecules']) == (3000, 30, True)
    assert kwargs['mantle_iron_dict'] == cfg['model']['mantle_iron']
    assert kwargs['mantle_iron_dict'] is not cfg['model']['mantle_iron'] # isocalc mutates it
    # T: the equilibrium temperature at the default albedo 0; F0 = Fp*10**log_F0_ratio, log_F0_ratio = -3
    assert args[5] == EqTemp(cfg['planet']['Fp'], A = 0.0) and args[3] == cfg['planet']['Fp']*1e-3


###_____Status codes and blobs (toy backend)_____###

def test_blob_schema():
    '''
    Checks the layout of the blobs emcee stores with every sample: unique fields; status, runtime, log_prior and
    log_like first, then every registry observable, the extras, and the three solve-mode fields last; all float64,
    and NAN_BLOB all NaN.
    '''
    assert len(set(model.BLOB_FIELDS)) == len(model.BLOB_FIELDS)
    assert model.BLOB_FIELDS[:4] == ('status', 'runtime', 'log_prior', 'log_like')
    assert model.BLOB_FIELDS[4:4 + len(config.OBSERVABLES)] == tuple(config.OBSERVABLES)
    assert model.BLOB_FIELDS[-8:-3] == ('exhausted', 't_lost', 'Matm', 'subcritical', 'n_atmod_calls')
    assert model.BLOB_FIELDS[-3:] == ('solve_value', 'solve_evals', 'log_jac') # solve mode: P*, runs, ln|dP*/du|
    assert model.blob_dtype() == [(f, 'f8') for f in model.BLOB_FIELDS]
    assert len(model.NAN_BLOB) == len(model.BLOB_FIELDS) and all(math.isnan(x) for x in model.NAN_BLOB)


def test_statuses(tmp_path, monkeypatch, worker_globals):
    '''
    Checks the status code of each kind of evaluation: 0 ok; 1 outside the prior (no model run); 3 when the input
    terms alone are -inf (no model run) or the data rule the result out (C/O detected in a stripped atmosphere);
    4 for invalid inputs (N_H <= 0 in the legacy composition); 2 for the expected model failures; 5 for any other
    exception, a bug, with its traceback in worker_errors/<pid>.log. A bug in extract is a 5, never a 2.
    '''
    cfg = example()
    ok = model.evaluate(CENTRE, cfg)
    assert ok['status'] == model.OK and math.isfinite(ok['lp']) and ok['runtime'] > 0 and ok['error'] is None
    assert ok['lp'] == ok['log_prior'] + ok['log_like'] and set(ok['terms']) == {'Mp', 'Rp'}
    assert set(ok['obs']) == set(model.OBS_FIELDS) and ok['params']['Mp'] == 5.6

    outside = model.evaluate([-2.8, 7.0, 5e9], cfg) # Mp above its prior, [4.5, 6.7]
    assert outside['status'] == model.OUTSIDE_PRIOR and outside['lp'] == -math.inf and outside['runtime'] == 0
    assert all(math.isnan(v) for v in outside['obs'].values())

    tight = example('observables.Mp.sigma=1e-200') # the input term alone is -inf: the model is not run
    skipped = model.evaluate([-2.8, 6.5, 5e9], tight)
    assert skipped['status'] == model.NO_LIKELIHOOD and skipped['runtime'] == 0 and skipped['terms']['Mp'] == -math.inf
    assert skipped['obs']['Mp'] == 6.5 and math.isnan(skipped['obs']['Rp'])

    detected = example('observables.C_O={use = true, value = 0.5, sigma = 0.1}') # C/O of a stripped atmosphere
    stripped = model.evaluate(STRIPPED, detected)
    assert stripped['status'] == model.NO_LIKELIHOOD and stripped['lp'] == -math.inf and stripped['runtime'] > 0
    assert stripped['obs']['exhausted'] == 1 and stripped['terms']['C_O'] == -math.inf

    invalid = model.evaluate(CENTRE, example('params.log_OtoH_enhancement=2.1')) # N_H < 0 above 2.03
    assert invalid['status'] == model.INVALID_INPUTS and 'N_H' in invalid['error']

    # a stand-in with the name of JAX's runtime error, which counts as a model failure
    class XlaRuntimeError(Exception):
        pass

    for error in (ValueError('melt grid'), RuntimeError('solver'), FloatingPointError(), ZeroDivisionError(),
                  np.linalg.LinAlgError(), XlaRuntimeError('jax')):
        def fail(*a, error = error):
            raise error
        monkeypatch.setattr(model, 'run_model', fail)
        res = model.evaluate(CENTRE, cfg)
        assert res['status'] == model.MODEL_FAILED and type(error).__name__ in res['error']

    model._RUN_DIR = str(tmp_path) # where the tracebacks of status 5 go
    for error in (KeyError('n_H2_a'), TypeError(), NameError(), AttributeError(), IndexError(), ImportError()):
        def fail(*a, error = error):
            raise error
        monkeypatch.setattr(model, 'run_model', fail)
        assert model.evaluate(CENTRE, cfg)['status'] == model.UNEXPECTED
    log = tmp_path/'worker_errors'/f'{os.getpid()}.log'
    assert log.exists() and 'KeyError' in log.read_text() and 'ImportError' in log.read_text()
    monkeypatch.undo() # the real run_model again

    def broken_extract(*a, **k):
        raise ValueError('a bug in extract')
    monkeypatch.setattr(model, 'extract', broken_extract) # never counted as a model failure
    assert model.evaluate(CENTRE, cfg)['status'] == model.UNEXPECTED


def test_log_prob_and_eval_task(worker_globals):
    '''
    Checks the functions a worker pool calls. log_prob needs init_worker first and returns (lp,) plus the blob, all
    floats and equal to evaluate's; a rejected point has the same length. eval_task returns the evaluate result
    without the solution, plus id, theta and, given a track grid, the track (NaN arrays when the model did not
    run).
    '''
    cfg = example()
    with pytest.raises(RuntimeError, match = 'init_worker'):
        model._CFG = None
        model.log_prob(np.array(CENTRE))
    model.init_worker(cfg)
    out = model.log_prob(np.array(CENTRE))
    assert len(out) == 1 + len(model.BLOB_FIELDS)
    assert all(isinstance(x, float) for x in out)
    blobs = np.array([out[1:]], dtype = model.blob_dtype())
    res = model.evaluate(CENTRE, cfg)
    assert out[0] == res['lp'] and blobs['Rp'][0] == res['obs']['Rp'] and blobs['status'][0] == 0
    rejected = model.log_prob(np.array([-2.8, 7.0, 5e9]))
    assert rejected[0] == -math.inf and rejected[1] == model.OUTSIDE_PRIOR and len(rejected) == len(out)

    grid = np.geomspace(1e6, 1e10, 40) # [yr]
    task = model.eval_task({'theta': CENTRE, 'id': 3, 'track_grid': grid, 'save_molecules': True,
                            'n_steps': 5000, 'n_atmodeller': 50})
    assert task['id'] == 3 and task['theta'] == CENTRE and 'sol' not in task and task['status'] == 0
    assert set(task['track']) == set(model.TRACK_KEYS)
    assert task['obs']['n_atmod_calls'] == 5000//50 + 1 # one every 50 steps, plus the final call
    plain = model.eval_task({'theta': CENTRE})
    assert 'track' not in plain and plain['id'] is None and plain['obs'] == res['obs']
    nothing = model.eval_task({'theta': [-2.8, 7.0, 5e9], 'track_grid': grid})
    assert nothing['status'] == 1 and all(np.isnan(v).all() for v in nothing['track'].values())


def test_extract_and_tracks_toy():
    '''
    Checks extract and make_track on the toy. At the centre: no exhaustion, VMRs summing to 1, Mdot >= Mdot_H > 0
    and Teq = 225.61 K; the track has every key (N_X in float64, the rest float32), is NaN after the end of the
    run, and has VMRs only with save_molecules. Stripped: fatm = Mdot = 0, Rp = the bare core, composition and
    escape ratios NaN, t_lost before the age, and the composition NaN in the track from then on.
    '''
    cfg = example()
    res = model.evaluate(CENTRE, cfg, save_molecules = True, return_sol = True)
    sol, obs = res['sol'], res['obs']
    assert obs == model.extract(sol, res['params'], cfg)
    assert obs['exhausted'] == 0 and math.isnan(obs['t_lost']) and obs['n_atmod_calls'] == 1001
    assert sum(obs[f'VMR_{mol}'] for mol in model.MOLECULES) == pytest.approx(1, abs = 1e-12)
    assert obs['Mdot'] >= obs['Mdot_H'] > 0 and obs['Teq'] == pytest.approx(225.61, abs = 0.01)

    grid = np.geomspace(1e6, 8e9, 60) # [yr], past the age of 5 Gyr
    track = model.make_track(sol, grid, cfg)
    assert list(track) == list(model.TRACK_KEYS)
    for key, values in track.items():
        assert values.shape == grid.shape
        assert values.dtype == (np.float64 if key.startswith('N_') else np.float32)
    after = grid > 5e9*(1 + 1e-5) # after the run's last time, age + dt with dt = (age - t0)/n_steps
    assert np.isnan(track['Rp'][after]).all() and np.isfinite(track['Rp'][~after]).all()
    assert np.isfinite(track['N_H'][~after]).all() and track['N_H'][0] > 1e45 # would overflow float32
    vmr = sum(track[f'VMR_{mol}'] for mol in model.MOLECULES)
    np.testing.assert_allclose(vmr[~after], 1, rtol = 1e-5)
    no_molecules = model.make_track(model.evaluate(CENTRE, cfg, return_sol = True)['sol'], grid, cfg)
    assert np.isnan(no_molecules['VMR_H2']).all() and np.isfinite(no_molecules['Rp'][~after]).all()

    stripped = model.evaluate(STRIPPED, cfg, return_sol = True)
    obs = stripped['obs']
    assert obs['exhausted'] == 1 and obs['fatm'] == 0 and obs['Mdot'] == 0 and math.isnan(obs['subcritical'])
    assert obs['Rp'] == pytest.approx(5.6**0.25, rel = 1e-12) # bare core
    assert all(math.isnan(obs[key]) for key in COMPOSITION + ('escape_He_H', 'escape_H_He'))
    assert 4e9 < obs['t_lost'] < 5e9
    track = model.make_track(stripped['sol'], grid, cfg)
    lost = (grid > obs['t_lost']*1.01) & ~after
    assert np.isnan(track['H_He'][lost]).all() and np.allclose(track['Rp'][lost], 5.6**0.25)


def test_outflow_observables_toy():
    '''
    Checks the outflow observables on the toy: Mdot_HHe = Mdot_H + Mdot_He, the H and He number fractions of the
    outflow add up to 1, and both are NaN without a wind. On hand-made fluxes an H-only wind gives exactly 1 and an
    He-only one 0. Against the LHS 1140 b datum (0.00115) a pure H wind is finite but far off (a term of about -75)
    and no wind is ruled out.
    '''
    cfg = example()
    obs = model.evaluate(CENTRE, cfg)['obs']
    assert obs['Mdot_HHe'] == obs['Mdot_H'] + obs['Mdot_He'] and obs['Mdot'] >= obs['Mdot_HHe'] > 0
    assert obs['outflow_H_fraction'] + obs['outflow_He_fraction'] == pytest.approx(1, abs = 1e-15)
    stripped = model.evaluate(STRIPPED, cfg)['obs'] # no wind at all: undefined
    assert math.isnan(stripped['outflow_H_fraction']) and math.isnan(stripped['outflow_He_fraction'])
    assert stripped['Mdot_HHe'] == 0
    # the definitions on hand-made fluxes: an H-only wind is exactly 1, an He-only one 0
    sol = model.evaluate(CENTRE, cfg, return_sol = True)['sol']
    params = prm.theta_to_params(CENTRE, cfg)
    for Phi_H, Phi_He, x in ((3.0, 0.0, 1.0), (0.0, 2.0, 0.0), (1.0, 3.0, 0.25)): # [atoms/m2/s]
        sol['Phi_H'][-1], sol['Phi_He'][-1] = Phi_H, Phi_He
        out = model.extract(sol, params, cfg)
        assert out['outflow_H_fraction'] == x and out['outflow_He_fraction'] == 1 - x
        area = 4*np.pi*sol['Rp'][-1]**2 # [m2]
        assert out['Mdot_HHe'] == pytest.approx((Phi_H*c.mu_H + Phi_He*c.mu_He)*area, rel = 1e-14)
    track = model.make_track(sol, np.geomspace(1e6, 5e9, 30), cfg)
    assert track['outflow_H_fraction'].dtype == np.float32 and np.isfinite(track['Mdot_HHe']).all()
    # the likelihood of the LHS 1140 b data: x = 1 (H-only) is finite, ~2.94 dex from 0.00115; no wind is -inf
    lhs = config.load_config(EXAMPLE, ['model.backend="toy"'])
    o = lhs['observables']['outflow_H_fraction']
    assert likelihood.term(1.0, o) == pytest.approx(-0.5*(np.log10(0.00115)/0.240)**2) # about -75
    assert likelihood.term(math.nan, o) == -math.inf and likelihood.term(0.0, o) < -1000 # floor 1e-15


def test_rp_fixed_toy():
    '''
    Checks Rp_fixed on the toy: isocalc gets Rp_override = Rp_fixed [m] with rad_evol False, and the radius stays
    at Rp_fixed for the whole run. Unset, Rp_override is False (isocalc's own radius).
    '''
    cfg = example('params.Rp_fixed=1.9')
    assert cfg['model']['rad_evol'] is False
    res = model.evaluate(CENTRE, cfg, return_sol = True)
    args, kwargs = model.planet_inputs(res['params'], cfg)
    assert kwargs['Rp_override'] == 1.9*c.Re and args[9] is False # args[9]: rad_evol
    assert res['obs']['Rp'] == pytest.approx(1.9, rel = 1e-14) and np.all(res['sol']['Rp'] == 1.9*c.Re)
    assert model.planet_inputs(prm.theta_to_params(CENTRE, example()), example())[1]['Rp_override'] is False


def test_atmodeller_calls():
    '''
    Checks the number of Atmodeller calls of a run whose atmosphere lasted n_end of n_steps steps: one every
    n_atmodeller steps while it lasts, plus the final call when it lasted to the end; none without Atmodeller or
    without an atmosphere.
    '''
    assert model.atmodeller_calls(100000, 100000, 100) == 1001
    assert model.atmodeller_calls(2000, 2000, 100) == 21
    assert model.atmodeller_calls(150, 1000, 100) == 2
    assert model.atmodeller_calls(1, 2000, 100) == 1
    assert model.atmodeller_calls(0, 2000, 100) == 0
    assert model.atmodeller_calls(1000, 1000, 0) == 0
    assert model.atmodeller_calls(1000, 1000, 1000) == 2 # the call at step 0 and the final call


###_____Worker pool (toy backend)_____###

def test_pool_toy(worker_globals):
    '''
    Checks the WorkerPool with 2 spawned workers on the toy: map gives the results of serial log_prob calls, bit
    for bit except the timing; the workers ignore SIGINT, since Ctrl-C is the parent's to handle; and
    imap_unordered returns every task.
    '''
    cfg = example()
    thetas = list(prm.sample_prior(cfg, 8, 0)) + [np.array([-2.8, 7.0, 5e9])] # and one outside the prior
    model.init_worker(cfg)
    serial = [model.log_prob(theta) for theta in thetas]
    with WorkerPool(cfg, 2) as pool:
        assert pool.n == 2 and pool.warmup_s < 60
        parallel = pool.map(model.log_prob, thetas)
        assert worker_rss_gb(pool) > 0
        assert pool.map(signal.getsignal, [signal.SIGINT]*2) == [signal.SIG_IGN]*2 # Ctrl-C is the parent's
        tasks = [{'theta': theta, 'id': i} for i, theta in enumerate(thetas)]
        unordered = sorted(pool.imap_unordered(model.eval_task, tasks), key = lambda r: r['id'])
    runtime = model.BLOB_FIELDS.index('runtime') + 1 # its position in a log_prob tuple
    for a, b in zip(serial, parallel):
        assert len(b) == 1 + len(model.BLOB_FIELDS)
        np.testing.assert_array_equal(np.delete(a, runtime), np.delete(b, runtime)) # bitwise except timing
    assert [r['id'] for r in unordered] == list(range(len(thetas)))
    assert [r['lp'] for r in unordered] == [p[0] for p in parallel]


def test_choose_workers(capsys):
    '''
    Checks the worker count under the memory budget: about 0.75 + 1.1e-6 n_steps GB per worker (0.86 GB at 1e5,
    0.15 GB on the toy), cut to fit min(max_memory_gb, 70% of the available memory), or max_memory_gb alone with
    check_available = false; never below one worker. A config.json saved before check_available existed is checked.
    '''
    cfg = example(backend = 'isocalc')
    assert est_worker_gb(1e5) == pytest.approx(0.86) and est_worker_gb(1e6, 'toy') == 0.15
    cfg['parallel']['max_memory_gb'] = 2.0
    n, reason = choose_workers(6, 1e5, cfg) # at most floor(2.0/0.86) = 2
    assert 1 <= n <= 2 and 'budget' in reason
    if n < 6:
        assert 'reducing workers' in capsys.readouterr().out
    cfg['parallel']['max_memory_gb'] = 0.1
    assert choose_workers(None, 1e6, cfg)[0] == 1
    cfg['parallel'].update(max_memory_gb = 8.0, check_available = False) # max_memory_gb alone: floor(8/0.86) = 9
    n, reason = choose_workers(6, 1e5, cfg)
    assert n == 6 and 'not checked' in reason
    assert choose_workers(6, 1e6, cfg)[0] == 4 # floor(8/1.85)
    del cfg['parallel']['check_available'] # a config.json from before the option: checked
    assert 'available)' in choose_workers(6, 1e5, cfg)[1]


def test_parent_imports_stay_light():
    '''
    Checks that importing the parent-side modules of isofate.mcmc loads neither jax, atmodeller, numba nor the
    coupled model: the parent process only coordinates, and the model, with its memory, lives in the workers.
    '''
    code = ('import sys, isofate.mcmc, isofate.mcmc.config, isofate.mcmc.params, isofate.mcmc.likelihood, '
            'isofate.mcmc.model, isofate.mcmc.pool, isofate.mcmc.toy, isofate.mcmc.env; '
            "bad = [m for m in ('jax', 'atmodeller', 'numba', 'isofate.isofate_coupler') if m in sys.modules]; "
            'assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], check = True)


###_____Real isocalc (slow)_____###

@pytest.fixture(scope = 'module')
def real():
    '''
    Three real isocalc runs at n_steps = 2000, shared by the slow tests: the centre, the centre with
    save_molecules, and a stripped planet
    '''
    # n_atmodeller = 20, not 100: the kernel rows at the last step carry the atmosphere/mantle split of the last
    # regular Atmodeller call, n_atmodeller*2.5 Myr old at 2000 steps, while atmodeller_final re-equilibrates.
    # Measured C_O vs N_C/N_O: 2.0% apart at 2000/100, 0.37% at 2000/20, 0.034% at the fiducial 1e5/100.
    cfg = example('model.n_steps=2000', 'model.n_atmodeller=20', backend = 'isocalc')
    return {'cfg': cfg,
            'centre': model.evaluate(CENTRE, cfg, return_sol = True),
            'molecules': model.evaluate(CENTRE, cfg, save_molecules = True, return_sol = True),
            'stripped': model.evaluate(STRIPPED, cfg, return_sol = True)}


@pytest.mark.slow
def test_real_extract(real):
    '''
    Checks extract on a real isocalc run at the centre: a radius of 1.6-1.9 R_earth, VMRs summing to 1, C/O within
    1% of N_C/N_O, one Atmodeller call every 20 steps plus the final one, a hot surface (T_surf > 1000 K), mmw of
    2-5 g/mol, Mdot = Mdot_H + Mdot_He, and He from conservation agreeing with the kernel row.
    '''
    res = real['centre']
    obs, sol = res['obs'], res['sol']
    assert res['status'] == model.OK and math.isfinite(res['lp'])
    assert math.isfinite(obs['Rp']) and 1.6 < obs['Rp'] < 1.9
    assert abs(sum(obs[f'VMR_{mol}'] for mol in model.MOLECULES) - 1) < 1e-6
    assert obs['C_O'] == pytest.approx(sol['N_C'][-1]/sol['N_O'][-1], rel = 0.01)
    assert obs['exhausted'] == 0 and obs['n_atmod_calls'] == 2000//20 + 1 and obs['T_surf'] > 1000
    assert 2 < obs['mmw'] < 5 and obs['Mdot'] == pytest.approx(obs['Mdot_H'] + obs['Mdot_He'], rel = 1e-3)
    n = model.final_molecules(sol) # He from conservation agrees with the kernel row
    assert n['He'] == pytest.approx(sol['N_He'][-1]/c.avogadro, rel = 1e-3)
    assert obs == model.extract(sol, res['params'], real['cfg'])


@pytest.mark.slow
def test_real_exhausted(real):
    '''
    Checks a real stripped run: fatm = Mdot = 0, a finite t_lost, Rp = the bare core, the composition NaN, and a
    finite lp (an Rp detection of a bare core is a valid point). The atmosphere is lost within the first 20 steps
    (50 Myr), so there was one Atmodeller call, and atmodeller_final holds its 21 NaN entries.
    '''
    res = real['stripped']
    obs = res['obs']
    assert obs['exhausted'] == 1 and obs['fatm'] == 0 and obs['Mdot'] == 0 and math.isfinite(obs['t_lost'])
    assert obs['Rp'] == pytest.approx(5.6**0.25, rel = 1e-12) # bare core
    assert all(math.isnan(obs[key]) for key in COMPOSITION)
    assert res['status'] == model.OK and math.isfinite(res['lp']) # an Rp detection of a bare core is finite
    assert len(res['sol']['atmodeller_final']) == 21 and obs['n_atmod_calls'] == 1


@pytest.mark.slow
def test_real_outflow_observables(real):
    '''
    Checks the outflow observables of real runs. The centre (log_f_atm -2.8, 5 Gyr) is H-rich and subcritical: He
    does not escape and the outflow is pure H. A stripped planet has no H/He wind at all, so both fractions are
    undefined. Mdot_HHe follows from the fluxes and the radius of the last step.
    '''
    obs = real['centre']['obs']
    assert obs['subcritical'] == 1 and obs['Mdot_He'] == 0
    assert obs['outflow_H_fraction'] == 1.0 and obs['outflow_He_fraction'] == 0.0
    assert obs['Mdot_HHe'] == obs['Mdot_H'] > 1e4 and obs['Mdot_HHe'] == pytest.approx(obs['Mdot'], rel = 1e-3)
    # stripped: no H/He wind at all, so the fractions are undefined
    gone = real['stripped']['obs']
    assert math.isnan(gone['outflow_H_fraction']) and math.isnan(gone['outflow_He_fraction'])
    assert gone['Mdot_HHe'] == 0
    sol = real['centre']['sol']
    assert obs['Mdot_HHe'] == pytest.approx((sol['Phi_H'][-1]*c.mu_H + sol['Phi_He'][-1]*c.mu_He)*4*np.pi
                                            *sol['Rp'][-1]**2, rel = 1e-12)


@pytest.mark.slow
def test_real_log_prob(real, worker_globals):
    '''
    Checks that log_prob after init_worker returns the lp and blob of evaluate on the real model.
    '''
    model.init_worker(real['cfg'])
    out = model.log_prob(np.array(CENTRE))
    assert len(out) == 1 + len(model.BLOB_FIELDS)
    blob = dict(zip(model.BLOB_FIELDS, out[1:]))
    assert blob['status'] == 0 and blob['Rp'] == real['centre']['obs']['Rp'] and out[0] == real['centre']['lp']


@pytest.mark.slow
def test_toy_has_the_keys_of_isocalc(real):
    '''
    Checks that the toy returns the keys of isocalc, in the same order and with the same array shapes, for an
    ordinary run, one with save_molecules and a stripped one, so that tests on the toy go through the code paths
    of the real model.
    '''
    cfg = real['cfg']
    for key, save_molecules in (('centre', False), ('molecules', True), ('stripped', False)):
        res = real[key]
        args, kwargs = model.planet_inputs(res['params'], cfg, save_molecules = save_molecules)
        fake = toy.isocalc(*args, **kwargs)
        assert list(fake) == list(res['sol']), key
        assert list(fake['atmodeller_final']) == list(res['sol']['atmodeller_final']), key
        assert all(fake[k].shape == res['sol'][k].shape for k in fake if k != 'atmodeller_final')


def test_albedo_sets_the_temperature():
    '''
    Checks that the Bond albedo reaches isocalc as T = EqTemp(Fp, A) = Teq(A = 0)*(1 - A)^(1/4) [K], and the input
    observable Teq with it: 225.6 K for LHS 1140 b at A = 0, 189.7 K at A = 0.5.
    '''
    cfg = example()
    for A in (0.0, 0.5):
        params = dict(prm.theta_to_params(CENTRE, cfg), albedo = A)
        args, kwargs = model.planet_inputs(params, cfg)
        assert args[5] == pytest.approx(EqTemp(cfg['planet']['Fp'], A = A), rel = 1e-14) # T [K]
        assert args[5] == pytest.approx(225.6134*(1 - A)**0.25, rel = 1e-6)
