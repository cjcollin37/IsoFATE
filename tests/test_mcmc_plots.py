'''
Tests of isofate.mcmc.plots on a synthetic MCMC run directory in the formats of SPEC sections 8 and 9:
chain.npz as written by sampler.NpzBackend.save (blobs structured as model.blob_dtype()), config.json from
config.save_resolved, summary.json, and tracks.npz. The model values come from the toy backend.
'''

import json
import os

import numpy as np
import pytest

from isofate.mcmc import config, model, plots
from isofate.mcmc import params as prm

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
OBSERVABLES = ['observables.Mdot.use=true', 'observables.Mdot.space="log10"', 'observables.Mdot.sigma=0.3',
               'observables.Mdot.value=2e5',
               'observables.Mdot.kind="upper"', 'observables.VMR_H2O.use=true', 'observables.VMR_H2O.value=1e-3',
               'observables.VMR_H2O.sigma=[0.5, 1.0]', 'observables.VMR_H2O.space="log10"',
               'observables.C_O.use=true', 'observables.C_O.value=0.5', 'observables.C_O.sigma=0.2']
RP_ONLY = ['params.eps.vary=false', 'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false',
           'solve.param=""', 'sampler.init="prescan"', 'sampler.reset_stuck_at=0'] # without solve mode
PNG = b'\x89PNG\r\n\x1a\n'


def synthetic_cfg(tmp_path):
    return config.load_config(EXAMPLE, ['model.backend=toy', 'model.n_steps=2000', 'model.n_atmodeller=20',
                                        'parallel.cache=false', 'parallel.warmup=false',
                                        'params.log_f_atm.bounds=[-7, -1]', f'output_root="{tmp_path}"', *RP_ONLY,
                                        *OBSERVABLES])


def make_run_dir(folder, n_steps = 60, n_walkers = 8, tracks = True, summary = True, seed = 1):
    '''
    A run directory with a burn-in-like chain (walkers spread over the prior that contract towards the centre),
    blobs of the toy model at every position, config.json, and optionally summary.json and tracks.npz.
    '''
    os.makedirs(folder, exist_ok = True)
    cfg = synthetic_cfg(folder)
    rng = np.random.default_rng(seed)
    lo, hi = prm.prior_box(cfg)
    centre = prm.centre_theta(cfg)
    start = prm.sample_prior(cfg, n_walkers, rng)
    decay = np.exp(-np.arange(n_steps)/8.0)[:, None, None]
    chain = centre + (start - centre)*decay + 0.03*(hi - lo)*rng.standard_normal((n_steps, n_walkers, len(lo)))
    chain = np.clip(chain, lo + 1e-9*(hi - lo), hi - 1e-9*(hi - lo))
    log_prob = np.empty((n_steps, n_walkers))
    blobs = np.empty((n_steps, n_walkers), dtype = model.blob_dtype())
    for i in range(n_steps):
        for w in range(n_walkers):
            res = model.evaluate(chain[i, w], cfg)
            log_prob[i, w] = res['lp']
            blobs[i, w] = model.blob(res)
    # the keys of NpzBackend.save (scratchpad emcee/npz_backend.py)
    state = np.random.RandomState(seed).get_state()
    np.savez_compressed(os.path.join(folder, 'chain.npz'), chain = chain, log_prob = log_prob,
                        accepted = rng.integers(5, n_steps, n_walkers).astype(float), iteration = n_steps,
                        nwalkers = n_walkers, ndim = len(lo), rs_name = np.array(state[0]), rs_key = state[1],
                        rs_pos = state[2], rs_has_gauss = state[3], rs_cached = state[4], blobs = blobs)
    config.save_resolved(cfg, os.path.join(folder, 'config.json'))
    if summary: # the keys plots reads (sampler.write_summary writes many more)
        with open(os.path.join(folder, 'summary.json'), 'w') as f:
            json.dump({'discard': 20, 'thin': 2, 'steps': n_steps, 'converged': False}, f)
    if tracks:
        samples = chain[20:].reshape(-1, len(lo))
        thetas = list(samples[rng.choice(len(samples), 15, replace = False)])
        thetas.append(chain.reshape(-1, len(lo))[np.argmax(log_prob)]) # the MAP
        grid = np.geomspace(cfg['model']['t0'], max(t[cfg['free'].index('age')] for t in thetas), 120)
        out = {key: [] for key in model.TRACK_KEYS}
        status, final = [], []
        for theta in thetas:
            res = model.evaluate(theta, cfg, save_molecules = True, return_sol = True)
            track = model.make_track(res['sol'], grid, cfg) if res['sol'] is not None else model.nan_track(grid.size)
            for key in model.TRACK_KEYS:
                out[key].append(track[key])
            status.append(res['status'])
            final.append(model.blob(res))
        np.savez_compressed(os.path.join(folder, 'tracks.npz'), grid_yr = grid, theta = np.array(thetas),
                            is_map = np.arange(len(thetas)) == len(thetas) - 1, status = np.array(status),
                            blobs = np.array(final, dtype = model.blob_dtype()),
                            **{key: np.array(v) for key, v in out.items()})
    return folder


def is_png(path):
    with open(path, 'rb') as f:
        return f.read(8) == PNG and os.path.getsize(path) > 10000


def test_plot_run_all_files(tmp_path):
    run_dir = make_run_dir(str(tmp_path/'run'))
    paths = plots.plot_run(run_dir)
    names = [os.path.basename(p) for p in paths]
    assert names == ['corner_params.png', 'corner_observables.png', 'traces.png', 'predictive.png', 'tracks.png']
    assert all(is_png(p) and os.path.dirname(p) == run_dir for p in paths)


def test_plot_run_without_optional_files(tmp_path, capsys):
    run_dir = make_run_dir(str(tmp_path/'run'), tracks = False, summary = False)
    paths = plots.plot_run(run_dir)
    assert [os.path.basename(p) for p in paths] == ['corner_params.png', 'corner_observables.png', 'traces.png',
                                                     'predictive.png']
    assert all(is_png(p) for p in paths)
    assert 'warning' not in capsys.readouterr().out # nothing failed, tracks.png just does not apply
    assert plots.plot_dir(run_dir) == paths


def test_discard_thin(tmp_path):
    run_dir = make_run_dir(str(tmp_path/'run'), tracks = False, summary = False)
    run = plots.read_run(run_dir)
    assert run['tracks'] is None and run['summary'] is None
    discard, thin = plots.discard_thin(run) # estimated from the chain: at most half of it
    assert 0 <= discard <= 30 and thin >= 1
    run['summary'] = {'discard': 20, 'thin': 2, 'steps': 60}
    assert plots.discard_thin(run) == (20, 2)
    run['summary']['steps'] = 50 # written for a shorter chain (a resumed run): not used
    assert plots.discard_thin(run) == (discard, thin)
    run['summary']['steps'] = 60
    samples, blobs = plots.posterior(run)
    assert samples.shape == (20*8, 3) and blobs.shape == (20*8,) # (60 - 20)/2 steps x 8 walkers
    assert np.array_equal(samples, run['chain'][21::2].reshape(-1, 3)) # emcee's get_chain(discard, thin)


def test_plot_dir_rejects_other_folders(tmp_path):
    with pytest.raises(FileNotFoundError):
        plots.plot_dir(str(tmp_path))


def test_track_panel_limits():
    # one track far off widens a panel by at most half the core range (linear) or a decade (log)
    lo, hi = plots._linear_limits([1.0, 2.0, np.nan], [0.5, 1.5, 10.0])
    assert lo == pytest.approx(0.5 - 0.08) and hi == pytest.approx(2.5 + 0.08)
    assert plots._linear_limits([np.nan], [1.0]) is None
    lo, hi = plots._log_limits([1e-3, 1e-2], [1e-30, 5e-3])
    assert 1e-4/1.5 < lo < 1e-4 and 1e-2 < hi < 1.3e-2
