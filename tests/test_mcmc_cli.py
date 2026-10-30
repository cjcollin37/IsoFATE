'''
Tests of the command line (python -m isofate.mcmc) on the toy backend, and of the documentation: the README's
tables are generated from the registries and every option appears in the README and the template.
'''

from isofate.mcmc import env
env.set_single_thread_env()

import json
import os
import signal
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from isofate.mcmc import config, likelihood, model, sampler
from isofate.mcmc import __main__ as cli
from isofate.mcmc.__main__ import main

README = os.path.join(os.path.dirname(config.__file__), 'README.md')
EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')


def toy_config(tmp_path, **sampler_options):
    '''
    A small toy-backend copy of the LHS 1140 b example in tmp_path, with Mdot observed in log10 space.
    '''
    with open(EXAMPLE, 'rb') as f:
        raw = tomllib.load(f)
    raw['name'] = 'toy'
    raw['output_root'] = str(tmp_path)
    raw['model'].update(backend = 'toy', n_steps = 20000)
    raw['params']['eps'] = {'value': 0.15} # 3 free parameters; the outflow observables off
    raw['observables']['outflow_H_fraction']['use'] = raw['observables']['Mdot_HHe']['use'] = False
    raw['observables']['Mdot'] = {'use': True, 'value': 2.03e5, 'sigma': [0.137, 0.121], 'space': 'log10'}
    raw['solve'] = {'param': ''} # without solve mode
    raw['sampler'] = {'max_steps': 20, 'checkpoint_every': 10, 'print_every': 10, 'check_every': 10,
                      'tracks': 6, 'track_points': 40, **sampler_options}
    raw['parallel'] = {'workers': 2}
    path = tmp_path/'toy.toml'
    path.write_text(config.dump_toml(raw))
    return str(path)


###_____documentation_____###

def readme_block(text, which):
    start = text.index('\n', text.index(f'<!-- BEGIN {which} table')) + 1
    return text[start:text.index(f'<!-- END {which} table -->')].strip()


def test_readme_tables_match_the_registries():
    with open(README) as f:
        text = f.read()
    params, observables = config.markdown_tables()
    assert readme_block(text, 'params') == params, 'regenerate the README tables with config.markdown_tables()'
    assert readme_block(text, 'observables') == observables
    for name in list(config.PARAMS) + list(config.OBSERVABLES):
        assert f'`{name}`' in text


def test_every_option_is_documented():
    with open(README) as f:
        readme = f.read()
    with open(config.TEMPLATE) as f:
        template = f.read()
    with open(EXAMPLE) as f:
        example = f.read()
    for name in config.PARAMS:
        assert f'[params.{name}]' in template
    for name in config.OBSERVABLES:
        assert f'[observables.{name}]' in template and f'[observables.{name}]' in example
    for section, defaults in (('model', config.MODEL_DEFAULTS), ('sampler', config.SAMPLER_DEFAULTS),
                              ('solve', config.SOLVE_DEFAULTS), ('parallel', config.PARALLEL_DEFAULTS),
                              ('resolution', config.RESOLUTION_DEFAULTS)):
        for key in defaults:
            assert f'`{key}' in readme, f'{section}.{key} is not in the README'
            assert f'{key} =' in template, f'{section}.{key} is not in template.toml'


###_____commands_____###

def test_help_and_light_imports():
    out = subprocess.run([sys.executable, '-m', 'isofate.mcmc', '--help'], capture_output = True, text = True,
                         timeout = 60)
    assert out.returncode == 0 and 'resolution' in out.stdout and 'examples' in out.stdout
    code = ('import sys, isofate.mcmc.__main__; '
            "print([m for m in ('jax', 'atmodeller', 'isofate.isofate_coupler') if m in sys.modules])")
    out = subprocess.run([sys.executable, '-c', code], capture_output = True, text = True, timeout = 60)
    assert out.stdout.strip() == '[]'


def test_template(tmp_path, capsys):
    path = str(tmp_path/'t.toml')
    assert main(['template', path]) == 0
    assert main(['template', path]) == 2 # never overwritten without --force
    assert 'exists' in capsys.readouterr().err
    assert main(['template', path, '--force']) == 0
    capsys.readouterr()
    assert main(['template']) == 0
    assert tomllib.loads(capsys.readouterr().out)['name'] == 'my_planet'
    assert config.load_config(path)['free'] == ['log_f_atm', 'age']


def test_check_toy(tmp_path, capsys):
    path = toy_config(tmp_path)
    assert main(['check', path, '--theta', 'log_f_atm=-2.5', '--set', 'model.n_atmodeller=50']) == 0
    out = capsys.readouterr().out
    assert 'status 0 (ok)' in out and 'log_f_atm = -2.5' in out and 'n_atmodeller = 50' in out
    assert 'Mdot' in out and 'All model outputs' in out and 'MCMC cost estimate' in out
    assert main(['check', path, '--theta', 'log_fatm=-2.5']) == 2 # unknown name: a suggestion
    assert "did you mean 'log_f_atm'" in capsys.readouterr().err
    assert main(['check', path, '--theta', 'log_f_atm=-9']) == 2 # outside the bounds
    assert main(['check', str(tmp_path/'missing.toml')]) == 2


def test_synth_round_trip(tmp_path, capsys):
    path = toy_config(tmp_path)
    out = str(tmp_path/'inj.toml')
    args = ['synth', path, '--theta', 'log_f_atm=-2.5', '--theta', 'age=6e9', '--out', out, '--sigma-scale', '0.5',
            '--set', 'observables.VMR_He.use=true', '--set', 'observables.VMR_He.value=0.3',
            '--set', 'observables.VMR_He.sigma=0.3', '--set', 'observables.VMR_He.kind="upper"']
    assert main(args) == 0
    assert main(args) == 2 # OUT exists
    cfg = config.load_config(out)
    assert cfg['name'] == 'inj' and cfg['output_root'] == str(tmp_path)
    assert cfg['params']['log_f_atm']['value'] == -2.5 and cfg['params']['age']['value'] == 6e9
    assert cfg['observables']['Rp']['sigma'] == 0.0125 and cfg['observables']['Mdot']['sigma'] == (0.0685, 0.0605)
    res = model.evaluate(np.array([-2.5, 5.6, 6e9]), cfg)
    assert res['obs']['Rp'] == cfg['observables']['Rp']['value']
    assert res['terms']['Rp'] == 0 and res['terms']['Mp'] == 0 and abs(res['terms']['Mdot']) < 1e-20
    assert res['terms']['VMR_He'] == pytest.approx(np.log(0.5)) # an upper limit at the model value
    # an undefined prediction (the composition of a stripped atmosphere) cannot become data
    args = ['synth', path, '--theta', 'log_f_atm=-5', '--theta', 'age=1e10', '--out', str(tmp_path/'bad.toml'),
            '--set', 'observables.VMR_He.use=true', '--set', 'observables.VMR_He.value=0.3',
            '--set', 'observables.VMR_He.sigma=0.3']
    assert main(args) == 2
    assert 'VMR_He = nan' in capsys.readouterr().err and not os.path.exists(tmp_path/'bad.toml')


def test_resolution_dry_run(tmp_path, capsys):
    path = toy_config(tmp_path)
    assert main(['resolution', path, '--grid', 'quick', '--n-steps', '1e4,3e4', '--points', '2', '--dry-run']) == 0
    out = capsys.readouterr().out
    assert '6 cells x 2 test points x 2 runs + 2 = 26 evaluations' in out
    assert 'n_steps: 1e4, 3e4;  n_atmodeller: 1000, 100, 10' in out
    assert main(['resolution', path, '--n-steps', '1e4', '--n-atmodeller', '100,10', '--dry-run']) == 0
    assert 'n_atmodeller: 100, 10' in capsys.readouterr().out
    assert not os.path.exists(tmp_path/'toy') # a dry run creates nothing


def test_run_resume_summary_tracks_plot(tmp_path, capsys):
    path = toy_config(tmp_path)
    assert main(['run', path, '--max-steps', '10', '--no-plots']) == 0
    (run_dir,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    assert sampler.load_run(run_dir)['chain'].shape == (10, 16, 3)
    with open(os.path.join(run_dir, 'config.toml')) as f:
        assert 'backend = "toy"' in f.read()
    assert main(['run', '--resume', run_dir, '--set', 'model.n_steps=1e4']) == 2 # the config is fixed
    assert main(['run', '--resume', run_dir, '--no-plots']) == 0 # to sampler.max_steps = 20
    run = sampler.load_run(run_dir)
    assert run['chain'].shape == (20, 16, 3) and run['summary']['steps'] == 20
    capsys.readouterr()
    assert main(['summary', run_dir]) == 0
    out = capsys.readouterr().out
    assert 'posterior of toy (finished)' in out and 'log-likelihood terms' in out and 'Mdot' in out
    assert main(['summary', run_dir, '--json']) == 0
    assert json.loads(capsys.readouterr().out)['steps'] == 20
    assert main(['summary', run_dir, '--update']) == 0 # recomputed, still 'finished'
    assert 'posterior of toy (finished)' in capsys.readouterr().out
    assert main(['tracks', run_dir, '--n', '4', '--workers', '1', '--no-plots']) == 0
    with np.load(os.path.join(run_dir, 'tracks.npz')) as d:
        # 40 log-spaced times plus the ages of the 5 tracks (each track ends on the grid)
        assert d['theta'].shape == (5, 3) and d['Rp'].shape == (5, d['grid_yr'].size)
        assert 40 < d['grid_yr'].size <= 45
    capsys.readouterr()
    assert main(['plot', run_dir]) == 0
    names = sorted(os.path.basename(p) for p in capsys.readouterr().out.split())
    assert names == ['corner_observables.png', 'corner_params.png', 'predictive.png', 'traces.png', 'tracks.png']
    assert main(['summary', str(tmp_path)]) == 2 # not a run directory


def test_interrupt_exit_code(tmp_path, monkeypatch, capsys):
    path = toy_config(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(sampler, 'run_mcmc', interrupted)
    assert main(['run', path]) == 130
    assert 'interrupted' in capsys.readouterr().err


def test_signal_handlers_and_nohup(monkeypatch):
    # SIGTERM and SIGHUP take the Ctrl-C path during a command, and are restored after it; a hang-up that is
    # ignored already (nohup) stays ignored, so that the run goes on after the terminal closes
    seen = {}

    def probe(args):
        seen.update(term = signal.getsignal(signal.SIGTERM), hup = signal.getsignal(signal.SIGHUP))
        return 0
    monkeypatch.setattr(cli, 'cmd_template', probe)
    previous = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        assert main(['template']) == 0
        assert seen == {'term': cli._terminate, 'hup': cli._terminate}
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        assert main(['template']) == 0
        assert seen == {'term': cli._terminate, 'hup': signal.SIG_IGN}
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)


###_____fixes found while integrating_____###

def test_log10_escape_rates_are_not_capped():
    '''
    Rates [kg/s] are clipped only below (floor); dimensionless ratios also above, at 1/floor.
    '''
    cfg = config.load_config(EXAMPLE, ['observables.Mdot.use=true', 'observables.Mdot.space="log10"',
                                       'observables.Mdot.value=2e5', 'observables.Mdot.sigma=0.13',
                                       'observables.escape_H_He.use=true', 'observables.escape_H_He.space="log10"',
                                       'observables.escape_H_He.sigma=0.25'])
    mdot, ratio = cfg['observables']['Mdot'], cfg['observables']['escape_H_He']
    assert mdot['ceiling'] == np.inf and ratio['ceiling'] == pytest.approx(1e12)
    assert likelihood.term(2e5, mdot) == 0.0
    assert likelihood.term(2e6, mdot) == pytest.approx(-0.5*(1/0.13)**2)
    assert likelihood.term(0.0, mdot) == pytest.approx(-0.5*((-3 - np.log10(2e5))/0.13)**2) # floor 1e-3
    assert likelihood.to_space(np.inf, ratio) == pytest.approx(12.0)
    old = {k: v for k, v in mdot.items() if k != 'ceiling'} # config.json written before the ceiling existed
    assert likelihood.to_space(5e4, old) == 3.0


def test_override_strings_parse_back():
    overrides = ['model.backend=toy', 'parallel.cache=false', 'params.age.bounds=[3.5e9, 1e10]',
                 'model.mantle_iron={type = "static", Fe_mass_fraction = 0.1}', 'observables.Mdot.use=true',
                 'observables.Mdot.space="log10"', 'observables.Mdot.value=2e5', 'observables.Mdot.sigma=0.13']
    cfg = config.load_config(EXAMPLE, overrides)
    assert cfg['overrides'][:2] == ['model.backend="toy"', 'parallel.cache=false']
    again = config.load_config(EXAMPLE, cfg['overrides'])
    assert {k: v for k, v in again.items() if k != 'overrides'} == {k: v for k, v in cfg.items() if k != 'overrides'}
    assert again['overrides'] == cfg['overrides']
