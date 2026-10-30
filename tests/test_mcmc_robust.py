'''
Tests of how the tool copes when things go wrong, on the toy backend: a worker that dies (during a map or between
two), hang-ups ignored by the workers, signals to the whole process group, run directories claimed atomically,
the lock of a run directory, SIGTERM during a run, the template's output_root, a stale summary, the --bisect-*
flags, the comparison labels, the checkpoint rate, a stuck-walker reset that would collapse the ensemble, and the
summary's warnings in run.log.
'''

from isofate.mcmc import env

env.set_single_thread_env() # before anything imports jax

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import types

import numpy as np
import pytest

from isofate.mcmc import config, plots, pool, sampler
from isofate.mcmc.__main__ import main

EXAMPLE = os.path.join(config.CONFIG_DIR, 'lhs1140b.toml')
# the example on the toy: 3 free parameters (eps fixed), Rp and Mp only, without solve mode; 16 walkers on 2
# workers, no tracks; a progress line, a checkpoint and a convergence check every 10 steps
BASE = ['model.backend="toy"', 'parallel.cache=false', 'parallel.warmup=false', 'parallel.workers=2',
        'model.n_steps=20000', 'params.eps.vary=false', 'params.albedo.vary=false',
        'observables.outflow_H_fraction.use=false', 'observables.Mdot_HHe.use=false', 'solve.param=""',
        'sampler.init="prescan"', 'sampler.reset_stuck_at=0',
        'sampler.nwalkers=16', 'sampler.tracks=0', 'sampler.print_every=10', 'sampler.checkpoint_every=10',
        'sampler.check_every=10']


def toy_cfg(tmp_path, *overrides):
    '''
    The toy config of BASE, writing to tmp_path, with more --set overrides
    '''
    return config.load_config(EXAMPLE, BASE + [f'output_root="{tmp_path}"', *overrides])


def toy_file(tmp_path, name = 'toy', **sampler_options):
    '''
    Writes the toy config of BASE to tmp_path/<name>.toml, for the commands; 10 steps unless sampler_options say
    otherwise

    Inputs:
        - tmp_path: folder of the file, and its output_root [Path]
        - name: the config's name [str]
        - sampler_options: more [sampler] options [dict]

    Output: path of the config [str]
    '''
    with open(EXAMPLE, 'rb') as f:
        raw = tomllib.load(f)
    raw.update(name = name, output_root = str(tmp_path))
    raw['model'].update(backend = 'toy', n_steps = 20000)
    raw['params']['eps'] = {'value': 0.15}
    raw['params']['albedo'] = {'value': 0.0} # fixed (the example samples it)
    raw['observables']['outflow_H_fraction']['use'] = raw['observables']['Mdot_HHe']['use'] = False
    raw['solve'] = {'param': ''}
    raw['parallel'] = {'workers': 2, 'cache': False, 'warmup': False}
    raw['sampler'] = dict({'max_steps': 10, 'checkpoint_every': 10, 'print_every': 10, 'check_every': 10,
                           'tracks': 0, 'nwalkers': 16, 'init': 'prescan'}, **sampler_options)
    path = tmp_path/f'{name}.toml'
    path.write_text(config.dump_toml(raw))
    return str(path)


###_____Worker pools_____###

def test_dead_worker_raises(tmp_path):
    '''
    Checks that a dead worker stops the pool instead of hanging it. A worker killed during a task: map and
    imap_unordered raise WorkerDied within a few poll intervals, since multiprocessing never returns that task.
    Workers killed while idle, between two maps (kill -TERM of the process group reaches them too): one of them
    waited for its next task holding the task queue's lock, so the pool's replacements could never take a task.
    The next map raises at once, and the teardown, where multiprocessing's own waits for that lock forever, ends
    within its timeout.
    '''
    cfg = toy_cfg(tmp_path)
    with pool.WorkerPool(cfg, 2, verbose = False) as wp:
        assert wp.map(abs, [-1, -2, -3]) == [1, 2, 3] # time.sleep and abs are picklable builtins
        victim = wp._pool._pool[0].pid
        threading.Timer(1.0, os.kill, (victim, signal.SIGKILL)).start() # killed 1 s into the map
        start = time.perf_counter()
        with pytest.raises(pool.WorkerDied, match = r'died \(killed by the OS'):
            wp.map(time.sleep, [4]*4)
        assert time.perf_counter() - start < 4 + 3*pool.POLL # not the 8 s of the tasks, nor forever
    with pool.WorkerPool(cfg, 2, verbose = False) as wp:
        victim = wp._pool._pool[1].pid
        threading.Timer(1.0, os.kill, (victim, signal.SIGKILL)).start()
        with pytest.raises(pool.WorkerDied):
            list(wp.imap_unordered(time.sleep, [4]*4))
    # workers killed while idle, between two maps
    wp = pool.WorkerPool(cfg, 2, verbose = False)
    try:
        assert wp.map(abs, [-1, -2]) == [1, 2]
        time.sleep(0.5) # the workers wait for their next task
        for p in list(wp._pool._pool):
            os.kill(p.pid, signal.SIGTERM)
        time.sleep(1.0) # the pool replaces them
        start = time.perf_counter()
        with pytest.raises(pool.WorkerDied):
            wp.map(abs, [-1, -2])
        assert time.perf_counter() - start < 1.0
    finally:
        start = time.perf_counter()
        wp.terminate(timeout = 3.0)
        assert time.perf_counter() - start < 3.0 + 4.0 # the timeout, then the kills and joins
        assert all(p.exitcode is not None for p in wp._pool._pool)


def test_workers_ignore_hangups(tmp_path):
    '''
    Checks that a worker survives SIGHUP and SIGINT. Ctrl-C and the hang-up of a closing terminal reach the whole
    process group; the parent alone handles them (it checkpoints, then stops the pool).
    '''
    with pool.WorkerPool(toy_cfg(tmp_path), 1, verbose = False) as wp:
        pid = wp._pool._pool[0].pid
        os.kill(pid, signal.SIGHUP)
        os.kill(pid, signal.SIGINT)
        time.sleep(1.0)
        assert wp.map(abs, [-3]) == [3] and [p.pid for p in wp._pool._pool] == [pid] # the same worker, still at work


# a run (argv[1]: its config) that at step 10, while its workers are idle, sends SIGTERM or SIGHUP (argv[2]) to its
# whole process group, or kills its workers alone (argv[2] = 'workers')
SIGNAL_SCRIPT = '''
import os, signal, sys, time
from isofate.mcmc import env
env.set_single_thread_env()
from isofate.mcmc import sampler
from isofate.mcmc.__main__ import main

if __name__ == '__main__':
    progress = sampler._Run.progress

    def signalling(self):
        progress(self)
        if self.step == 10:
            time.sleep(0.5) # the workers wait for the next step: idle, one of them holding the task queue's lock
            if sys.argv[2] == 'workers': # the OS kills the workers alone (e.g. for memory)
                for p in self.pool.pool._pool._pool:
                    os.kill(p.pid, signal.SIGKILL)
                time.sleep(1.0)
            else: # kill -TERM -PGID, or closing the terminal: the whole process group
                os.killpg(os.getpgid(0), getattr(signal, sys.argv[2]))
                time.sleep(5.0) # the parent's handler raises KeyboardInterrupt here
    sampler._Run.progress = signalling
    sys.exit(main(['run', sys.argv[1], '--no-plots']))
'''


@pytest.mark.parametrize('target', ['SIGTERM', 'SIGHUP', 'workers'])
def test_signal_to_the_process_group(tmp_path, target):
    '''
    Checks that a run saves its chain up to the last step and releases its lock however a signal ends it. A signal
    to the whole process group (kill -TERM -PGID, or SIGHUP from a closing terminal) reaches the workers too:
    SIGTERM kills them, while they ignore SIGHUP. A worker that dies idle may hold the task queue's lock, and
    multiprocessing's teardown would then wait for it forever before the chain is saved. Workers killed alone
    (e.g. by the OS for memory) would leave the next step waiting forever. The exit code is 130 after a signal,
    1 after dead workers.
    '''
    script = tmp_path/'group_signal.py' # not signal.py, which would shadow the module
    script.write_text(SIGNAL_SCRIPT)
    path = toy_file(tmp_path, max_steps = 100000, checkpoint_every = 100000) # only the end saves the chain
    # in a process group of its own, so that the signal to the group spares pytest
    proc = subprocess.Popen([sys.executable, str(script), path, target], stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE, text = True, cwd = tmp_path, start_new_session = True)
    try:
        out, err = proc.communicate(timeout = 240)
    except subprocess.TimeoutExpired: # a hung run: kill its whole group
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        pytest.fail(f'the run hung after {target}:\n{out[-2000:]}\n{err[-2000:]}')
    assert proc.returncode == (1 if target == 'workers' else 130), out[-2000:] + err[-2000:]
    (run_dir,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    assert sampler.load_run(run_dir)['iteration'] == 10 and not os.path.exists(os.path.join(run_dir, sampler.LOCK))
    if target == 'workers':
        assert 'died (killed by the OS' in err
    else:
        assert f'python -m isofate.mcmc run --resume {run_dir}' in out


###_____Run directories_____###

def test_output_dirs_are_claimed_atomically(tmp_path):
    '''
    Checks that 12 threads asking for the same run directory at once get 12 different ones (base, base_1, ...,
    base_11): os.makedirs without exist_ok claims a name atomically. An explicitly given folder that holds files is
    refused, never overwritten.
    '''
    base = str(tmp_path/'mcmc_20260101-000000')
    paths = []
    threads = [threading.Thread(target = lambda: paths.append(config.new_output_dir(base))) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(paths)) == 12 and base in paths and f'{base}_11' in paths # base, base_1, ..., base_11
    (tmp_path/'full').mkdir()
    (tmp_path/'full'/'x').write_text('x')
    with pytest.raises(FileExistsError, match = 'never overwrites'):
        config.empty_output_dir(str(tmp_path/'full'), 'a scan')
    assert config.empty_output_dir(str(tmp_path/'new'), 'a scan') == str(tmp_path/'new')


def test_lock_of_a_run_directory(tmp_path):
    '''
    Checks the lock of a run directory, <run_dir>/.lock holding the pid and host of the process that runs it. A lock
    held by a live process on this host refuses the run, and release_lock leaves another process's lock alone. A
    lock whose process is gone is stale and taken over, with a note.
    '''
    folder = str(tmp_path)
    path, note = sampler.acquire_lock(folder)
    assert note is None and open(path).read().split()[0] == str(os.getpid())
    sampler.release_lock(path)
    assert not os.path.exists(path)
    # a live process holds it: refused
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        with open(path, 'w') as f:
            f.write(f'{other.pid} {os.uname().nodename}\n')
        with pytest.raises(FileExistsError, match = f'in use by another session \\(pid {other.pid}'):
            sampler.acquire_lock(folder)
        sampler.release_lock(path) # not ours: kept
        assert os.path.exists(path)
    finally:
        other.kill()
        other.wait()
    # its process is gone: taken over
    path, note = sampler.acquire_lock(folder)
    assert 'stale lock' in note and open(path).read().split()[0] == str(os.getpid())
    sampler.release_lock(path)


def test_resume_refused_while_running(tmp_path):
    '''
    Checks that a run releases its lock at the end, and that a resume of a run whose lock a live process holds is
    refused before anything is sampled.
    '''
    cfg = toy_cfg(tmp_path, 'sampler.max_steps=10')
    run_dir = sampler.run_mcmc(cfg, workers = 2, make_plots = False)
    assert not os.path.exists(os.path.join(run_dir, sampler.LOCK)) # released at the end
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        with open(os.path.join(run_dir, sampler.LOCK), 'w') as f:
            f.write(f'{other.pid} {os.uname().nodename}\n')
        with pytest.raises(FileExistsError, match = 'in use by another session'):
            sampler.run_mcmc(None, run_dir = run_dir, resume = True, max_steps = 20, make_plots = False)
        assert sampler.load_run(run_dir)['iteration'] == 10
    finally:
        other.kill()
        other.wait()


def test_sigterm_saves_the_chain(tmp_path):
    '''
    Checks that SIGTERM (kill PID) during a run takes the Ctrl-C path: the chain is checkpointed at the signal (here
    the only checkpoint), the summary says 'interrupted', the resume command is printed, the lock is released, and
    the exit code is 130.
    '''
    path = toy_file(tmp_path, max_steps = 100000, checkpoint_every = 100000)
    proc = subprocess.Popen([sys.executable, '-m', 'isofate.mcmc', 'run', path, '--no-plots'], stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE, text = True, cwd = tmp_path)
    try:
        log = None
        for _ in range(600): # until the run has sampled a few dozen steps
            time.sleep(0.1)
            logs = [os.path.join(r, 'run.log') for r, _, files in os.walk(tmp_path) if 'run.log' in files]
            steps = [int(n) for n in re.findall(r'step +(\d+)/100000', open(logs[0]).read())] if logs else []
            if steps and max(steps) >= 40:
                log = logs[0]
                break
        assert log, proc.communicate(timeout = 5)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout = 60)
    finally:
        proc.kill()
    assert proc.returncode == 130, err
    run_dir = os.path.dirname(log)
    assert sampler.load_run(run_dir)['iteration'] >= 40 # checkpointed at the signal, not lost
    assert json.load(open(os.path.join(run_dir, 'summary.json')))['state'] == 'interrupted'
    assert f'run --resume {run_dir}' in out and not os.path.exists(os.path.join(run_dir, sampler.LOCK))


###_____Commands_____###

def test_template_output_root(tmp_path, capsys):
    '''
    Checks the output_root that the template command writes. A template written into a configs/ folder gets
    output_root = "..", so that the results land next to that folder; anywhere else the line is commented out and
    the command says to set it.
    '''
    (tmp_path/'configs').mkdir()
    path = tmp_path/'configs'/'my_planet.toml'
    assert main(['template', str(path)]) == 0
    assert 'with output_root = ".."' in capsys.readouterr().out
    assert '\noutput_root = ".."' in path.read_text() and config.load_config(str(path))['output_root'] == str(tmp_path)
    elsewhere = tmp_path/'my_planet.toml'
    assert main(['template', str(elsewhere)]) == 0
    assert 'set output_root in it' in capsys.readouterr().out and '# output_root = ".."' in elsewhere.read_text()


def test_summary_notices_a_newer_chain(tmp_path, capsys):
    '''
    Checks that summary recomputes a summary.json written for fewer steps than the chain holds (e.g. by an earlier
    session of a resumed run), and says so.
    '''
    assert main(['run', toy_file(tmp_path), '--no-plots']) == 0
    (run_dir,) = [os.path.join(tmp_path, 'toy', d) for d in os.listdir(tmp_path/'toy')]
    path = os.path.join(run_dir, 'summary.json')
    summary = json.load(open(path))
    summary['steps'] = 5 # as written by an earlier session of a resumed run
    json.dump(summary, open(path, 'w'))
    capsys.readouterr()
    assert main(['summary', run_dir]) == 0
    out = capsys.readouterr().out
    assert 'written for 5 steps, the chain has 10' in out and '10 steps x 16 walkers' in out
    assert json.load(open(path))['steps'] == 10


def test_bisect_flags_need_the_bisection(tmp_path, capsys):
    '''
    Checks that --bisect-observable and --bisect-param are refused where no bisection runs (run without
    sampler.init = "bisect", resolution without source = "bisect") and accepted where one does.
    '''
    path = toy_file(tmp_path)
    assert main(['run', path, '--bisect-observable', 'Rp']) == 2
    assert 'apply to sampler.init = "bisect" only' in capsys.readouterr().err
    assert main(['resolution', path, '--bisect-param', 'age', '--dry-run']) == 2
    assert 'source = "bisect" only' in capsys.readouterr().err
    assert main(['resolution', path, '--bisect-observable', 'Rp', '--source', 'bisect', '--dry-run',
                 '--n-steps', '1e4', '--n-atmodeller', '100']) == 0


def test_check_bisect_fails_early(tmp_path, capsys):
    '''
    Checks check --bisect: a bad bisection target is an error before any model run, and a search without a root
    (an Rp datum below the bare-core radius) ends with what to try.
    '''
    path = toy_file(tmp_path)
    assert main(['check', path, '--bisect', '--bisect-param', 'eps']) == 2 # eps is fixed here
    captured = capsys.readouterr()
    assert 'must be a free parameter' in captured.err and 'Evaluating at' not in captured.out
    assert main(['check', path, '--bisect', '--set', 'observables.Rp.value=0.5']) == 1
    out = capsys.readouterr().out
    assert 'bare-core radius' in out and 'or wider bounds may help' in out


def test_compare_default_labels():
    '''
    Checks the default labels of compare: the run directory names when the runs share a config name, the config
    names when those differ, and name/run directory when both are needed.
    '''
    def run(name, folder):
        return {'cfg': {'name': name}, 'run_dir': f'/x/{name}/{folder}'}
    assert plots.default_labels([run('a', 'mcmc_1'), run('a', 'mcmc_2')]) == ['mcmc_1', 'mcmc_2']
    assert plots.default_labels([run('a', 'mcmc_1'), run('b', 'mcmc_1')]) == ['a', 'b']
    assert plots.default_labels([run('a', 'm1'), run('a', 'm2'), run('b', 'm1')]) == ['a/m1', 'a/m2', 'b/m1']


###_____The sampler_____###

def test_checkpoint_rate():
    '''
    Checks the rate of the periodic checkpoints: every checkpoint_every steps, but at most once a minute and at
    most every 20 times the duration of the last save, so that saving a long chain (about 20 s at 1e5 steps) costs
    at most 5% of the run.
    '''
    now = time.perf_counter()
    run = types.SimpleNamespace(s = {'checkpoint_every': 50}, t_saved = now, save_s = 0.01) # a _Run that just saved
    due = lambda step: sampler._Run.checkpoint_due(run, step)
    assert not due(50) and not due(51)
    run.t_saved = now - sampler.CHECKPOINT_MIN_S - 1 # a minute later
    assert due(50) and not due(51)
    run.save_s = 10.0 # a large chain: 200 s between saves
    assert not due(50)
    run.t_saved = now - 201
    assert due(100)


def test_reset_does_not_collapse_the_ensemble(tmp_path, monkeypatch):
    '''
    Checks the stuck-walker reset when most walkers look stuck: copying the few others onto them would collapse the
    ensemble into copies of a few points, so it replaces none and says why in run.log.
    '''
    def most_stuck(acceptance, lp): # 12 of the 16 walkers
        return np.arange(len(lp)) < 12
    monkeypatch.setattr(sampler, 'stuck_walkers', most_stuck)
    cfg = toy_cfg(tmp_path, 'sampler.reset_stuck_at=10', 'sampler.max_steps=20')
    run_dir = sampler.run_mcmc(cfg, workers = 2, make_plots = False)
    log = open(os.path.join(run_dir, 'run.log')).read()
    assert '12 of 16 walkers look stuck' in log and 'would collapse the ensemble' in log
    run = sampler.load_run(run_dir)
    assert run['extra']['replaced'].tolist() == [] and run['summary']['reset_step'] == 10


def test_run_log_shows_the_summary_warnings(tmp_path):
    '''
    Checks that the summary's own warnings (here: not converged, and only 10 steps) appear in run.log too, not only
    the config's.
    '''
    run_dir = sampler.run_mcmc(toy_cfg(tmp_path, 'sampler.max_steps=10'), workers = 2, make_plots = False)
    log = open(os.path.join(run_dir, 'run.log')).read()
    assert 'warning: not converged' in log and 'warning: only 10 steps' in log
