'''
Command line tool of isofate.mcmc: python -m isofate.mcmc COMMAND ... (see --help, and README.md in this folder).

The commands, in the usual order: template, check, scan, resolution, run, then summary, plot, tracks and compare
to look at the results; synth writes synthetic data for injection-recovery tests and forecasts. run, resolution,
scan and tracks evaluate the model in a pool of spawned worker processes (pool.py), so this process never imports
jax, atmodeller or isofate.isofate_coupler for them and stays at about 0.1 GB. check and synth run the model here,
in one process. Each command imports the modules it needs when it runs, so --help stays quick.
'''

import argparse
import datetime
import json
import math
import os
import signal
import sys
import time

from isofate.mcmc import env

# Text of --help: the overview and the examples, printed as written (RawDescriptionHelpFormatter)
DESCRIPTION = '''\
Config-driven emcee fits of coupled IsoFATE-Atmodeller evolutionary tracks to the observables of one planet.

The usual order:
  1. template   write the commented config template, then edit the planet, the priors and the data
  2. check      one model evaluation in this process: inputs, model vs data, warnings, the cost of an MCMC;
                --bisect also shows whether the bisection initializer finds the data along one parameter, and in
                solve mode ([solve] in the config) it prints one solve
  3. scan       the model on a 1-D or 2-D grid of free parameters: where does the posterior live?
  4. resolution choose model.n_steps and model.n_atmodeller: the cheapest converged resolution
  5. run        the MCMC (checkpointed; Ctrl-C and `run --resume RUN_DIR` continue it bit-identically)
  6. summary / plot / tracks / compare   look at the posterior, its evolutionary histories, and compare runs

Every command that reads a config takes --set section.key=value (TOML values) to override any option.
'''

EXAMPLES = '''\
examples (C=lhs1140b.toml, a copy of the example config isofate/mcmc/configs/lhs1140b.toml):
  python -m isofate.mcmc template my_planet.toml
  python -m isofate.mcmc check $C
  python -m isofate.mcmc check $C --theta log_f_atm=-3 --theta age=7.6e9 --set model.n_steps=3e4
  python -m isofate.mcmc check $C --bisect
  python -m isofate.mcmc check $C --theta u_outflow_H_fraction=-3.1     (solve mode: u in place of log_f_atm)
  python -m isofate.mcmc scan $C --x log_f_atm=-3.1:-2.9:81 --workers 6
  python -m isofate.mcmc scan $C --x log_f_atm=-3.25:-2.85:41 --y age=3.5e9:1e10:14 --obs outflow_H_fraction,Rp
  python -m isofate.mcmc synth $C --theta log_f_atm=-2.8 --out lhs1140b_synth.toml --sigma-scale 0.5
  python -m isofate.mcmc resolution $C --grid quick --dry-run
  python -m isofate.mcmc resolution $C --grid quick --points 4
  python -m isofate.mcmc resolution $C --points 3 --n-steps 3e4,1e5,3e5 --n-atmodeller 300,100,30   (solve mode)
  python -m isofate.mcmc run $C --set model.n_steps=30000 --set model.n_atmodeller=100
  python -m isofate.mcmc run --resume RUN_DIR --max-steps 10000
  python -m isofate.mcmc summary RUN_DIR
  python -m isofate.mcmc tracks RUN_DIR --n 100
  python -m isofate.mcmc plot RUN_DIR        (or a resolution_* or scan_* directory)
  python -m isofate.mcmc compare RUN_A RUN_B --labels dynamic,static
'''


class UsageError(Exception):
    '''
    Wrong command line arguments: main prints the message without a traceback and returns exit code 2
    '''


###_____Shared helpers_____###

def _split_theta(items):
    '''
    Splits the --theta items of check into parameter values and the solve-mode coordinate u_<observable>
    An item u_NAME without '=' stays with the parameters, so that _theta_overrides reports it.

    Inputs:
        - items: --theta strings 'name=value'; None: none [list of str]

    Output: (params, u)
        - params: the items that set parameters, unchanged [list of str]
        - u: (name, value) of a u_<observable>=value item (the last one wins), or None [tuple]
    '''
    params, u = [], None
    for item in items or []:
        name, sep, value = item.partition('=')
        if name.strip().startswith('u_') and sep:
            try:
                u = (name.strip(), float(value))
            except ValueError:
                raise UsageError(f'--theta {item!r}: {value!r} is not a number') from None
        else:
            params.append(item)
    return params, u


def _theta_overrides(items):
    '''
    Converts --theta name=value pairs into --set overrides of params.<name>.value
    That value is the point check and synth evaluate, and in scan the value of a parameter off the grid. For a
    free parameter it must lie inside the prior bounds, which loading the config checks. Raises UsageError for an
    item that is not name=value or whose value is not a number.

    Inputs:
        - items: --theta strings 'name=value'; None: none [list of str]

    Output: overrides 'params.<name>.value=<value>' [list of str]
    '''
    out = []
    for item in items or []:
        name, sep, value = item.partition('=')
        if not sep or not name.strip():
            raise UsageError(f'--theta {item!r} must look like name=value, e.g. --theta log_f_atm=-2.8')
        try:
            float(value)
        except ValueError:
            raise UsageError(f'--theta {item!r}: {value!r} is not a number') from None
        out.append(f'params.{name.strip()}.value={value.strip()}')
    return out


def _bisect_overrides(args):
    '''
    Converts --bisect-observable and --bisect-param into --set overrides of the [sampler] table

    Inputs:
        - args: parsed arguments; a command without these options gives no overrides [argparse.Namespace]

    Output: overrides 'sampler.bisect_observable="NAME"' and 'sampler.bisect_param="NAME"' [list of str]
    '''
    out = []
    for key in ('bisect_observable', 'bisect_param'):
        value = getattr(args, key, None)
        if value:
            out.append(f'sampler.{key}={json.dumps(value)}') # JSON quotes make a valid TOML string
    return out


def _load(path, overrides):
    '''
    Loads and validates the config file at path with its --set overrides; UsageError when the file is missing
    '''
    from isofate.mcmc import config
    if not os.path.isfile(path):
        raise UsageError(f'config file {path} not found')
    return config.load_config(path, overrides)


def _numbers(text):
    '''
    argparse type of the list options: '1e4,3e4,1e5' -> [1e4, 3e4, 1e5]; anything else is a usage error
    '''
    try:
        return [float(x) for x in text.split(',') if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f'{text!r} is not a comma-separated list of numbers') from None


def _print_warnings(warnings):
    '''
    Prints the config warnings, one per line, or "Warnings: none"
    '''
    print('Warnings:' + ('' if warnings else ' none'))
    for w in warnings:
        print(f'  - {w}')


def _evaluate_here(cfg, theta, twice = True):
    '''
    Evaluates the model in this process (check, synth)
    Sets the single-thread env and the compile caches (isocalc backend with parallel.cache), then evaluates once
    or twice. The first call compiles the model; the second, made only when the first gave status 0 or 3, gives
    the runtime without compilation.

    Inputs:
        - cfg: resolved config [dict]
        - theta: physical free parameters, in cfg['free'] order [array]
        - twice: evaluate a second time, to time the model without compilation [Bool]

    Output: (res, first)
        - res: model.evaluate result of the last evaluation [dict]
        - first: wall time of the first call, compilation included [s]
    '''
    from isofate.mcmc import model
    env.set_single_thread_env()
    if cfg['parallel']['cache'] and cfg['model']['backend'] == 'isocalc':
        try:
            env.enable_compile_caches(cfg)
        except Exception as e: # not fatal: the model then compiles from scratch
            print(f'warning: compile caches not enabled ({type(e).__name__}: {e})')
    start = time.perf_counter()
    res = model.evaluate(theta, cfg)
    first = time.perf_counter() - start # [s]
    if twice and res['status'] in (model.OK, model.NO_LIKELIHOOD):
        res = model.evaluate(theta, cfg)
    return res, first


###_____template_____###

OUTPUT_ROOT_LINE = '# output_root = ".."' # the line of template.toml that cmd_template uncomments


def cmd_template(args):
    '''
    Runs the template command: prints the commented config template, or writes it to OUT
    An existing OUT is only overwritten with --force. Written into a folder named configs/ (a results folder's),
    the copy gets output_root = ".." uncommented, so its runs go to the folder above configs/ instead of
    ./mcmc_results in the working directory.

    Inputs:
        - args: parsed arguments (out, None to print; force) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import config
    with open(config.TEMPLATE) as f:
        text = f.read()
    if args.out is None:
        sys.stdout.write(text)
        return 0
    if os.path.exists(args.out) and not args.force:
        raise UsageError(f'{args.out} exists; use --force to overwrite it')
    folder = os.path.basename(os.path.dirname(os.path.abspath(args.out)))
    note = ''
    if folder == 'configs' and OUTPUT_ROOT_LINE in text: # a results folder's configs/: its runs go next to it
        # Drop the leading '# ' and pad 2 spaces, so the comment after the value keeps its column
        text = text.replace(OUTPUT_ROOT_LINE, OUTPUT_ROOT_LINE[2:] + '  ', 1)
        note = ' with output_root = ".." (runs go to the folder above configs/)'
    with open(args.out, 'w') as f:
        f.write(text)
    print(f'wrote {args.out}{note}; edit it, then: python -m isofate.mcmc check {args.out}')
    if not note:
        print('  (set output_root in it: without it results go to ./mcmc_results in the working directory)')
    return 0


###_____check_____###

def _setup_text(cfg, params):
    '''
    The resolved inputs check prints first: the star, the orbit and insolation, F0, the bare-core radius and the
    model settings

    Inputs:
        - cfg: resolved config, in SI units [dict]
        - params: full parameter set at the check point (params.theta_to_params) [dict]

    Output: text lines [list of str]
    '''
    from isofate.constants import Ls, Me, Ms, Re, Rs, au2m, s2day
    from isofate.isofunks import R_core
    from isofate.orbit_params import EqTemp
    # cfg holds SI values: printed in solar units, days, AU and Earth radii
    star, planet, m = cfg['star'], cfg['planet'], cfg['model']
    Fp = planet['Fp'] # bolometric insolation [W/m2]
    lines = [f"{cfg['name']}: {cfg['description']}" if cfg['description'] else cfg['name'],
             f"  config {cfg['config_path']}", f"  output_root {cfg['output_root']}", '']
    star_text = f"  star:   M = {star['M']/Ms:.4g} M_sun"
    if star['R'] is not None:
        star_text += f", R = {star['R']/Rs:.4g} R_sun"
    if star['Teff'] is not None:
        star_text += f", Teff = {star['Teff']:.4g} K"
    lines.append(star_text + f", L = {star['L']/Ls:.4g} L_sun")
    # No period when the config gives semi_major_axis alone
    period = f"P = {planet['P']*s2day:.5g} days, " if planet['P'] is not None else ''
    # Teq from the insolation and the Bond albedo (full redistribution); F0 = Fp*10**log_F0_ratio is the XUV flux
    # at t0, and R_core(Mp) the radius of the planet once its atmosphere is gone, (Mp/M_earth)**0.25 R_earth
    lines.append(f"  planet: {period}d = {planet['d']/au2m:.5g} AU, Fp = {Fp:.5g} W/m2, Teq = "
                 f"{float(EqTemp(Fp, A = params['albedo'])):.5g} K (albedo {params['albedo']:g})")
    lines.append(f"          F0 = {Fp*10**params['log_F0_ratio']:.4g} W/m2 (XUV at t0), bare-core radius "
                 f"{R_core(params['Mp']*Me)/Re:.4g} R_earth at Mp = {params['Mp']:.4g} M_earth")
    lines.append(f"  model:  backend {m['backend']}, n_steps = {m['n_steps']}, n_atmodeller = {m['n_atmodeller']}, "
                 f"composition {m['composition']}, mechanism {m['mechanism']}")
    lines.append(f"          flux_model {m['flux_model']}, step_fn {str(m['step_fn']).lower()}, dynamic_phi "
                 f"{str(m['dynamic_phi']).lower()}, t0 = {m['t0']:g} yr")
    return lines


def _params_text(cfg, params):
    '''
    The parameter table of check: value, unit and prior of every parameter, FREE marking the sampled ones

    Inputs:
        - cfg: resolved config [dict]
        - params: full parameter set at the check point [dict]

    Output: text lines [list of str]
    '''
    lines = [f"  {'parameter':<22} {'value':>11}  {'unit':<8} prior"]
    for name, p in cfg['params'].items():
        if p['prior'] == 'normal':
            prior = f"normal(mu = {p['mu']:g}, sigma = {p['sigma']:g})"
            prior += f" in [{p['bounds'][0]:g}, {p['bounds'][1]:g}]" if p['bounds'] else '' # optional truncation
        elif p['vary']:
            prior = f"{p['prior']} [{p['bounds'][0]:g}, {p['bounds'][1]:g}]"
        else:
            prior = 'fixed'
        value = 'unset' if params[name] is None else f'{params[name]:.5g}' # None: an optional one left off (Rp_fixed)
        lines.append(f"  {name:<22} {value:>11}  {p['unit'] or '-':<8} {prior}" + ('   FREE' if p['vary'] else ''))
    return lines


def _outputs_text(obs):
    '''
    Every model output of an evaluation (model.OBS_FIELDS: the observables and the extras), three per line

    Inputs:
        - obs: model outputs by name, in their units [dict]

    Output: text lines [list of str]
    '''
    from isofate.mcmc import model
    items = [f'{name} = {obs[name]:.5g}' for name in model.OBS_FIELDS]
    width = max(len(s) for s in items) + 2
    return ['  ' + ''.join(f'{s:<{width}}' for s in items[i:i + 3]).rstrip() for i in range(0, len(items), 3)]


def _cost_text(cfg, runtime, runs = 1):
    '''
    The MCMC cost estimate that check prints, scaled from the runtime of one model run
    Gives the time per step, of max_steps, of the initialization and of the tracks, and the memory, for the
    worker count and the walkers a run would get. Every evaluation is assumed to cost as much as this one.

    Inputs:
        - cfg: resolved config [dict]
        - runtime: isocalc time of one model run [s]
        - runs: model runs per evaluation; in solve mode those of one cold solve, from both prior bounds [int]

    Output: text lines [list of str]
    '''
    from isofate.mcmc import config
    from isofate.mcmc.pool import choose_workers, est_worker_gb
    from isofate.mcmc.sampler import duration_text
    m, s = cfg['model'], cfg['sampler']
    workers, _ = choose_workers(None, m['n_steps'], cfg) # the count a run would get (it says so if memory cuts it)
    lines = [f"MCMC cost estimate (from this evaluation, {runtime:.3g} s per model run):"]
    gb = est_worker_gb(m['n_steps'], m['backend']) # [GB]
    if not cfg['free']:
        return lines + ['  no free parameters: nothing to sample']
    nwalkers = config.choose_nwalkers(cfg, workers)
    per_step = runtime*runs*nwalkers/workers # [s] every walker evaluated once per step
    lines.append(f"  {nwalkers} walkers on {workers} workers: about {per_step:.3g} s/step ({3600/per_step:.0f} "
                 'steps/h) if every evaluation costs as much as this one'
                 + (f' ({runs} model runs per solve from both prior bounds; warm starts during sampling need about a '
                    'third of that once the table is refitted to the chain)' if runs > 1 else ''))
    lines.append('  (stripped tracks are faster, O-rich ones slower; measured runs took 1.2-1.7x this estimate)')
    # Model runs of the initialization: one batch of draws (the prescan: all of its points)
    if s['init'] == 'solve':
        n_init = 3*nwalkers*runs # 3*nwalkers draws, each a cold solve
        init = f"the solve-mode initialization (about {n_init} model runs per batch of {3*nwalkers} draws)"
    elif s['init'] == 'bisect':
        draws = 3*nwalkers if s['bisect_draws'] == 'auto' else s['bisect_draws']
        n_init = draws*(2 + s['bisect_iters']) # each draw: both prior bounds, then at most bisect_iters rounds
        init = f"the bisection (at most {n_init} evaluations per batch of {draws} draws)"
    else:
        n_init = s['prescan_factor']*nwalkers # one per Latin-hypercube point ("prior" and "ball" land here too)
        init = f"the prescan ({n_init} evaluations)"
    lines.append(f"  max_steps = {s['max_steps']}: about {duration_text(per_step*s['max_steps'])}; {init} about "
                 f"{duration_text(runtime*n_init/workers)}; {s['tracks']} tracks about "
                 f"{duration_text(runtime*s['tracks']/workers)}")
    lines.append(f"  memory: about {gb:.2f} GB per worker, {gb*workers:.1f} GB for {workers} workers "
                 f"(parallel.max_memory_gb = {cfg['parallel']['max_memory_gb']:g})")
    return lines


def _bisect_here(cfg, theta):
    '''
    Runs check --bisect: one bisection from theta, in this process, printed round by round
    The bisection is the root search of init = "bisect": it moves sampler.bisect_param between its prior bounds,
    the other parameters fixed, until sampler.bisect_observable reaches its data value (to bisect_tol sigma, in
    the likelihood space). It shows whether the initializer finds the data along that parameter from here.

    Inputs:
        - cfg: resolved config [dict]
        - theta: physical free parameters to start from, in cfg['free'] order [array]

    Output: True when the bisection converged [Bool]
    '''
    from isofate.mcmc import likelihood, model, sampler, search
    st = search.settings(cfg) # the parameter and its prior bracket, the observable and its target
    o = st['ocfg']
    print(f"Bisection of {st['param']} in [{st['lo']:.6g}, {st['hi']:.6g}] on {st['observable']} "
          f"({o['space']} space, {o['kind']} at {o['value']:.5g}); the other parameters stay at "
          + (', '.join(f'{n} = {x:.5g}' for n, x in zip(cfg['free'], theta) if n != st['param']) or 'their values')
          + ':', flush = True)
    result = search.run_bisection(cfg, [theta], model.LocalPool(cfg), st) # one draw; LocalPool evaluates here
    result['settings'] = st # read by iteration_table
    print('\n'.join(search.iteration_table(result, 0, cfg)))
    outcome = int(result['outcome'][0])
    done = f"{result['rounds'][0]} rounds, {result['evaluations']} evaluations, {result['seconds']:.1f} s"
    if outcome == search.CONVERGED:
        blob = result['blobs'][0]
        obs = {f: float(blob[f]) for f in model.OBS_FIELDS} # every observable at the converged point
        ll, terms = likelihood.log_like(obs, cfg)
        print(f"converged: {st['param']} = {result['x'][0]:.7g} ({done}), lp = {result['lp'][0]:.6g}; "
              'the observables there:')
        print(sampler.comparison_text(obs, terms, cfg))
        return True
    print(f"no root: {search.OUTCOMES[outcome]} ({done})")
    if outcome in (search.ABOVE, search.BELOW, search.UNDEFINED): # same sign, or undefined, at both prior bounds
        hint = search.no_root_hint(st['observable'], outcome, st['param'])
        print(f"  the observable does not change sign across the prior here{hint}: another bisect_observable or "
              'bisect_param, other values of the other parameters (--theta), or wider bounds may help')
    return False


def _solve_here(cfg, phys_theta, u = None):
    '''
    Runs the solve-mode part of check: one solve at the check point, in this process, printed run by run
    Solve mode samples u, the model value of the solve observable in its likelihood space (log10 for a log10
    observable), in place of solve.param. Each evaluation finds the value P* inside the prior of that parameter
    with T(model at P*) = u, and the density gains the Jacobian |dP*/du| = 1/|dT/dP| of that change of variables,
    so the posterior over the physical parameters is unchanged (see "Solve mode" in the README). Here the solve
    starts cold, from both prior bounds, as at the initialization (no warm-start table).

    Inputs:
        - cfg: resolved config [dict]
        - phys_theta: physical free parameters at the check point; all but solve.param stay fixed [array]
        - u: the sampled coordinate (--theta u_<observable>=...); None: the datum of the solve observable [float]

    Output: (res, converged)
        - res: solve.evaluate result, with the history of the root search [dict]
        - converged: True when the solve gave status 0 [Bool]
    '''
    from isofate.mcmc import likelihood, model, sampler, search, solve
    st = solve.settings(cfg) # the solved parameter and its prior bracket, the observable, the u support, tol
    o = st['ocfg']
    if u is None:
        u = likelihood.observed(o) # the datum, in the likelihood space
    lo, hi = st['bounds'] # support of u: the datum -/+ range_sigma sigma
    theta = solve.physical(phys_theta, u, cfg) # the sampler's theta: u in the slot of the solved parameter
    others = ', '.join(f'{n} = {x:.5g}' for n, x in zip(cfg['free'], phys_theta) if n != st['param'])
    print(f"Solve mode: {st['param']} in [{st['lo']:.6g}, {st['hi']:.6g}] solved for {st['name']} = {u:.6g} "
          f"({o['space']} space of {st['observable']}; u support [{lo:.5g}, {hi:.5g}], {cfg['solve']['range_sigma']:g} "
          f"sigma), the other parameters at {others or 'their values'}; from both prior bounds (no warm-start table):",
          flush = True)
    start = time.perf_counter()
    res = solve.evaluate(theta, cfg, history = True) # no table: a cold solve
    seconds = time.perf_counter() - start # [s]
    if res.get('history'):
        print('\n'.join(search.history_table(res['history'], st, st['observable'], u, st['tol'])))
        # The same T at several P inside the u support: T cannot be inverted there. The plateau branch covers a
        # plateau of stripped planets at a prior bound (solve.plateau_covered); any other gets no mass.
        flat = solve.plateau(res['history'], st, u)
        if flat:
            T, a, b, term = flat
            if solve.plateau_covered(res['history'], st, u, flat):
                what = (f"the plateau branch (solve.plateau) samples that stretch of {st['param']} through the u "
                        'that have no root')
            elif st['plateau']: # e.g. x = 1 of planets that keep an atmosphere
                what = (f"it is not one of planets stripped at a bound of {st['param']}'s prior, which the plateau "
                        f"branch covers, so solve mode leaves out the posterior mass of that stretch of {st['param']}; "
                        'if the term is not far below the best fit, sample without solve mode')
            else:
                what = (f"with solve.plateau = false solve mode leaves out the posterior mass of that stretch of "
                        f"{st['param']}; if the term is not far below the best fit, set solve.plateau = true or "
                        'sample without solve mode')
            print(f"  note: a plateau: {st['observable']} gives T = {T:.6g} at every {st['param']} evaluated from "
                  f"{a:.6g} to {b:.6g} (e.g. the bare-core radius of a stripped planet), inside the u support, where "
                  f"its term is {term:.3g}. T cannot be inverted there: {what} (see \"Solve mode\" in the README)")
        if 'plateau' in res: # this u has no root and a bound is stripped: the plateau branch's point
            pl, P_m = res['plateau'], res['plateau']['P']
            stretches = ' and '.join(f'{min(a, b):.6g} to {max(a, b):.6g}' for a, b in pl['stretches'] if a != b)
            if res['status'] in (model.OK, model.NO_LIKELIHOOD):
                fate = 'stripped, on the plateau, so it counts'
            elif res['status'] == model.OUTSIDE_PRIOR:
                fate = 'not on the plateau, so it has a root of its own and counts there'
            else:
                fate = 'where the model run failed (below)'
            print(f"  plateau branch: no root; the u without a root ({stretches}) map linearly onto the prior of "
                  f"{st['param']} (|d{st['param']}/du| = {pl['jac']:.5g}), the plateau at T = {pl['T']:.6g}: "
                  f"{st['param']} = {P_m:.6g}, {fate}")
    status = int(res['status'])
    done = f"{res['solve_evals']} model runs, {seconds:.1f} s, isocalc {res['runtime']:.3g} s"
    if status in (model.OK, model.NO_LIKELIHOOD) and math.isfinite(res['solve_value']):
        # log_jac = ln|dP*/du| = -ln|dT/dP|, so |dT/dP| = exp(-log_jac); the plateau branch's is its map's
        slope = ('the plateau branch\'s map' if 'plateau' in res else
                 f"dT/d{st['param']} = {math.exp(-res['log_jac']) if math.isfinite(res['log_jac']) else math.nan:.5g}")
        print(f"solved: {st['param']}* = {res['solve_value']:.7g}, ln|d{st['param']}*/du| = {res['log_jac']:.6g} "
              f"({slope}), {done}; lp = log_prior + log_jac + log_like = {res['log_prior']:.5g} + "
              f"{res['log_jac']:.5g} + {res['log_like']:.5g} = {res['lp']:.6g}"
              + (f"; {res['error']}" if res['error'] else ''))
        print(sampler.comparison_text(res['obs'], res['terms'], cfg))
        return res, status == model.OK # status 3: a root, but lp = -inf (the data, or an unusable derivative)
    print(f"no solve: status {status} ({model.STATUS_NAMES.get(status, '?')}): {res['error']} ({done})")
    return res, False


def cmd_check(args):
    '''
    Runs the check command: one model evaluation at the check point, in this process
    Loads CONFIG with its --set and --theta overrides and prints the resolved inputs and the parameters. Then it
    evaluates the model twice (the second time without compilation) and prints each active observable against its
    data with its log-likelihood term, every model output, the warnings and the cost of an MCMC. With --bisect it
    also runs one bisection from that point, and in solve mode one solve.

    Inputs:
        - args: parsed arguments (config, theta, set, bisect, bisect_observable, bisect_param) [argparse.Namespace]

    Output: exit code: 0 when the model ran (status 0 or 3) and the bisection and the solve converged, else 1 [int]
    '''
    from isofate.mcmc import config, model, sampler, search
    from isofate.mcmc import params as prm
    items, u = _split_theta(args.theta) # u: --theta u_<observable>=value, the solve-mode coordinate
    cfg = _load(args.config, args.set + _theta_overrides(items) + _bisect_overrides(args))
    if args.bisect: # a bad bisection target fails before any model run
        search.settings(cfg)
    if u is not None:
        if not config.solve_on(cfg):
            raise UsageError(f'--theta {u[0]}=... needs solve mode ([solve] param set in the config)')
        if u[0] != cfg['solve']['name']:
            raise UsageError(f"--theta {u[0]}: the sampled coordinate of this config is {cfg['solve']['name']}")
    # The check point: the params' values (or --theta), physical also in solve mode
    theta = prm.centre_theta(cfg)
    params = prm.theta_to_params(theta, cfg)
    print('\n'.join(_setup_text(cfg, params)) + '\n')
    print('\n'.join(_params_text(cfg, params)) + '\n')
    print('Active observables: ' + (', '.join(cfg['observables']) or 'none') + '\n')

    print(f"Evaluating at {', '.join(f'{n} = {x:.6g}' for n, x in zip(cfg['free'], theta)) or 'the fixed values'} "
          f"(first call compiles the model) ...", flush = True)
    res, first = _evaluate_here(cfg, theta)
    status = int(res['status'])
    print(f"  status {status} ({model.STATUS_NAMES[status]})" + (f": {res['error']}" if res['error'] else ''))
    print(f"  isocalc {res['runtime']:.3g} s per evaluation after compilation (first call {first:.3g} s)")
    if status in (model.OK, model.NO_LIKELIHOOD) or res['terms']:
        print(sampler.comparison_text(res['obs'], res['terms'], cfg))
    print(f"  log_like = {res['log_like']:.6g}, log_prior = {res['log_prior']:.6g}, lp = {res['lp']:.6g}")
    if status in (model.OK, model.NO_LIKELIHOOD):
        print('All model outputs at the end of the run:')
        print('\n'.join(_outputs_text(res['obs'])))
    print()
    converged = True # the exit code also needs the bisection and the solve, if any, to converge
    if args.bisect:
        converged = _bisect_here(cfg, theta)
        print()
    runs = 1 # model runs per evaluation, for the cost estimate
    if config.solve_on(cfg):
        solved, ok = _solve_here(cfg, theta, None if u is None else u[1])
        converged = converged and ok
        runs = max(int(solved['solve_evals']), 1) # a cold solve, the derivative included
        print()
    _print_warnings(cfg['warnings'])
    for note in cfg.get('notes', []): # information, not warnings, e.g. init = "bisect" turned into "solve"
        print(f'  note: {note}')
    print()
    if status in (model.OK, model.NO_LIKELIHOOD) and res['runtime'] > 0: # runtime 0: isocalc did not run
        print('\n'.join(_cost_text(cfg, res['runtime'], runs)))
    return 0 if status in (model.OK, model.NO_LIKELIHOOD) and converged else 1


###_____synth_____###

def cmd_synth(args):
    '''
    Runs the synth command: writes a copy of CONFIG whose data are the model prediction at theta
    Each active observable gets the model value at theta (the params' values, or --theta) as its value and keeps
    its sigma (times --sigma-scale); the input observables (Mp, age, Teq) take the parameter values. In the copy
    the params' values are theta, the name is the stem of OUT (so its runs get their own folder) and output_root
    is absolute. Use it for injection-recovery tests and forecasts. The file is loaded again to validate it, and
    the log-likelihood of the injected point is printed: 0 for detections, log(1/2) per limit, since the model then
    sits exactly at the limit.

    Inputs:
        - args: parsed arguments (config, theta, set, out, sigma_scale, force) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import config, likelihood, model
    from isofate.mcmc import params as prm
    if os.path.exists(args.out) and not args.force:
        raise UsageError(f'{args.out} exists; use --force to overwrite it')
    if not args.sigma_scale > 0: # also catches NaN
        raise UsageError(f'--sigma-scale must be > 0, got {args.sigma_scale:g}')
    overrides = args.set + _theta_overrides(args.theta)
    cfg = _load(args.config, overrides)
    raw = config.read_toml(args.config, overrides) # the raw TOML, the same overrides applied: the copy's skeleton
    if not cfg['observables']:
        raise config.ConfigError('no active observables to synthesize: switch some on (use = true)')
    theta = prm.centre_theta(cfg) # the injected point
    res, _ = _evaluate_here(cfg, theta, twice = False) # one run: no timing needed
    if res['status'] not in (model.OK, model.NO_LIKELIHOOD):
        raise RuntimeError(f"the model did not run at this theta: status {int(res['status'])} "
                           f"({model.STATUS_NAMES[int(res['status'])]}){': ' + res['error'] if res['error'] else ''}")
    bad = []
    for name, o in cfg['observables'].items():
        x = likelihood.to_space(res['obs'][name], o) # the model as the likelihood sees it (floors applied)
        if not math.isfinite(x):
            bad.append(f"{name} = {res['obs'][name]:g}")
            continue
        entry = raw['observables'][name]
        entry['value'] = 10**x if o['space'] == 'log10' else x # value is always in the physical unit
        if args.sigma_scale != 1:
            s = o['sigma'] # a number, or (lo, hi) for an asymmetric error
            entry['sigma'] = [args.sigma_scale*v for v in s] if isinstance(s, tuple) else args.sigma_scale*s
    if bad:
        raise config.ConfigError(f"the model gives no finite value for {', '.join(bad)} at this theta (e.g. a "
                                 'stripped atmosphere has no composition); switch them off with --set '
                                 'observables.NAME.use=false or choose another --theta')
    stem = os.path.splitext(os.path.basename(args.out))[0]
    raw['name'] = stem
    raw['output_root'] = cfg['output_root'] # absolute: the file may live elsewhere than the original
    point = ', '.join(f'{n} = {x:.8g}' for n, x in zip(cfg['free'], theta)) or 'the fixed values'
    raw['description'] = f"synthetic data from {cfg['name']} at {point}"
    # Where the file came from, as TOML comments at its top
    header = (f"# Synthetic data written by `python -m isofate.mcmc synth` on "
              f"{datetime.datetime.now():%Y-%m-%d %H:%M}\n# from {cfg['config_path']}\n"
              + ''.join(f"# {'with' if i == 0 else '    '} --set {o}\n" for i, o in enumerate(args.set))
              + f"# at {point} (the params' values below are this point)\n"
              f"# observable values = the model prediction there; sigma x {args.sigma_scale:g}\n\n")
    with open(args.out, 'w') as f:
        f.write(header + config.dump_toml(raw))
    new = config.load_config(args.out) # validates what was written
    ll, terms = likelihood.log_like(res['obs'], new) # the injected point against its own synthetic data
    print(f'wrote {args.out}: {len(new["observables"])} observables at the model prediction, {point}')
    for name, o in new['observables'].items():
        print(f"  {name:<14} {o['value']:<14.6g} sigma {o['sigma']}{' dex' if o['space'] == 'log10' else ''}"
              f"  ({o['kind']}, {o['space']})  term {terms[name] + 0.0:.3g}") # + 0.0: no '-0'
    print(f'  log_like at the injected point = {ll:.4g} (0 for detections; log(1/2) per limit)')
    return 0


###_____resolution, run, tracks_____###

def cmd_resolution(args):
    '''
    Runs the resolution command: the study that chooses model.n_steps and model.n_atmodeller (resolution.py)
    Every test point runs at every cell of the grid and is compared with the finest cell, in units of each
    observable's sigma; the recommendation is the cheapest cell that stays converged. The grid is --grid quick or
    default, or the lists of --n-steps and --n-atmodeller, with the missing list taken from the chosen grid.

    Inputs:
        - args: parsed arguments (config, set, grid, n_steps, n_atmodeller, points, source, workers, tol, out,
          dry_run, solve, bisect_observable, bisect_param) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import config, resolution
    cfg = _load(args.config, args.set + _bisect_overrides(args))
    solve_mode = args.solve # None (neither --solve nor --no-solve): solve mode when [solve] is active
    if _bisect_overrides(args) and config.resolution_source(cfg, args.source, solve_mode) != 'bisect':
        raise UsageError('--bisect-observable and --bisect-param apply to test points of source = "bisect" only (this '
                         f'study: {config.resolution_source(cfg, args.source, solve_mode)!r}); add --source bisect'
                         + (' --no-solve' if config.solve_on(cfg) else ''))
    grid = args.grid
    if args.n_steps or args.n_atmodeller: # explicit lists: a dict grid
        base = resolution.QUICK_GRID if args.grid == 'quick' else cfg['resolution']
        grid = {'n_steps': args.n_steps or base['n_steps']}
        if args.n_atmodeller:
            grid['n_atmodeller'] = args.n_atmodeller
        # A missing n_atmodeller comes from [resolution], or is [0] for an uncoupled config (n_atmodeller = 0),
        # so the quick grid's list is only filled in for a coupled one
        elif args.grid == 'quick' and cfg['model']['n_atmodeller'] > 0:
            grid['n_atmodeller'] = base['n_atmodeller']
    resolution.run_resolution(cfg, grid = grid, points = args.points, source = args.source,
                              workers = args.workers, tol = args.tol, out_dir = args.out, dry_run = args.dry_run,
                              solve_mode = solve_mode)
    return 0


def cmd_run(args):
    '''
    Runs the run command: a new MCMC of CONFIG, or with --resume RUN_DIR the continuation of a run
    A resume continues from the last checkpoint with the run's own config.json and gives the same chain, bit for
    bit, as an uninterrupted run; only --workers, --max-steps and --no-plots may differ from the first start.

    Inputs:
        - args: parsed arguments (config, resume, set, workers, nwalkers, max_steps, no_plots, bisect_observable,
          bisect_param) [argparse.Namespace]

    Output: exit code 0; an interrupt or a failure raises, after run_mcmc has saved the chain [int]
    '''
    from isofate.mcmc import sampler
    if args.resume:
        if args.config or args.set or args.nwalkers or _bisect_overrides(args):
            raise UsageError('run --resume RUN_DIR continues a run with its own config.json: CONFIG, --set, '
                             '--nwalkers and --bisect-* cannot change it (start a new run instead)')
        if not os.path.isdir(args.resume):
            raise UsageError(f'run directory {args.resume} not found')
        sampler.run_mcmc(None, workers = args.workers, run_dir = args.resume, resume = True,
                         max_steps = args.max_steps, make_plots = not args.no_plots)
        return 0
    if not args.config:
        raise UsageError('run needs a CONFIG file, or --resume RUN_DIR')
    # --nwalkers N is --set sampler.nwalkers=N
    overrides = args.set + ([f'sampler.nwalkers={args.nwalkers}'] if args.nwalkers else []) + _bisect_overrides(args)
    cfg = _load(args.config, overrides)
    if _bisect_overrides(args) and cfg['sampler']['init'] != 'bisect': # they would change nothing
        raise UsageError('--bisect-observable and --bisect-param apply to sampler.init = "bisect" only (this run: '
                         f"{cfg['sampler']['init']!r}" + ('; in solve mode the observable is solve.observable'
                                                           if cfg['sampler']['init'] == 'solve' else
                                                           '; add --set sampler.init=bisect') + ')')
    sampler.run_mcmc(cfg, workers = args.workers, max_steps = args.max_steps, make_plots = not args.no_plots)
    return 0


def cmd_tracks(args):
    '''
    Runs the tracks command: re-runs the MAP and n posterior samples of a run as evolutionary tracks
    The samples run with save_molecules through a worker pool; their time series go to RUN_DIR/tracks.npz, then
    the run's figures are redrawn (unless --no-plots).

    Inputs:
        - args: parsed arguments (run_dir, n, workers, seed, no_plots) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import plots, tracks
    path = tracks.compute_tracks(args.run_dir, n = args.n, workers = args.workers, seed = args.seed)
    print(f'wrote {path}')
    if not args.no_plots:
        written = plots.plot_run(args.run_dir)
        print('plots: ' + ', '.join(os.path.basename(p) for p in written))
    return 0


###_____scan_____###

def cmd_scan(args):
    '''
    Runs the scan command: the model on a 1-D (--x) or 2-D (--x, --y) grid of free parameters
    The other parameters stay at their values (or --theta). The grid runs through a worker pool and goes to
    <output_root>/<name>/scan_<stamp>/ with scan.npz and scan.png: a look at where the posterior lives before
    an MCMC.

    Inputs:
        - args: parsed arguments (config, x, y, obs, theta, workers, out, set) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import scan
    cfg = _load(args.config, args.set + _theta_overrides(args.theta))
    x = scan.parse_axis(args.x, cfg) # 'name=lo:hi:n' -> (name, n values inside the prior)
    y = scan.parse_axis(args.y, cfg) if args.y else None
    scan.run_scan(cfg, x, y, observables = args.obs, workers = args.workers, out_dir = args.out)
    return 0


###_____compare, plot, summary_____###

def cmd_compare(args):
    '''
    Runs the compare command: overlays the posteriors of two or more MCMC runs and prints compare.txt
    The figures and compare.txt go to --out, by default compare_<stamp>/ under the first run's
    <output_root>/<name>/.

    Inputs:
        - args: parsed arguments (runs, labels 'A,B' with one label per run, out) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import plots, sampler
    if len(args.runs) < 2:
        raise UsageError('compare needs at least two run directories')
    for run_dir in args.runs:
        if not os.path.exists(os.path.join(run_dir, sampler.CHAIN)):
            raise UsageError(f'{run_dir} has no {sampler.CHAIN}: not an MCMC run directory (or no checkpoint yet)')
    labels = [x.strip() for x in args.labels.split(',')] if args.labels else None
    if labels is not None and len(labels) != len(args.runs):
        raise UsageError(f'--labels gives {len(labels)} labels for {len(args.runs)} runs')
    out_dir, paths = plots.compare_runs(args.runs, labels, args.out)
    with open(os.path.join(out_dir, 'compare.txt')) as f:
        print(f.read())
    print('wrote ' + ', '.join(os.path.basename(p) for p in paths) + f' in {out_dir}')
    return 0


def cmd_plot(args):
    '''
    Runs the plot command: writes the figures of an MCMC run, a resolution study or a scan into its directory
    The kind of directory is told by its files: chain.npz, results.npz or scan.npz. Prints the paths written.

    Inputs:
        - args: parsed arguments (dir) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import plots
    written = plots.plot_dir(args.dir)
    print('\n'.join(written) if written else 'no figures apply to the files in this directory')
    return 0


def cmd_summary(args):
    '''
    Runs the summary command: prints RUN_DIR/summary.json readably (or as it is, with --json)
    The file is recomputed from the chain when it is missing, when it was written for another number of steps
    than chain.npz holds (e.g. while a resumed run is going), or with --update. A recomputed summary is "partial"
    unless the file it replaces described this very chain as finished or interrupted.

    Inputs:
        - args: parsed arguments (run_dir, update, json) [argparse.Namespace]

    Output: exit code 0 [int]
    '''
    from isofate.mcmc import config, sampler
    import numpy as np
    path = os.path.join(args.run_dir, sampler.SUMMARY)
    chain = os.path.join(args.run_dir, sampler.CHAIN)
    if not os.path.exists(chain):
        raise UsageError(f'{args.run_dir} has no {sampler.CHAIN}: not an MCMC run directory (or no checkpoint yet)')
    with np.load(chain) as d: # cheap: one scalar of the archive
        steps = int(d['iteration'])
    summary = None
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
    current = summary is not None and summary.get('steps') == steps # written for the chain as it is now
    if not current or args.update:
        # The run may be going (e.g. resumed): partial, unless the file is the finished state of this very chain
        state = summary['state'] if current and summary.get('state') in ('finished', 'interrupted') else 'partial'
        why = ('missing' if summary is None else f"written for {summary.get('steps')} steps, the chain has {steps}"
               if not current else 'update')
        print(f'computing {path} from the chain ({why}; state: {state})')
        summary = sampler.write_summary(args.run_dir, state = state)
    if args.json:
        print(json.dumps(summary, indent = 1))
        return 0
    cfg = config.load_resolved(os.path.join(args.run_dir, 'config.json')) # for the data of the observables
    print(sampler.summary_text(summary, cfg, full = True))
    return 0


###_____Argument parsing_____###

def _add_bisect(p):
    '''
    Adds --bisect-observable and --bisect-param (check, resolution, run) to the subparser p
    '''
    p.add_argument('--bisect-observable', metavar = 'NAME', help = 'the observable the bisection brings to its data '
                   '(same as --set sampler.bisect_observable=NAME; default Rp)')
    p.add_argument('--bisect-param', metavar = 'NAME', help = 'the free parameter it moves (same as --set '
                   'sampler.bisect_param=NAME; default log_f_atm)')


def _add_set(p):
    '''
    Adds the repeatable --set KEY=VALUE override of any config option to the subparser p
    '''
    p.add_argument('--set', action = 'append', default = [], metavar = 'KEY=VALUE',
                   help = 'override a config option, e.g. --set model.n_steps=3e4 --set params.age.vary=false '
                          '(TOML values; repeatable)')


def make_parser():
    '''
    Builds the argument parser: one subcommand per command, with its help, description and examples
    Each subcommand sets func, the cmd_* function that main calls with the parsed arguments.

    Output: parser [argparse.ArgumentParser]
    '''
    parser = argparse.ArgumentParser(prog = 'python -m isofate.mcmc', description = DESCRIPTION, epilog = EXAMPLES,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest = 'command', metavar = 'COMMAND') # no command: args.command is None
    fmt = argparse.RawDescriptionHelpFormatter # keeps the line breaks of the descriptions and examples

    p = sub.add_parser('template', help = 'write (or print) the commented config template', formatter_class = fmt,
                       description = 'The config template: every option with its default and units, commented.',
                       epilog = 'example: python -m isofate.mcmc template my_planet.toml')
    p.add_argument('out', nargs = '?', metavar = 'OUT', help = 'file to write (default: print it)')
    p.add_argument('--force', action = 'store_true', help = 'overwrite OUT')
    p.set_defaults(func = cmd_template)

    p = sub.add_parser('check', help = 'one evaluation in this process: inputs, model vs data, cost',
                       formatter_class = fmt, description = (
                           'Loads and validates CONFIG and runs the model in this process at the params\' values '
                           '(or --theta): prints the resolved star and planet, the parameters, each active '
                           "observable's model value against the data with its log-likelihood term, every model "
                           'output, the warnings, and the cost of an MCMC. The model runs twice so the time '
                           'excludes compilation.'),
                       epilog = 'examples:\n  python -m isofate.mcmc check lhs1140b.toml --theta log_f_atm=-3 '
                                '--theta age=7.6e9\n  python -m isofate.mcmc check lhs1140b.toml --bisect '
                                '--bisect-observable Rp')
    p.add_argument('config', metavar = 'CONFIG', help = 'TOML config')
    p.add_argument('--theta', action = 'append', default = [], metavar = 'NAME=VALUE',
                   help = 'evaluate here instead of at params.NAME.value (repeatable; same as --set '
                          'params.NAME.value=VALUE)')
    p.add_argument('--bisect', action = 'store_true',
                   help = 'also run one bisection (init = "bisect") from this point and print every round: does '
                          'the observable bracket its data along the parameter? (In solve mode check always prints '
                          'one solve; --theta u_<observable>=VALUE sets its u)')
    _add_bisect(p)
    _add_set(p)
    p.set_defaults(func = cmd_check)

    p = sub.add_parser('synth', help = 'synthetic data: the config with the model prediction at theta as data',
                       formatter_class = fmt, description = (
                           'Writes a copy of CONFIG whose active observables have the model prediction at theta '
                           '(the params\' values, or --theta) as their value, keeping sigma (times --sigma-scale). '
                           'Input observables (Mp, age, Teq) take the parameter values. The params\' values in the '
                           'copy are theta, its name is the stem of OUT and its output_root is absolute. Use it '
                           'for injection-recovery tests and forecasts.'),
                       epilog = 'example: python -m isofate.mcmc synth lhs1140b.toml --theta log_f_atm=-2.8 '
                                '--theta age=6e9 --out lhs1140b_inj.toml')
    p.add_argument('config', metavar = 'CONFIG', help = 'TOML config')
    p.add_argument('--theta', action = 'append', default = [], metavar = 'NAME=VALUE',
                   help = 'the injected point (repeatable; default: the params\' values)')
    p.add_argument('--out', required = True, metavar = 'OUT.toml', help = 'the synthetic config to write')
    p.add_argument('--sigma-scale', type = float, default = 1.0, metavar = 'X',
                   help = 'multiply every sigma by X, e.g. 0.5 for a forecast with twice the precision')
    p.add_argument('--force', action = 'store_true', help = 'overwrite OUT')
    _add_set(p)
    p.set_defaults(func = cmd_synth)

    p = sub.add_parser('resolution', help = 'choose model.n_steps and model.n_atmodeller',
                       formatter_class = fmt, description = (
                           'Runs test points at every (n_steps, n_atmodeller) cell of a grid and compares them with '
                           'the finest cell, in units of each observable\'s sigma. Recommends the cheapest cell '
                           'that is converged (and stays so at every finer cell). Writes '
                           '<output_root>/<name>/resolution_<stamp>/ with table.txt (also printed), '
                           'recommendation.json, results.npz and two figures. Check the cost first with '
                           '--dry-run.'),
                       epilog = 'examples:\n  python -m isofate.mcmc resolution lhs1140b.toml --grid quick --dry-run\n'
                                '  python -m isofate.mcmc resolution lhs1140b.toml --n-steps 1e4,3e4,1e5 '
                                '--n-atmodeller 300,100,30 --points 4')
    p.add_argument('config', metavar = 'CONFIG', help = 'TOML config')
    p.add_argument('--grid', choices = ('quick', 'default'), default = 'default',
                   help = 'quick: n_steps 1e4, 3e4, 1e5, 3e5 x n_atmodeller 1000, 100, 10; default: the config\'s '
                          '[resolution] lists (1e4-1e6 x 1000-1)')
    p.add_argument('--n-steps', type = _numbers, metavar = 'LIST', help = 'n_steps values, e.g. 1e4,3e4,1e5')
    p.add_argument('--n-atmodeller', type = _numbers, metavar = 'LIST', help = 'n_atmodeller values, e.g. 100,10')
    p.add_argument('--points', type = int, help = 'number of test points (default resolution.points)')
    p.add_argument('--source', metavar = 'S', help = 'test points: auto (default: solve in solve mode, bisect when '
                                                     'sampler.init = "bisect", else prescan), solve, prescan, prior, '
                                                     'bisect, or chain:RUN_DIR (posterior draws of a run, which '
                                                     'adds the posterior test: use 100 or more points)')
    p.add_argument('--workers', type = int, help = 'worker processes (default parallel.workers, capped by memory)')
    p.add_argument('--tol', type = float, help = 'pass when every observable is within tol sigma (default 0.1)')
    p.add_argument('--out', metavar = 'DIR',
                   help = 'output directory (default <output_root>/<name>/resolution_<stamp>)')
    p.add_argument('--dry-run', action = 'store_true', help = 'print the plan and the estimated time, run nothing')
    # --solve and --no-solve share one dest: None when neither is given, then solve mode follows [solve]
    p.add_argument('--solve', dest = 'solve', action = 'store_true', default = None,
                   help = 'evaluate every cell in solve mode, at fixed (u, other parameters) with a re-solve (the '
                          'default when [solve] is active)')
    p.add_argument('--no-solve', dest = 'solve', action = 'store_false',
                   help = 'evaluate the cells at fixed physical parameters even in solve mode')
    _add_bisect(p)
    _add_set(p)
    p.set_defaults(func = cmd_resolution)

    p = sub.add_parser('run', help = 'the MCMC (or --resume RUN_DIR)', formatter_class = fmt, description = (
        'Runs the MCMC of CONFIG in <output_root>/<name>/mcmc_<stamp>/: worker pool with warm-up, a preflight '
        'evaluation at the centre, the initialization of sampler.init (prescan, bisect, prior, ball, or solve in '
        'solve mode, where the sampler moves in u_<observable> and each evaluation solves for [solve] param), then '
        'emcee with checkpoints every sampler.checkpoint_every steps and an autocorrelation-time stopping rule. At '
        'the end: summary.json, posterior tracks and the figures. Ctrl-C saves the chain; --resume continues it '
        'bit-identically.'),
        epilog = 'examples:\n  python -m isofate.mcmc run lhs1140b.toml --workers 6 --max-steps 2000\n'
                 '  python -m isofate.mcmc run --resume RUN_DIR --max-steps 5000')
    p.add_argument('config', nargs = '?', metavar = 'CONFIG', help = 'TOML config (not with --resume)')
    p.add_argument('--resume', metavar = 'RUN_DIR', help = 'continue this run from its chain.npz')
    p.add_argument('--workers', type = int, help = 'worker processes (default parallel.workers, capped by memory)')
    p.add_argument('--nwalkers', type = int, help = 'walkers (default sampler.nwalkers: auto, a multiple of '
                                                    '4*workers)')
    p.add_argument('--max-steps', type = int, help = 'total steps, counted from the start of the run (default '
                                                     'sampler.max_steps)')
    p.add_argument('--no-plots', action = 'store_true', help = 'skip the figures at the end')
    _add_bisect(p)
    _add_set(p)
    p.set_defaults(func = cmd_run)

    p = sub.add_parser('tracks', help = 'posterior evolutionary tracks of a run', formatter_class = fmt,
                       description = 'Re-runs the MAP and N posterior samples of RUN_DIR with save_molecules and '
                                     'writes their time series to RUN_DIR/tracks.npz, then redraws the figures.',
                       epilog = 'example: python -m isofate.mcmc tracks RUN_DIR --n 100 --workers 4')
    p.add_argument('run_dir', metavar = 'RUN_DIR')
    p.add_argument('--n', type = int, help = 'posterior samples (default sampler.tracks)')
    p.add_argument('--workers', type = int, help = 'worker processes (default parallel.workers)')
    p.add_argument('--seed', type = int, default = 0, help = 'seed of the sample draw (default 0)')
    p.add_argument('--no-plots', action = 'store_true', help = 'do not redraw the figures')
    p.set_defaults(func = cmd_tracks)

    p = sub.add_parser('scan', help = 'the model on a 1-D or 2-D grid of free parameters', formatter_class = fmt,
                       description = (
                           'Evaluates the model on a grid of one (--x) or two (--x, --y) free parameters, the others '
                           "at their params' values (or --theta), through a worker pool. Writes "
                           '<output_root>/<name>/scan_<stamp>/ with scan.npz (every blob) and scan.png: 1-D, the '
                           'log-probability and each observable against x, in its likelihood space, with the data '
                           'band; 2-D, heat maps with the data as a contour. Use it to see where the posterior '
                           'lives before running the MCMC.'),
                       epilog = 'examples:\n  python -m isofate.mcmc scan lhs1140b.toml --x log_f_atm=-3.1:-2.9:81\n'
                                '  python -m isofate.mcmc scan lhs1140b.toml --x log_f_atm=-3.25:-2.85:41 '
                                '--y age=3.5e9:1e10:14 --workers 6')
    p.add_argument('config', metavar = 'CONFIG', help = 'TOML config')
    p.add_argument('--x', required = True, metavar = 'NAME=LO:HI:N', help = 'first axis: N values from LO to HI')
    p.add_argument('--y', metavar = 'NAME=LO:HI:N', help = 'second axis (a 2-D scan)')
    p.add_argument('--obs', type = lambda text: [x.strip() for x in text.split(',') if x.strip()], metavar = 'LIST',
                   help = 'observables to draw, e.g. outflow_H_fraction,Rp,Mdot_HHe (default: the active ones; '
                          'any blob field works)')
    p.add_argument('--theta', action = 'append', default = [], metavar = 'NAME=VALUE',
                   help = 'the value of another parameter (repeatable; same as --set params.NAME.value=VALUE)')
    p.add_argument('--workers', type = int, help = 'worker processes (default parallel.workers, capped by memory)')
    p.add_argument('--out', metavar = 'DIR', help = 'output directory (default <output_root>/<name>/scan_<stamp>)')
    _add_set(p)
    p.set_defaults(func = cmd_scan)

    p = sub.add_parser('compare', help = 'compare two (or more) MCMC runs', formatter_class = fmt,
                       description = (
                           'Overlays the posteriors of runs (e.g. two settings or two planets): compare_corner.png '
                           '(shared free parameters), compare_predictive.png (shared active observables with the '
                           'data), compare_tracks.png (one column per run, same panels and y limits per row) and '
                           "compare.txt (medians and 16/84 of every parameter and observable, acceptance, tau, "
                           'status counts, s/step). Default output: the first run\'s '
                           '<output_root>/<name>/compare_<stamp>/.'),
                       epilog = 'example: python -m isofate.mcmc compare RUN_A RUN_B --labels dynamic_phi,static_phi')
    p.add_argument('runs', nargs = '+', metavar = 'RUN_DIR', help = 'MCMC run directories (two or more)')
    p.add_argument('--labels', metavar = 'A,B', help = 'comma-separated names of the runs in the figures')
    p.add_argument('--out', metavar = 'DIR', help = 'output directory')
    p.set_defaults(func = cmd_compare)

    p = sub.add_parser('plot', help = 'the figures of a run, resolution or scan directory', formatter_class = fmt,
                       description = 'Writes the PNGs of an MCMC run directory (chain.npz), of a resolution '
                                     'study (results.npz) or of a scan (scan.npz) into that directory.',
                       epilog = 'example: python -m isofate.mcmc plot RUN_DIR')
    p.add_argument('dir', metavar = 'DIR')
    p.set_defaults(func = cmd_plot)

    p = sub.add_parser('summary', help = 'print the posterior summary of a run', formatter_class = fmt,
                       description = 'Prints RUN_DIR/summary.json readably (computing it from the chain if it '
                                     'is missing, e.g. while the run is going).',
                       epilog = 'example: python -m isofate.mcmc summary RUN_DIR')
    p.add_argument('run_dir', metavar = 'RUN_DIR')
    p.add_argument('--update', action = 'store_true', help = 'recompute summary.json from the current chain')
    p.add_argument('--json', action = 'store_true', help = 'print the raw JSON')
    p.set_defaults(func = cmd_summary)
    return parser


###_____Entry point_____###

def _terminate(signum, frame):
    '''
    Handler of SIGTERM and SIGHUP: raises KeyboardInterrupt, so a kill or a closed terminal takes the Ctrl-C path
    '''
    raise KeyboardInterrupt(f'signal {signal.Signals(signum).name}')


def main(argv = None):
    '''
    Entry point of python -m isofate.mcmc: parses the arguments, runs the command, maps errors to exit codes
    The message of a UsageError or ConfigError (exit 2), or of a FileNotFoundError, FileExistsError or
    RuntimeError (exit 1; WorkerDied is one), is printed without a traceback; any other exception is a bug and
    keeps its traceback. While the command runs, SIGTERM and SIGHUP raise KeyboardInterrupt, as Ctrl-C does, and
    the previous handlers come back at the end.

    Inputs:
        - argv: the arguments after the program name; None: sys.argv[1:] [list of str]

    Output: exit code: 0 ok, 1 failure, 2 usage or config error (or no command), 130 interrupted [int]
    '''
    from isofate.mcmc.config import ConfigError
    parser = make_parser()
    args = parser.parse_args(argv) # argparse exits here for --help (0) and a malformed command line (2)
    if args.command is None:
        parser.print_help()
        return 2
    env.set_single_thread_env() # before anything imports jax (check and synth run the model here)
    # A kill (SIGTERM, also to the whole process group) or a closed terminal (SIGHUP) takes the Ctrl-C path:
    # checkpoint, summary, resume command, instead of a silent loss of the steps since the last checkpoint. A
    # hang-up that is ignored already stays ignored: under nohup the run goes on after the terminal closes
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if sig == signal.SIGHUP and signal.getsignal(sig) == signal.SIG_IGN:
            continue
        previous[sig] = signal.signal(sig, _terminate)
    try:
        return args.func(args)
    except KeyboardInterrupt: # in a run, run_mcmc has saved the chain and printed the resume command
        print('\ninterrupted', file = sys.stderr)
        return 130
    except (UsageError, ConfigError) as e:
        print(f'error: {e}', file = sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, RuntimeError) as e: # messages that say what went wrong; a run's
        print(f'error: {e}', file = sys.stderr)                     # traceback is also in its run.log
        return 1
    finally:
        for sig, handler in previous.items(): # main may run again in the same process (Python use, the tests)
            signal.signal(sig, handler)


if __name__ == '__main__': # spawned workers import this module as __mp_main__: nothing must run then
    sys.exit(main())
