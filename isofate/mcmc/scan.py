'''
Grid scans (python -m isofate.mcmc scan): the model on a 1-D or 2-D grid of one or two free parameters, the
other parameters at their params' values, to see where the posterior lives before running the MCMC (e.g. how
thin the He-dominated outflow island of LHS 1140 b is in log_f_atm, and how it moves with age).

<output_root>/<name>/scan_<stamp>/ holds
    config.json  the resolved config
    scan.npz     x_name, x (nx,), and y_name, y (ny,) for 2-D; theta (n, ndim) in grid order (y slowest), lp (n,),
                 blobs (n,) as model.blob_dtype(), shape ((nx,) or (ny, nx)), obs (the observables drawn)
    scan.png     1-D: log-probability and one panel per observable against x, in the likelihood space, with the
                 data +- sigma band; 2-D: a heat map of lp and one per observable, the data as a contour
    solve_check  (in scan.npz) in solve mode, when an axis is the solved parameter: the lines of solve_report
The scan is in the physical parameters, also in solve mode (the model at fixed values, no solving). In solve mode
it also reports where T(observable) is not monotonic in the solved parameter inside the u range (solve_report):
solve mode assumes one root per bracket; and where it is constant there (a plateau, which solve mode cannot
sample).
The parent process never imports the coupled model; the evaluations run in a WorkerPool (model.eval_task).
'''

import math
import os
import time

import numpy as np

from isofate.mcmc import config, likelihood, model, search
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool


def parse_axis(text, cfg):
    '''
    'name=lo:hi:n' (n evenly spaced values from lo to hi, both included) -> (name, values). name must be a free
    parameter and every value inside its prior.
    '''
    name, sep, spec = str(text).partition('=')
    name = name.strip()
    parts = spec.split(':')
    if not sep or len(parts) != 3:
        raise config.ConfigError(f'scan axis {text!r} must look like name=lo:hi:n, e.g. log_f_atm=-3.1:-2.9:81')
    try:
        lo, hi, n = float(parts[0]), float(parts[1]), int(float(parts[2]))
    except ValueError:
        raise config.ConfigError(f'scan axis {text!r}: lo, hi and n must be numbers') from None
    if n < 1 or (n > 1 and not lo < hi):
        raise config.ConfigError(f'scan axis {text!r} needs n >= 1 and lo < hi')
    if name not in cfg['free']:
        config._registry_name(name, config.PARAMS, 'scan axis')
        raise config.ConfigError(f'scan axis {name!r} must be a free parameter (free: {", ".join(cfg["free"])}); '
                                 f'add --set params.{name}.vary=true --set params.{name}.bounds=[LO,HI]')
    values = np.linspace(lo, hi, n) if n > 1 else np.array([lo])
    outside = [x for x in values if not prm.in_support(name, x, cfg)]
    if outside:
        bounds = cfg['params'][name]['bounds']
        raise config.ConfigError(f'scan axis {name}: {len(outside)} of {n} values lie outside the prior {bounds}; '
                                 f'widen it with --set params.{name}.bounds=[LO,HI]')
    return name, values


def scan_observables(cfg, names = None):
    '''
    The observables a scan draws: names (any blob observable or extra, e.g. Mdot_HHe, fatm, exhausted), else the
    active ones computed by the model.
    '''
    if not names:
        return [n for n, o in cfg['observables'].items() if not o['input']]
    out = []
    for name in names:
        out.append(config._registry_name(name, dict.fromkeys(model.OBS_FIELDS), '--obs'))
    return out


def grid_thetas(cfg, x, y = None):
    '''
    (thetas (n, ndim), shape): the centre (the params' values) with the scanned parameters set, y slowest.
    '''
    centre = prm.centre_theta(cfg)
    (x_name, xs), j_x = x, cfg['free'].index(x[0])
    if y is None:
        thetas = np.tile(centre, (len(xs), 1))
        thetas[:, j_x] = xs
        return thetas, (len(xs),)
    (y_name, ys), j_y = y, cfg['free'].index(y[0])
    if j_y == j_x:
        raise config.ConfigError('the two scan axes must be different parameters')
    X, Y = np.meshgrid(xs, ys)
    thetas = np.tile(centre, (X.size, 1))
    thetas[:, j_x], thetas[:, j_y] = X.ravel(), Y.ravel()
    return thetas, X.shape


def _new_dir(cfg, out_dir):
    if out_dir is None:
        base = os.path.join(cfg['output_root'], cfg['name'], f"scan_{time.strftime('%Y%m%d-%H%M%S')}")
        return config.new_output_dir(base, first = 2)
    return config.empty_output_dir(out_dir, 'a scan')


def _monotonic_lines(p, T, lo, hi):
    '''
    Problems of one line T(p) (p ascending) for solve mode inside the u range [lo, hi]: separate stretches inside
    it (several roots for some u) and turning points of T inside it. Returns (stretches, turns): lists of
    (p_first, p_last) and of p values.
    '''
    inside = np.isfinite(T) & (T >= lo) & (T <= hi)
    stretches, turns = [], []
    k = 0
    while k < len(T):
        if not inside[k]:
            k += 1
            continue
        m = k
        while m + 1 < len(T) and inside[m + 1]:
            m += 1
        stretches.append((p[k], p[m]))
        d = np.sign(np.diff(T[k:m + 1]))
        d = d[d != 0]
        turns += [p[k + 1 + i] for i in np.flatnonzero(d[1:] != d[:-1])]
        k = m + 1
    return stretches, turns


def _plateau_lines(p, T, lo, hi, eps):
    '''
    Plateaus of one line T(p) (p ascending) inside the u range [lo, hi]: runs of at least two neighbouring grid
    points with the same T (within eps), e.g. the bare-core radius of a stripped planet. Returns a list of
    (p_first, p_last, T).
    '''
    out, k = [], 0
    while k < len(T) - 1:
        m = k
        while m + 1 < len(T) and np.isfinite(T[k]) and abs(T[m + 1] - T[k]) <= eps:
            m += 1
        if m > k and lo <= T[k] <= hi:
            out.append((p[k], p[m], float(T[k])))
        k = m + 1
    return out


def solve_report(cfg, data):
    '''
    Solve mode: lines saying whether T(solve observable) is monotonic in the solved parameter inside the u range
    along the scan (per line of a 2-D scan), so that the one-root assumption can be checked. [] without solve
    mode; a hint when no axis is the solved parameter.
    '''
    if not config.solve_on(cfg):
        return []
    sv = cfg['solve']
    P, O, (lo, hi) = sv['param'], sv['observable'], sv['bounds']
    axes = [str(data['x_name'])] + ([str(data['y_name'])] if 'y_name' in data else [])
    if P not in axes:
        return [f'solve check: scan {P} (--x {P}=LO:HI:N) to see whether {O} is monotonic in it inside the u range']
    T = search.g_of(data['blobs'], O, cfg['observables'][O], 0.0).reshape(tuple(int(n) for n in data['shape']))
    if T.ndim == 1:
        lines, p, other = [T], np.asarray(data['x'], dtype = float), None
    elif P == str(data['x_name']): # rows: y fixed
        lines, p, other = list(T), np.asarray(data['x'], dtype = float), (str(data['y_name']), data['y'])
    else:
        lines, p, other = list(T.T), np.asarray(data['y'], dtype = float), (str(data['x_name']), data['x'])
    space = 'log10 ' if cfg['observables'][O]['space'] == 'log10' else ''
    eps = 1e-9*likelihood.sigma_eff(cfg['observables'][O])
    out, bad, flat = [], 0, []
    for i, line in enumerate(lines):
        stretches, turns = _monotonic_lines(p, line, lo, hi)
        where = f' at {other[0]} = {other[1][i]:.4g}' if other else ''
        flat += [(where, a, b, v) for a, b, v in _plateau_lines(p, line, lo, hi, eps)]
        if not stretches:
            out.append(f'  {space}{O} never inside the u range [{lo:.4g}, {hi:.4g}]{where}')
        elif len(stretches) > 1 or turns:
            bad += 1
            out.append(f'  NOT monotonic{where}: {len(stretches)} separate stretch(es) inside the u range ('
                       + ', '.join(f'{P} {a:.5g} to {b:.5g}' for a, b in stretches) + ')'
                       + (f'; turning points at {P} = ' + ', '.join(f'{t:.5g}' for t in turns) if turns else ''))
    head = (f'solve check: {space}{O} along {P} inside the u range [{lo:.4g}, {hi:.4g}]: '
            + ('monotonic on every line of the scan (one root per u, as solve mode assumes)' if not bad else
               f'{bad} of {len(lines)} line(s) are not monotonic: there solve mode finds one of several roots, while '
               'the density is a sum over them (use a finer scan to check, or another solve.param/observable)'))
    if flat:
        head += (f'; but {O} is constant (a plateau) over part of {P} on {len({w for w, *_ in flat})} of {len(lines)} '
                 f'line(s), inside the u range: T cannot be inverted there, so solve mode leaves out the posterior '
                 f'mass of that part of {P} (e.g. a stripped planet at the bare-core radius; see "Solve mode" in '
                 'the README)')
        out = [f'  plateau{w}: {space}{O} = {v:.6g} for {P} {a:.5g} to {b:.5g}' for w, a, b, v in flat] + out
    return [head] + [line for line in out if bad or flat or 'never' in line][:20]


def _table(cfg, data, names):
    '''
    The 1-D scan as text: one line per point with lp and the observables.
    '''
    x_name, xs = str(data['x_name']), data['x']
    blobs, lp = data['blobs'], data['lp']
    width = max(12, max(len(n) for n in names) + 2)
    lines = [f'  {x_name:>12} {"lp":>12}' + ''.join(f'{n:>{width}}' for n in names) + '  status']
    for i, x in enumerate(xs):
        lines.append(f'  {x:>12.6g} {lp[i]:>12.5g}' + ''.join(f'{blobs[n][i]:>{width}.5g}' for n in names)
                     + f"  {model.STATUS_NAMES.get(int(blobs['status'][i]), '?')}")
    return lines


def run_scan(cfg, x, y = None, observables = None, workers = None, out_dir = None):
    '''
    Evaluates the grid of x = (name, values) [and y], writes scan.npz, config.json and scan.png into out_dir
    (default <output_root>/<name>/scan_<stamp>/, never overwritten), prints a summary and returns out_dir.
    '''
    start = time.perf_counter()
    names = scan_observables(cfg, observables)
    thetas, shape = grid_thetas(cfg, x, y)
    n = len(thetas)
    fixed = [f'{p} = {v:.5g}' for p, v in zip(cfg['free'], prm.centre_theta(cfg)) if p not in (x[0], y and y[0])]
    print(f"Scan of {cfg['name']}: {x[0]} ({len(x[1])} values, {x[1][0]:.6g} to {x[1][-1]:.6g})"
          + (f" x {y[0]} ({len(y[1])} values, {y[1][0]:.6g} to {y[1][-1]:.6g})" if y else '')
          + f" = {n} evaluations at n_steps = {cfg['model']['n_steps']}, n_atmodeller = "
          f"{cfg['model']['n_atmodeller']}; fixed: {', '.join(fixed) or 'nothing else'}", flush = True)
    out_dir = _new_dir(cfg, out_dir)
    config.save_resolved(cfg, os.path.join(out_dir, 'config.json'))
    requested = cfg['parallel']['workers'] if workers is None else int(workers)
    lp = np.full(n, -math.inf)
    blobs = np.empty(n, dtype = model.blob_dtype())
    blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
    with WorkerPool(cfg, max(1, min(requested, n)), run_dir = out_dir) as pool:
        loop = time.perf_counter()
        last = loop
        tasks = [{'theta': theta, 'id': i} for i, theta in enumerate(thetas)]
        for done, r in enumerate(pool.imap_unordered(model.eval_task, tasks), 1):
            blobs[r['id']], lp[r['id']] = model.blob(r), r['lp']
            now = time.perf_counter()
            if now - last > 30 or done == n:
                last = now
                eta = (now - loop)/done*(n - done)
                print(f'[scan] {done}/{n} evaluations, {now - loop:.0f} s' + (f', about {eta:.0f} s left'
                                                                             if done < n else ''), flush = True)
    data = {'x_name': x[0], 'x': np.asarray(x[1], dtype = float), 'theta': thetas, 'lp': lp, 'blobs': blobs,
            'shape': np.array(shape), 'obs': np.array(names), 'free': np.array(cfg['free']), 'name': cfg['name']}
    if y is not None:
        data.update(y_name = y[0], y = np.asarray(y[1], dtype = float))
    report = solve_report(cfg, data)
    if report:
        data['solve_check'] = np.array(report)
    np.savez_compressed(os.path.join(out_dir, 'scan.npz'), **data)

    status = ', '.join(f"{int(k)} {model.STATUS_NAMES[int(s)]}" for s, k in
                       zip(*np.unique(blobs['status'][np.isfinite(blobs['status'])], return_counts = True)))
    print(f'[scan] done in {time.perf_counter() - start:.1f} s: {status}')
    if y is None:
        print('\n'.join(_table(cfg, data, names)))
    finite = np.isfinite(lp)
    if finite.any():
        best = int(np.argmax(np.where(finite, lp, -np.inf)))
        print(f"best point: lp = {lp[best]:.6g} at " + ', '.join(f'{p} = {v:.6g}' for p, v in
                                                                  zip(cfg['free'], thetas[best])))
    else:
        print('no point of the scan has a finite log-probability')
    if report:
        print('\n'.join(report))
    try:
        from isofate.mcmc import plots
        path = plots.plot_scan(out_dir)
        print(f'wrote {path}')
    except Exception as e: # the data are saved already
        print(f'[scan] warning: the plot failed ({type(e).__name__}: {e})', flush = True)
    print(f'Results: {out_dir}', flush = True)
    return out_dir


def load_scan(scan_dir):
    '''
    The arrays of <scan_dir>/scan.npz as a dict, with the config.
    '''
    with np.load(os.path.join(scan_dir, 'scan.npz'), allow_pickle = False) as d:
        data = {key: d[key] for key in d.files}
    data['cfg'] = config.load_resolved(os.path.join(scan_dir, 'config.json'))
    return data
