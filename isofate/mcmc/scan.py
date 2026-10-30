'''
Grid scans (python -m isofate.mcmc scan): the model on a 1-D or 2-D grid of one or two free parameters, the
other parameters at their params' values, to see where the posterior lives before running the MCMC (e.g. how
thin the He-dominated outflow island of LHS 1140 b is in log_f_atm, and how it moves with age).

<output_root>/<name>/scan_<stamp>/ holds
    config.json  the resolved config
    scan.npz     x_name, x (nx,), and y_name, y (ny,) for 2-D; theta (n, ndim) in grid order (y slowest), lp (n,),
                 blobs (n,) as model.blob_dtype(), shape ((nx,) or (ny, nx)), obs (the observables drawn), free,
                 name, and in solve mode solve_check (the lines of solve_report)
    scan.png     1-D: log-probability and one panel per observable against x, in the likelihood space, with the
                 data +- sigma band; 2-D: a heat map of lp and one per observable, the data as a contour

The scan is in the physical parameters, also in solve mode (the model at fixed values, no solving). In solve mode
it also checks the assumptions solve mode rests on. Solve mode inverts T(observable) along the solved parameter
P, and it assumes one root per bracket: where T is not monotonic in P inside the u range, some values of u have several
roots, and solve mode finds only one of them while the density is a sum over them all. Where T is constant over
a stretch of P (a plateau, e.g. the bare-core radius of a stripped planet), that stretch has no u of its own:
the plateau branch of solve mode (solve.plateau) samples a plateau of stripped planets through the u that have no
root, and solve mode leaves out the posterior mass of any other. solve_report finds both, along a scan with the
solved parameter as an axis, and says which plateaus are of stripped planets.

The parent process never imports the coupled model; the evaluations run in a WorkerPool (model.eval_task).
'''

import math
import os
import time

import numpy as np

from isofate.mcmc import config, likelihood, model, search
from isofate.mcmc import params as prm
from isofate.mcmc.pool import WorkerPool


###_____Axes and the grid_____###

def parse_axis(text, cfg):
    '''
    Parses a scan axis 'name=lo:hi:n': n evenly spaced values from lo to hi, both included (lo alone for n = 1)

    Inputs:
        - text: the axis, e.g. 'log_f_atm=-3.1:-2.9:81' [str]
        - cfg: resolved config [dict]

    Output: tuple
        - name: the parameter [str]
        - values: the grid values (n,) [array]
    Raises ConfigError for a malformed axis, n < 1, lo >= hi with n > 1, a parameter that is not free, or a value
    outside the prior.
    '''
    name, sep, spec = str(text).partition('=')
    name = name.strip()
    parts = spec.split(':')
    if not sep or len(parts) != 3:
        raise config.ConfigError(f'scan axis {text!r} must look like name=lo:hi:n, e.g. log_f_atm=-3.1:-2.9:81')
    try:
        lo, hi, n = float(parts[0]), float(parts[1]), int(float(parts[2])) # n may be written 81 or 8.1e1
    except ValueError:
        raise config.ConfigError(f'scan axis {text!r}: lo, hi and n must be numbers') from None
    if n < 1 or (n > 1 and not lo < hi):
        raise config.ConfigError(f'scan axis {text!r} needs n >= 1 and lo < hi')
    if name not in cfg['free']:
        config._registry_name(name, config.PARAMS, 'scan axis') # an unknown name raises, with a suggestion
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
    Returns the observables a scan draws: names (any blob observable or extra, e.g. Mdot_HHe, fatm, exhausted),
    else the active ones computed by the model (not the inputs). An unknown name raises ConfigError.
    '''
    if not names:
        return [n for n, o in cfg['observables'].items() if not o['input']]
    out = []
    for name in names:
        out.append(config._registry_name(name, dict.fromkeys(model.OBS_FIELDS), '--obs'))
    return out


def grid_thetas(cfg, x, y = None):
    '''
    Builds the points of a scan: the centre (the params' values) with the scanned parameters set

    Inputs:
        - cfg: resolved config [dict]
        - x: (name, values) of the first axis, as parse_axis returns it [tuple]
        - y: (name, values) of the second axis, or None for a 1-D scan [tuple]

    Output: tuple
        - thetas: physical parameters (n, ndim) in grid order, x fastest and y slowest [array]
        - shape: (nx,) for 1-D, (ny, nx) for 2-D [tuple]
    Raises ConfigError when both axes are the same parameter.
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
    X, Y = np.meshgrid(xs, ys) # (ny, nx): ravel runs along x first
    thetas = np.tile(centre, (X.size, 1))
    thetas[:, j_x], thetas[:, j_y] = X.ravel(), Y.ravel()
    return thetas, X.shape


def _new_dir(cfg, out_dir):
    '''
    Creates the scan directory: <output_root>/<name>/scan_<stamp>/ (scan_<stamp>_2, _3, ... when that exists), or
    out_dir, which may exist only if empty.
    '''
    if out_dir is None:
        base = os.path.join(cfg['output_root'], cfg['name'], f"scan_{time.strftime('%Y%m%d-%H%M%S')}")
        return config.new_output_dir(base, first = 2)
    return config.empty_output_dir(out_dir, 'a scan')


###_____The solve-mode check_____###

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
        m = k # the stretch inside the range runs from k to m
        while m + 1 < len(T) and inside[m + 1]:
            m += 1
        stretches.append((p[k], p[m]))
        d = np.sign(np.diff(T[k:m + 1])) # direction of each step; equal neighbours (0) are skipped
        d = d[d != 0]
        turns += [p[k + 1 + i] for i in np.flatnonzero(d[1:] != d[:-1])] # where the direction flips
        k = m + 1
    return stretches, turns


def _plateau_lines(p, T, lo, hi, eps, stripped = None):
    '''
    Plateaus of one line T(p) (p ascending) inside the u range [lo, hi]: runs of at least two neighbouring grid
    points with the same T (within eps), e.g. the bare-core radius of a stripped planet. Returns a list of
    (p_first, p_last, T, whether every point of it is stripped); stripped: the exhausted flags of the line (None:
    not known, False).
    '''
    out, k = [], 0
    while k < len(T) - 1:
        m = k # the run of points equal to T[k] reaches m
        while m + 1 < len(T) and np.isfinite(T[k]) and abs(T[m + 1] - T[k]) <= eps:
            m += 1
        if m > k and lo <= T[k] <= hi:
            out.append((p[k], p[m], float(T[k]), stripped is not None and bool(np.all(stripped[k:m + 1]))))
        k = m + 1
    return out


def solve_report(cfg, data):
    '''
    Checks along a scan whether solve mode can invert its observable
    T is the solve observable in its likelihood space, as the root finder sees it (search.g_of: the outflow
    fractions count as undefined where H and He carry less than half of the escaping mass). On every line of the
    scan along the solved parameter (each row or column of a 2-D scan), T should be monotonic inside the u range,
    so that the one-root assumption holds, and should not be constant over a stretch (a plateau).

    Inputs:
        - cfg: resolved config [dict]
        - data: the scan arrays, as run_scan builds them or load_scan reads them [dict]

    Output: lines of text [list]: a head line with the verdict, then at most 20 detail lines (plateaus, lines that
        are not monotonic, and lines on which T never enters the u range); [] without solve mode, and a hint to
        scan the solved parameter when no axis is it
    '''
    if not config.solve_on(cfg):
        return []
    sv = cfg['solve']
    P, O, (lo, hi) = sv['param'], sv['observable'], sv['bounds']
    axes = [str(data['x_name'])] + ([str(data['y_name'])] if 'y_name' in data else [])
    if P not in axes:
        return [f'solve check: scan {P} (--x {P}=LO:HI:N) to see whether {O} is monotonic in it inside the u range']
    # T itself (g with target 0), in the shape of the grid
    shape = tuple(int(n) for n in data['shape'])
    T = search.g_of(data['blobs'], O, cfg['observables'][O], 0.0).reshape(shape)
    S = (np.asarray(data['blobs']['exhausted']) == 1).reshape(shape) # stripped
    if T.ndim == 1:
        lines, strips, p, other = [T], [S], np.asarray(data['x'], dtype = float), None
    elif P == str(data['x_name']): # rows: y fixed
        lines, strips, p, other = list(T), list(S), np.asarray(data['x'], dtype = float), (str(data['y_name']),
                                                                                           data['y'])
    else: # columns: x fixed
        lines, strips, p, other = list(T.T), list(S.T), np.asarray(data['y'], dtype = float), (str(data['x_name']),
                                                                                               data['x'])
    space = 'log10 ' if cfg['observables'][O]['space'] == 'log10' else ''
    eps = 1e-9*likelihood.sigma_eff(cfg['observables'][O]) # neighbours closer than this count as equal
    out, bad, flat = [], 0, []
    for i, line in enumerate(lines):
        stretches, turns = _monotonic_lines(p, line, lo, hi)
        where = f' at {other[0]} = {other[1][i]:.4g}' if other else ''
        flat += [(where, a, b, v, bare) for a, b, v, bare in _plateau_lines(p, line, lo, hi, eps, strips[i])]
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
        if not sv.get('plateau', False):
            what = f'with solve.plateau = false solve mode leaves out the posterior mass of that part of {P}'
        elif all(bare for *_, bare in flat):
            what = f'the plateau branch samples that part of {P} through the u that have no root (stripped planets)'
        else: # e.g. x = 1 of planets that keep an atmosphere: the branch covers stripped planets only
            what = (f'solve mode leaves out the posterior mass of the part of {P} where the planets keep an '
                    'atmosphere (the plateau branch covers stripped planets only)')
        head += (f'; but {O} is constant (a plateau) over part of {P} on {len({w for w, *_ in flat})} of {len(lines)} '
                 f'line(s), inside the u range (e.g. a stripped planet at the bare-core radius): T cannot be inverted '
                 f'there, and {what} (see "Solve mode" in the README)')
        out = [f"  plateau{w}: {space}{O} = {v:.6g} for {P} {a:.5g} to {b:.5g}{' (stripped)' if bare else ''}"
               for w, a, b, v, bare in flat] + out
    return [head] + [line for line in out if bad or flat or 'never' in line][:20]


###_____Running a scan_____###

def _table(cfg, data, names):
    '''
    The 1-D scan as text: one line per point with lp, the observables and the status.
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
    Runs a scan: evaluates the grid in a worker pool, saves it, and prints a summary
    Writes config.json, scan.npz and scan.png into the scan directory (see the module docstring), and prints the
    progress, the status counts, the table of a 1-D scan, the best point and, in solve mode, solve_report.

    Inputs:
        - cfg: resolved config [dict]
        - x: (name, values) of the first axis (parse_axis) [tuple]
        - y: (name, values) of the second axis, or None for a 1-D scan [tuple]
        - observables: names of the observables to draw (scan_observables); default the active model observables
        - workers: worker count, at most one per point; default parallel.workers [int]
        - out_dir: the scan directory; default <output_root>/<name>/scan_<stamp>/, never overwritten [str]

    Output: the scan directory [str]
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
    config.save_resolved(cfg, os.path.join(out_dir, 'config.json')) # first, so the directory says what it holds
    requested = cfg['parallel']['workers'] if workers is None else int(workers)
    lp = np.full(n, -math.inf) # placeholders until each result comes in
    blobs = np.empty(n, dtype = model.blob_dtype())
    blobs[...] = np.array(model.NAN_BLOB, dtype = model.blob_dtype())
    with WorkerPool(cfg, max(1, min(requested, n)), run_dir = out_dir) as pool:
        loop = time.perf_counter()
        last = loop
        tasks = [{'theta': theta, 'id': i} for i, theta in enumerate(thetas)]
        for done, r in enumerate(pool.imap_unordered(model.eval_task, tasks), 1): # as they finish; the id says where
            blobs[r['id']], lp[r['id']] = model.blob(r), r['lp']
            now = time.perf_counter()
            if now - last > 30 or done == n: # a progress line every 30 s at most, and one at the end
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

    # 'k status' for each status code that occurred
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
    Reads a scan directory

    Output: the arrays of scan.npz by name, plus 'cfg', the resolved config of config.json [dict]
    '''
    with np.load(os.path.join(scan_dir, 'scan.npz'), allow_pickle = False) as d:
        data = {key: d[key] for key in d.files}
    data['cfg'] = config.load_resolved(os.path.join(scan_dir, 'config.json'))
    return data
