'''
Figures of an MCMC run directory (plot_run), of a resolution study (plot_resolution) and of a grid scan
(plot_scan), written as PNG files at 150 dpi next to the data; plot_dir(path) picks the right one. compare_runs
overlays two or more runs in a compare_<stamp>/ directory.

The figures are matplotlib.figure.Figure objects rendered by Agg: nothing here touches pyplot's global state or
needs a display (no matplotlib.use, no plt.show), so importing this module in a notebook changes nothing.
Style: one blue for the posterior, categorical colours in a fixed order for series, a single-hue blue ramp for
magnitudes, near-black and grey text, light hairline grids and no top or right spines.

In solve mode the run figures show the physical parameters (the solved one from the blob field solve_value):
corner_params.png, the tracks and the comparisons; corner_sampled.png shows the sampler's coordinates (u in place of
the solved parameter) and traces.png the sampler's coordinates plus the solved parameter.
'''

import logging
import math
import os
import textwrap
import time
import warnings

import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from isofate.mcmc import config, likelihood, model, sampler
from isofate.mcmc import params as prm
from isofate.mcmc.resolution import fmt_n, fmt_seconds

DPI = 150
CATEGORICAL = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
BLUES = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b') # light -> dark
BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
INK, INK2 = '#0b0b0b', '#52514e' # primary and secondary text
GRID, AXIS = '#e8e7e3', '#b9b8b1'
HATCH = '#a3a29b' # of skipped cells
BAND = '#d6d5cf' # measurement +-1 sigma
SKIPPED = '#f3f2ee'
FLOORED = '#8d8c86' # 2-D scan cells at the log10 floor (a model value of 0)
RUN_STYLES = ('-', '--', ':', '-.') # line style per run in comparison overlays, so identical posteriors stay visible


def _run_lw(i, n, lw = 1.4):
    '''
    Line width of run i of n in a comparison overlay: earlier runs are drawn wider, so a later run on top of an
    identical one leaves its edges visible.
    '''
    return lw + 0.9*(n - 1 - i)
STYLE = {'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
         'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'legend.fontsize': 8.5, 'legend.frameon': False, 'text.color': INK, 'axes.labelcolor': INK,
         'axes.titlecolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': AXIS,
         'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
         'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'xtick.major.size': 3,
         'ytick.major.size': 3, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'hatch.linewidth': 0.6}
UNIT_TEXT = {'M_earth': r'$M_\oplus$', 'R_earth': r'$R_\oplus$', 'W/m2': r'W m$^{-2}$', 'dex': ''}


def _style():
    return matplotlib.rc_context(STYLE)


def _save(fig, folder, name):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi = DPI)
    return path


def _axis_label(label, unit, space = 'linear'):
    unit = UNIT_TEXT.get(unit, unit)
    text = f'log$_{{10}}$ {label}' if space == 'log10' else label
    return f'{text} [{unit}]' if unit else text


def _space(values, ocfg):
    flat = [likelihood.to_space(x, ocfg) for x in np.ravel(values)]
    return np.array(flat, dtype = float).reshape(np.shape(values))


def _compact(v, _ = None):
    '''
    Tick label 5e4, 0.2, 1e-5 (no '+' or leading zeros in the exponent).
    '''
    return f'{v:.3g}'.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')


def _ink_on(colour):
    '''
    Near-black or white text, whichever reads on a fill colour.
    '''
    r, g, b = to_rgb(colour)
    return INK if 0.2126*r + 0.7152*g + 0.0722*b > 0.45 else 'white'


def _grid_of_axes(fig, n, ncols):
    '''
    n axes in rows of ncols; the unused slots of the last row are switched off (returned separately).
    '''
    nrows = math.ceil(n/ncols)
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze = False)).ravel()
    for ax in axes[n:]:
        ax.set_axis_off()
    return axes[:n], axes[n:]


###_____Run directories_____###

def read_run(run_dir):
    '''
    sampler.load_run(run_dir) (cfg, chain, log_prob, blobs, accepted, summary or None, ...) plus tracks: the
    arrays of tracks.npz, or None.
    '''
    run = sampler.load_run(run_dir)
    run['tracks'] = None
    path = os.path.join(run_dir, 'tracks.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle = False) as data:
            run['tracks'] = {key: data[key] for key in data.files}
    return run


# (discard, thin) and the posterior samples exactly as in summary.json (else sampler's tau rule)
discard_thin = sampler.discard_thin
posterior = sampler.posterior


def physical_posterior(run):
    '''
    (physical samples (N, ndim), blobs (N,), the sampler's coordinates (N, ndim)) of a run after burn-in and
    thinning; the first and the last are the same without solve mode.
    '''
    samples, blobs = posterior(run)
    return sampler.physical_samples(samples, blobs, run['cfg']), blobs, samples


def _param_columns(samples, cfg):
    '''
    Free parameters for display (age in Gyr) and their axis labels.
    '''
    data, labels = np.array(samples, dtype = float), []
    for i, name in enumerate(cfg['free']):
        p = cfg['params'][name]
        if name == 'age':
            data[:, i] /= 1e9
            labels.append('age [Gyr]')
        else:
            labels.append(_axis_label(p['label'], p['unit']))
    return data, labels


def _robust_range(x):
    '''
    (lo, hi) of the finite values x: their full range, or the 0.5-99.5 percentile range when a few outliers make
    the full range more than 3 times wider (they would squash the histogram into one bin).
    '''
    lo, hi = np.min(x), np.max(x)
    p_lo, p_hi = np.percentile(x, [0.5, 99.5])
    return (p_lo, p_hi) if hi - lo > 3*(p_hi - p_lo) else (lo, hi)


def _ranges(data, truths = None):
    '''
    corner ranges: the data range (_robust_range; padded when flat), widened to a truth value if it lies within
    one span.
    '''
    ranges = []
    for i in range(data.shape[1]):
        lo, hi = _robust_range(data[:, i])
        span = hi - lo
        if truths is not None and np.isfinite(truths[i]) and lo - span <= truths[i] <= hi + span:
            lo, hi = min(lo, truths[i]), max(hi, truths[i])
        if hi - lo <= 1e-12*max(abs(lo), abs(hi), 1e-300):
            pad = 0.05*abs(lo) if lo != 0 else 0.5
            lo, hi = lo - pad, hi + pad
        pad = 0.03*(hi - lo)
        ranges.append((lo - pad, hi + pad))
    return ranges


def _corner(data, labels, path, truths = None):
    import corner
    K = data.shape[1]
    factor = 2.0 # inches per panel, as corner's default layout
    size = min(0.5*factor + factor*K + factor*(K - 1)*0.05 + 0.2*factor, 16.0)
    fig = Figure(figsize = (size, size))
    quiet = logging.root.manager.disable
    logging.disable(logging.WARNING) # corner logs 'Too few points to create valid contours' for small samples
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corner.corner(data, fig = fig, labels = labels, quantiles = [0.16, 0.5, 0.84], show_titles = True,
                          title_fmt = '.3g', color = BLUE, range = _ranges(data, truths),
                          truths = truths if K > 1 else None, # corner 2.2 fails on truths with one column
                          truth_color = ORANGE, plot_datapoints = len(data) < 5000, max_n_ticks = 4,
                          hist_kwargs = {'color': BLUE, 'linewidth': 1.2}, title_kwargs = {'fontsize': 9},
                          label_kwargs = {'fontsize': 9})
    finally:
        logging.disable(quiet)
    if K == 1 and truths is not None and np.isfinite(truths[0]):
        fig.axes[0].axvline(truths[0], color = ORANGE, lw = 1.2)
    for ax in fig.axes:
        ax.grid(False)
        ax.tick_params(labelsize = 7.5, colors = INK2)
    fig.savefig(path, dpi = DPI)
    return path


def _plot_corner_params(run, post, run_dir):
    samples = post[0]
    if not run['cfg']['free'] or len(samples) < 2:
        return None
    keep = np.all(np.isfinite(samples), axis = 1) # an old solve-mode run without solve_value: NaN
    if keep.sum() < 2:
        return None
    data, labels = _param_columns(samples[keep], run['cfg'])
    return _corner(data, labels, os.path.join(run_dir, 'corner_params.png'))


def _plot_corner_sampled(run, post, run_dir):
    '''
    corner_sampled.png (solve mode only): the sampler's coordinates, u in place of the solved parameter.
    '''
    cfg = run['cfg']
    if not config.solve_on(cfg) or len(post) < 3 or len(post[2]) < 2:
        return None
    data, labels = _param_columns(post[2], prm.sampled_cfg(cfg))
    return _corner(data, labels, os.path.join(run_dir, 'corner_sampled.png'))


def _model_observables(cfg):
    return [name for name, o in cfg['observables'].items() if not o['input']]


def _plot_corner_observables(run, post, run_dir):
    cfg, blobs = run['cfg'], post[1]
    names = _model_observables(cfg)
    if blobs is None or not names:
        return None
    data = np.column_stack([_space(blobs[name], cfg['observables'][name]) for name in names])
    data = data[np.all(np.isfinite(data), axis = 1)]
    if len(data) < 2:
        return None
    # constant columns (e.g. Phi_He/Phi_H = 0 at every sample) only add empty panels; predictive.png shows them
    keep = [j for j in range(data.shape[1]) if np.ptp(data[:, j]) > 0]
    names, data = [names[j] for j in keep], data[:, keep]
    if not names:
        return None
    labels = [_axis_label(cfg['observables'][n]['label'], cfg['observables'][n]['unit'],
                          cfg['observables'][n]['space']) for n in names]
    truths = [likelihood.observed(cfg['observables'][n]) for n in names]
    return _corner(data, labels, os.path.join(run_dir, 'corner_observables.png'), truths)


def _plot_traces(run, post, run_dir):
    cfg, chain, log_prob = run['cfg'], run['chain'], run['log_prob']
    n_steps, n_walkers, ndim = chain.shape
    if n_steps < 1:
        return None
    discard, _ = discard_thin(run)
    values, labels = _param_columns(chain.reshape(-1, ndim), prm.sampled_cfg(cfg))
    values = values.reshape(chain.shape)
    blobs = run.get('blobs')
    if config.solve_on(cfg) and blobs is not None and 'solve_value' in blobs.dtype.names: # the solved parameter too
        sv = cfg['solve']
        solved, label = _param_columns(blobs['solve_value'].reshape(-1, 1), dict(cfg, free = [sv['param']]))
        values = np.concatenate([values, solved.reshape(n_steps, n_walkers, 1)], axis = 2)
        labels = labels + [label[0] + ' (solved)']
    nrows = values.shape[2]
    fig = Figure(figsize = (8.0, 1.25*(nrows + 1) + 0.8), layout = 'constrained')
    axes = fig.subplots(nrows + 1, 1, sharex = True, squeeze = False)[:, 0]
    # step numbers of the run, as in the summary: after a stuck-walker reset the chain starts at reset_step + 1
    reset = int(run.get('extra', {}).get('reset_step', 0))
    steps = np.arange(1, n_steps + 1) + reset
    for i, ax in enumerate(axes[:-1]):
        ax.plot(steps, values[:, :, i], color = BLUE, alpha = 0.3, lw = 0.5)
        # the (sampled)/(solved) note on its own line: a long label would run past a 1.25-inch panel into the title
        ax.set_ylabel(labels[i].replace(' (sampled)', '\n(sampled)').replace(' (solved)', '\n(solved)'))
    ax = axes[-1]
    ax.plot(steps, log_prob, color = BLUE, alpha = 0.3, lw = 0.5)
    finite = log_prob[np.isfinite(log_prob)]
    below = 0.0 # fraction of the log prob values below the axis
    if finite.size:
        # down to the 2nd percentile of all values, but no further than 3 spans of the second half below its
        # maximum: walkers that start at lp = -1e6 must not flatten the equilibrium into a line
        hi = np.max(finite)
        late = log_prob[n_steps//2:]
        late = late[np.isfinite(late)]
        lo = np.percentile(finite, 2)
        if late.size:
            lo = max(lo, hi - 3*(hi - np.percentile(late, 1)))
        pad = 0.08*(hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        below = np.mean(log_prob < lo - pad)
    ax.set_ylabel('log prob')
    ax.set_xlabel('step' + (f' (steps 1-{reset}, before the stuck-walker reset, are in burnin.npz)' if reset
                            else ''))
    ax.set_xlim(reset + 1, reset + max(n_steps, 2))
    for ax in axes:
        ax.axvline(reset + discard, color = INK2, ls = '--', lw = 1)
        ax.grid(axis = 'x', visible = False)
    axes[0].annotate(f'discard = {discard}', xy = (reset + discard, 1), xycoords = ('data', 'axes fraction'),
                     xytext = (3, 2), textcoords = 'offset points', color = INK2, fontsize = 8, va = 'bottom')
    clipped = f'; {100*below:.2g}% of the log prob values are below the axis' if below > 0 else ''
    span = f'steps {reset + 1}-{reset + n_steps}' if reset else f'{n_steps} steps'
    fig.suptitle(f"{cfg['name']}: walker traces ({n_walkers} walkers, {span}{clipped})", x = 0.01, ha = 'left',
                 fontsize = 10)
    return _save(fig, run_dir, 'traces.png')


def _sigma_sides(ocfg):
    '''
    (lower, upper) effective sigma of an observable in its likelihood space.
    '''
    return likelihood.sigma_eff(ocfg, above = False), likelihood.sigma_eff(ocfg, above = True)


def _plot_predictive(run, post, run_dir):
    cfg, blobs = run['cfg'], post[1]
    names = list(cfg['observables'])
    if blobs is None or not names:
        return None
    ncols = min(3, len(names))
    nrows = math.ceil(len(names)/ncols)
    fig = Figure(figsize = (3.4*ncols, 2.3*nrows + 0.8), layout = 'constrained')
    axes, _ = _grid_of_axes(fig, len(names), ncols)
    for ax, name in zip(axes, names):
        ocfg = cfg['observables'][name]
        scale = 1e9 if name == 'age' and ocfg['space'] == 'linear' else 1.0 # age in Gyr
        x = _space(blobs[name], ocfg)/scale
        finite = x[np.isfinite(x)]
        v = likelihood.observed(ocfg)/scale
        lo_s, hi_s = (s/scale for s in _sigma_sides(ocfg))
        p_lo, p_hi = _robust_range(finite) if finite.size else (v, v)
        lo, hi = min(p_lo, v - 2*lo_s), max(p_hi, v + 2*hi_s)
        pad = 0.05*(hi - lo) if hi > lo else 1.0
        ax.set_xlim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(MaxNLocator(6)) # long labels like -0.015 collide at the default density
        outside = int(np.sum((finite < lo - pad) | (finite > hi + pad)))
        finite = finite[(finite >= lo - pad) & (finite <= hi + pad)]
        if ocfg['kind'] == 'detection':
            ax.axvspan(v - lo_s, v + hi_s, color = BAND, lw = 0, alpha = 0.7)
        else:
            sign = -1 if ocfg['kind'] == 'upper' else 1 # the allowed side
            ax.annotate('', xy = (v + sign*0.22*(hi - lo + 2*pad), 0.88), xytext = (v, 0.88),
                        xycoords = ('data', 'axes fraction'),
                        arrowprops = {'arrowstyle': '-|>', 'color': INK, 'lw': 1.2, 'shrinkA': 0, 'shrinkB': 0})
        if finite.size:
            if np.ptp(finite) > 1e-9*max(np.max(np.abs(finite)), 1e-300):
                ax.hist(finite, bins = 30, color = BLUE, alpha = 0.85, histtype = 'stepfilled', linewidth = 0)
            else: # the same value everywhere (up to round-off)
                ax.axvline(np.median(finite), color = BLUE, lw = 2.5)
        ax.axvline(v, color = INK, lw = 1.2)
        notes = []
        if finite.size + outside < x.size:
            notes.append(f'{100*(x.size - finite.size - outside)/x.size:.0f}% not finite (NaN or inf)')
        if outside:
            notes.append(f'{outside} sample{"s" if outside > 1 else ""} beyond the axis')
        if notes:
            ax.text(0.98, 0.97, '\n'.join(notes), transform = ax.transAxes, ha = 'right', va = 'top', color = INK2,
                    fontsize = 7.5)
        unit = 'Gyr' if scale != 1 else ocfg['unit']
        ax.set_xlabel(_axis_label(ocfg['label'], unit, ocfg['space']))
        kind = {'detection': '', 'upper': ' (upper limit)', 'lower': ' (lower limit)'}[ocfg['kind']]
        ax.set_title(f"{name}{kind}{' (input)' if ocfg['input'] else ''}", loc = 'left')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.grid(False)
    handles = [Patch(color = BLUE, alpha = 0.85, label = 'posterior model value'),
               Patch(color = BAND, label = r'observed $\pm1\sigma$'),
               Line2D([], [], color = INK, lw = 1.2, label = 'observed value or limit (arrow: allowed side)')]
    fig.legend(handles = handles, loc = 'outside upper center', ncols = 3)
    return _save(fig, run_dir, 'predictive.png')


TRACK_PANELS = (('Rp', r'$R_p$ [$R_\oplus$]', False), ('fatm', r'$f_{\rm atm}$', True),
                ('Mdot', r'$\dot M$ [kg/s]', True), ('outflow_H_fraction', 'outflow H/(H+He)', True),
                ('H_He', 'H/He (atoms)', True), ('C_O', 'C/O (N_C/N_O)', True), ('D_H', 'D/H', True),
                ('T_surf', r'$T_{\rm surf}$ [K]', False))
MDOT_HHE_PANEL = ('Mdot_HHe', r'$\dot M_{\rm H+He}$ [kg/s]', True) # instead of Mdot when Mdot_HHe is fitted


def _track_array(tracks, key, n_grid):
    '''
    A (n, n_grid) track array as float64 with inf (e.g. C/O once all O is dissolved) as NaN, or None.
    '''
    a = tracks.get(key)
    if a is None or np.ndim(a) != 2 or a.shape[1] != n_grid:
        return None
    a = np.asarray(a, dtype = float)
    return np.where(np.isinf(a), np.nan, a)


def _observed_age_gyr(run, post):
    cfg = run['cfg']
    if 'age' in cfg['free'] and len(post[0]):
        return float(np.median(post[0][:, cfg['free'].index('age')]))/1e9
    return cfg['params']['age']['value']/1e9


def _draw_observation(ax, x, ocfg, log):
    '''
    A measurement at time x [Gyr] in physical units: an error bar, or a limit arrow. Returns the y values it
    spans.
    '''
    v, (lo_s, hi_s) = ocfg['value'], _sigma_sides(ocfg)
    if ocfg['space'] == 'log10':
        low, high = v*10**-lo_s, v*10**hi_s
    else:
        low, high = v - lo_s, v + hi_s
    if ocfg['kind'] == 'detection':
        ax.errorbar([x], [v], yerr = [[v - low], [high - v]], fmt = 'o', ms = 4.5, color = INK, mec = 'white',
                    mew = 1, elinewidth = 1.2, capsize = 2.5, zorder = 6)
        return low, high
    end = (v/4 if ocfg['kind'] == 'upper' else v*4) if log else (low if ocfg['kind'] == 'upper' else high)
    ax.plot([x, x], [v, end], alpha = 0) # so that the y limits include the arrow
    ax.plot([x], [v], marker = '_', ms = 9, mew = 1.5, color = INK, zorder = 6)
    ax.annotate('', xy = (x, end), xytext = (x, v), annotation_clip = False, zorder = 6,
                arrowprops = {'arrowstyle': '-|>', 'color': INK, 'lw': 1.2, 'shrinkA': 0, 'shrinkB': 0})
    return v, end


def _short_log_ticks(ax):
    '''
    Plain labels (0.5, 2, 1e-5) on a log y axis spanning less than 2.5 decades, which matplotlib labels with one
    power of ten or with '6 x 10^0'-style minor labels: m x 10^n for the first of m = (1, 2, 5), (1, 2, 3, 5),
    (1, 1.5, 2, 3, 5, 7), (1, ..., 9) that gives at least 3 ticks.
    '''
    lo, hi = ax.get_ylim()
    if not 0 < lo < hi or np.log10(hi/lo) >= 2.5:
        return
    powers = range(math.floor(np.log10(lo)), math.ceil(np.log10(hi)) + 1)
    for subs in ((1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 5, 7), range(1, 10)):
        ticks = [m*10.0**p for p in powers for m in subs if lo <= m*10.0**p <= hi]
        if len(ticks) >= 3:
            break
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _log_limits(core, tracks):
    '''
    y limits of a log panel: the range of core (bands, MAP, data), widened to the individual tracks by at most one
    decade on each side, so that a single track that dives by 30 decades (an atmosphere being lost) does not
    flatten the bands. None if core has no positive values.
    '''
    core = np.asarray(core, dtype = float)
    core = core[np.isfinite(core) & (core > 0)]
    if not core.size:
        return None
    lo, hi = core.min(), core.max()
    tracks = np.asarray(tracks, dtype = float)
    tracks = tracks[np.isfinite(tracks) & (tracks > 0)]
    if tracks.size:
        lo, hi = max(min(lo, tracks.min()), lo/10), min(max(hi, tracks.max()), hi*10)
    pad = max(hi/lo, 1.5)**0.04 # a few percent of the range in log
    return lo/pad, hi*pad


def _linear_limits(core, tracks):
    '''
    y limits of a linear panel: the range of core (bands, MAP, data), widened to the individual tracks by at most
    half of that range on each side, so that one track far off (e.g. a radius of 7 R_earth at 1 Myr) does not
    flatten the others. None if core has no finite values.
    '''
    core = np.asarray(core, dtype = float)
    core = core[np.isfinite(core)]
    if not core.size:
        return None
    lo, hi = core.min(), core.max()
    span = hi - lo if hi > lo else max(abs(hi), 1.0)*0.1
    tracks = np.asarray(tracks, dtype = float)
    tracks = tracks[np.isfinite(tracks)]
    if tracks.size:
        lo, hi = max(min(lo, tracks.min()), lo - span/2), min(max(hi, tracks.max()), hi + span/2)
    pad = 0.04*(hi - lo)
    return lo - pad, hi + pad


def _track_sets(tracks):
    '''
    (t [Gyr], ok, is_map, sample) of a tracks dict, or None without usable tracks.
    '''
    if not tracks or 'grid_yr' not in tracks:
        return None
    t = np.asarray(tracks['grid_yr'], dtype = float)/1e9 # [Gyr]
    n_tracks = len(next(a for k, a in tracks.items() if _track_array(tracks, k, t.size) is not None))
    status = np.asarray(tracks.get('status', np.zeros(n_tracks)), dtype = float)
    is_map = np.asarray(tracks.get('is_map', np.zeros(n_tracks)), dtype = bool)
    ok = status == model.OK
    sample = ok & ~is_map
    if not sample.any():
        return None
    return t, ok, is_map, sample


def _track_panels(tracks, cfg, t, sample, n_vmr = 4):
    '''
    The panels (key, label, log) that a tracks dict can fill: TRACK_PANELS (Mdot_HHe instead of Mdot when it is
    fitted) and the VMRs of the n_vmr most abundant molecules at the end.
    '''
    base = [MDOT_HHE_PANEL if p[0] == 'Mdot' and 'Mdot_HHe' in cfg['observables']
            and _track_array(tracks, 'Mdot_HHe', t.size) is not None else p for p in TRACK_PANELS]
    panels = [p for p in base if _track_array(tracks, p[0], t.size) is not None]
    vmr = []
    for mol in model.MOLECULES:
        a = _track_array(tracks, f'VMR_{mol}', t.size)
        if a is None:
            continue
        last = [row[np.isfinite(row)][-1] for row in a[sample] if np.isfinite(row).any()]
        if last:
            vmr.append((np.median(last), mol))
    panels += [(f'VMR_{mol}', f'VMR {mol}', True) for _, mol in sorted(vmr, reverse = True)[:n_vmr]]
    return [p for p in panels if np.isfinite(_track_array(tracks, p[0], t.size)[sample]).any()]


def _draw_track_panel(ax, tracks, sets, key, log, cfg, age, shown, colour = None):
    '''
    One track panel: 95% and 68% bands, the median, the shown individual tracks, the MAP and the observation.
    colour: None for the run's blue ramp (bands, median) and the orange MAP; a categorical colour (compare)
    draws everything in that hue. Returns (core values for the y limits, the shown tracks).
    '''
    t, ok, is_map, sample = sets
    a = _track_array(tracks, key, t.size)
    y = a[sample]
    enough = np.isfinite(y).sum(axis = 0) >= max(3, 0.25*len(y)) # bands where >= 1/4 of the tracks run
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q = np.nanpercentile(y, [2.5, 16, 50, 84, 97.5], axis = 0)
    q[:, ~enough] = np.nan
    if log:
        q[q <= 0] = np.nan
    if colour is None:
        band95, band68, thin, median, map_colour = BLUES[0], BLUES[1], BLUES[4], BLUES[5], ORANGE
        a95 = a68 = 1.0
    else:
        band95 = band68 = thin = median = map_colour = colour
        a95, a68 = 0.15, 0.3
    ax.fill_between(t, q[0], q[4], color = band95, alpha = a95, lw = 0)
    ax.fill_between(t, q[1], q[3], color = band68, alpha = a68, lw = 0)
    for i in shown:
        ax.plot(t, a[i], color = thin, alpha = 0.25, lw = 0.6)
    ax.plot(t, q[2], color = median, lw = 1.6)
    core = [q]
    if (ok & is_map).any():
        core.append(a[np.flatnonzero(ok & is_map)[0]])
        ax.plot(t, core[-1], color = map_colour, lw = 1.1, ls = '-' if colour is None else '--')
    if key in cfg['observables']:
        core.append(_draw_observation(ax, age, cfg['observables'][key], log))
    ax.set_xscale('log')
    ax.set_xlim(t[0], t[-1])
    return np.concatenate([np.ravel(c) for c in core]), a[shown]


def _shown_tracks(sample, seed = 0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.flatnonzero(sample), size = min(20, sample.sum()), replace = False)


def _plot_tracks(run, post, run_dir):
    tracks, cfg = run['tracks'], run['cfg']
    sets = _track_sets(tracks)
    if sets is None:
        return None
    t, ok, is_map, sample = sets
    panels = _track_panels(tracks, cfg, t, sample)
    if not panels:
        return None

    ncols = 4 if len(panels) > 6 else 3
    nrows = math.ceil(len(panels)/ncols)
    fig = Figure(figsize = (3.2*ncols, 2.35*nrows + 0.7), layout = 'constrained')
    axes, spare = _grid_of_axes(fig, len(panels), ncols)
    shown = _shown_tracks(sample)
    age = _observed_age_gyr(run, post)
    for ax, (key, label, log) in zip(axes, panels):
        core, individual = _draw_track_panel(ax, tracks, sets, key, log, cfg, age, shown)
        if log:
            ax.set_yscale('log')
        limits = _log_limits(core, individual) if log else _linear_limits(core, individual)
        if limits:
            ax.set_ylim(*limits)
        if log:
            _short_log_ticks(ax)
        ax.set_title(label, loc = 'left')
    for ax in axes[-ncols:]:
        ax.set_xlabel('time [Gyr]')
    handles = [Line2D([], [], color = BLUES[5], lw = 1.6, label = 'median'),
               Patch(color = BLUES[1], label = '68%'), Patch(color = BLUES[0], label = '95%'),
               Line2D([], [], color = BLUES[4], alpha = 0.5, lw = 0.8, label = f'{len(shown)} posterior tracks')]
    if (ok & is_map).any():
        handles.append(Line2D([], [], color = ORANGE, lw = 1.1, label = 'MAP'))
    if any(key in cfg['observables'] for key, _, _ in panels):
        handles.append(Line2D([], [], color = INK, marker = 'o', ms = 4.5, lw = 1.2, label = 'observed (at the age)'))
    if len(spare):
        spare[0].legend(handles = handles, loc = 'center left')
    else: # below the panels: above them it would collide with the title
        fig.legend(handles = handles, loc = 'outside lower center', ncols = len(handles))
    fig.suptitle(f"{cfg['name']}: posterior evolutionary tracks ({int(sample.sum())} samples)", x = 0.01,
                 ha = 'left', fontsize = 10)
    return _save(fig, run_dir, 'tracks.png')


def plot_run(run_dir):
    '''
    All figures of an MCMC run that its files allow: corner_params.png, corner_sampled.png (solve mode),
    corner_observables.png, traces.png, predictive.png and tracks.png (needs tracks.npz). summary.json is
    optional (discard and thin are then estimated from the chain). A figure that fails is reported and skipped.
    Returns the paths written.
    '''
    run = read_run(run_dir)
    post = physical_posterior(run)
    paths = []
    with _style():
        for plot in (_plot_corner_params, _plot_corner_sampled, _plot_corner_observables, _plot_traces,
                     _plot_predictive, _plot_tracks):
            try:
                path = plot(run, post, run_dir)
            except Exception as e:
                print(f'[plots] warning: {plot.__name__[6:]} failed ({type(e).__name__}: {e})', flush = True)
                continue
            if path:
                paths.append(path)
    return paths


###_____Resolution studies_____###

def _read_results(res_dir):
    with np.load(os.path.join(res_dir, 'results.npz')) as data:
        return {key: data[key] for key in data.files if key != 'blobs'}


def _heatmap(r, res_dir):
    ns_grid, na_grid = list(r['grid_n_steps']), list(r['grid_n_atmodeller'])
    rows, cols = len(ns_grid), len(na_grid)
    tol, ref, rec = float(r['tol']), int(r['reference']), int(r['recommended'])
    worst = np.array(r['worst_d'], dtype = float)
    worst[ref] = 0.0
    positive = worst[np.isfinite(worst) & (worst > 0)]
    vmin = min(positive.min() if positive.size else tol, tol)/2
    vmax = max(positive.max() if positive.size else tol, tol)*2
    norm, cmap = LogNorm(vmin, vmax), LinearSegmentedColormap.from_list('blues', BLUES)

    fig = Figure(figsize = (max(1.3*cols + 2.6, 7.0), 0.78*rows + 2.5), layout = 'constrained')
    ax = fig.subplots()
    for c, verdict in enumerate(r['verdict']):
        i, j = ns_grid.index(r['n_steps'][c]), na_grid.index(r['n_atmodeller'][c])
        if verdict == 'skipped':
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor = SKIPPED, edgecolor = HATCH, hatch = '///', lw = 0))
            ax.add_patch(Rectangle((j, i), 1, 1, fill = False, edgecolor = 'white', lw = 2))
            ax.text(j + 0.5, i + 0.5, 'skipped', ha = 'center', va = 'center', color = INK2, fontsize = 7.5)
            continue
        d = worst[c]
        colour = cmap(norm(np.clip(d, vmin, vmax))) if np.isfinite(d) or np.isinf(d) else SKIPPED
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor = colour, edgecolor = 'white', lw = 2))
        text = 'ref' if c == ref else ('inf' if np.isinf(d) else '?' if np.isnan(d) else f'{d:.2g}')
        ink = _ink_on(colour)
        ax.text(j + 0.5, i + 0.42, text, ha = 'center', va = 'center', color = ink, fontsize = 9,
                fontweight = 'semibold')
        ax.text(j + 0.5, i + 0.7, fmt_seconds(r['mean_runtime'][c]), ha = 'center', va = 'center', color = ink,
                fontsize = 7)
        if verdict in ('stable', 'reference', 'pass'):
            ax.add_patch(Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fill = False, edgecolor = ink, lw = 1.2,
                                   ls = '-' if verdict != 'pass' else (0, (3, 2))))
        if c == rec:
            ax.add_patch(Rectangle((j + 0.025, i + 0.025), 0.95, 0.95, fill = False, edgecolor = ORANGE, lw = 3.2,
                                   clip_on = False, zorder = 5))
    rec_shift = int(r['recommended_shift']) if 'recommended_shift' in r else -1
    if rec_shift >= 0 and rec_shift != rec: # the second recommendation, by the equivalent shift: dashed
        i, j = ns_grid.index(r['n_steps'][rec_shift]), na_grid.index(r['n_atmodeller'][rec_shift])
        ax.add_patch(Rectangle((j + 0.025, i + 0.025), 0.95, 0.95, fill = False, edgecolor = ORANGE, lw = 2.6,
                               ls = (0, (4, 2)), clip_on = False, zorder = 5))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0) # matrix order: n_steps grows downwards, like table.txt
    ax.set_xticks(np.arange(cols) + 0.5, [str(n) for n in na_grid])
    ax.set_yticks(np.arange(rows) + 0.5, [fmt_n(n) for n in ns_grid])
    ax.tick_params(length = 0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(r'n_atmodeller (steps between Atmodeller calls)   $\rightarrow$ more calls, higher cost')
    ax.set_ylabel('n_steps')
    bar = fig.colorbar(ScalarMappable(norm, cmap), ax = ax, fraction = 0.06, pad = 0.02)
    bar.ax.axhline(tol, color = INK, lw = 1.5)
    bar.set_label(f'worst d [σ]   (line: tol = {tol:g})', color = INK)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors = INK2)
    handles = [Patch(facecolor = 'white', edgecolor = INK, lw = 1.2, label = 'stable'),
               Patch(facecolor = 'white', edgecolor = INK, lw = 1.2, ls = (0, (3, 2)),
                     label = 'passes, a finer cell fails'),
               Patch(facecolor = 'white', edgecolor = ORANGE, lw = 2.5, label = 'recommended'),
               Patch(facecolor = SKIPPED, edgecolor = HATCH, hatch = '///', lw = 0, label = 'skipped')]
    if rec_shift >= 0 and rec_shift != rec:
        handles.insert(3, Patch(facecolor = 'white', edgecolor = ORANGE, lw = 2.0, ls = (0, (4, 2)),
                                label = 'recommended by the equivalent shift'))
    fig.legend(handles = handles, loc = 'outside lower center', ncols = len(handles), handlelength = 1.6)
    title = f"{r['name']}: resolution error of each cell against the reference"
    if rec >= 0:
        title += f"\nrecommended: n_steps = {fmt_n(r['n_steps'][rec])}, n_atmodeller = {r['n_atmodeller'][rec]}"
    if rec_shift >= 0 and rec_shift != rec: # its own line: appended to the one above it ran off the figure
        title += (f"\nby the equivalent shift of {r['shift_param']} (dashed): n_steps = "
                  f"{fmt_n(r['n_steps'][rec_shift])}, n_atmodeller = {r['n_atmodeller'][rec_shift]} (table.txt)")
    fig.suptitle(title, x = 0.01, ha = 'left', fontsize = 10)
    ax.set_title('number: worst d = |model − reference| / σ, max over the test points and observables\n'
                 'below it: mean isocalc time per evaluation', loc = 'left', fontsize = 8, color = INK2)
    return _save(fig, res_dir, 'resolution_heatmap.png')


def _convergence(r, res_dir):
    names, labels = list(r['obs_names']), list(r['obs_labels'])
    units, spaces = list(r['obs_units']), list(r['obs_spaces'])
    tol, dlogl_tol, ref = float(r['tol']), float(r['dlogl_tol']), int(r['reference'])
    used, computed = r['used'], r['computed']
    n_steps, n_atm = r['n_steps'], r['n_atmodeller']
    coupled = n_atm > 0
    x = np.where(coupled, n_steps/np.where(coupled, n_atm, 1), n_steps).astype(float)
    xlabel = ('Atmodeller calls per run = n_steps / n_atmodeller' if coupled.any() else 'n_steps')
    # worst over the test points, per cell: observables (C, J), then |delta log L|
    if used.any():
        worst = np.max(np.where(used[:, None, None], r['d'], -np.inf), axis = 0)
        worst_ll = np.array(r['worst_dlogl'], dtype = float)
    else:
        worst = np.full((len(n_steps), len(names)), np.nan)
        worst_ll = np.full(len(n_steps), np.nan)
    series = [(worst[:, j], _axis_label(labels[j], units[j], spaces[j]), tol, 'd [σ]') for j in range(len(names))]
    series.append((worst_ll, r'$|\Delta \log L|$', dlogl_tol, r'$|\Delta \log L|$'))

    ncols = min(3, len(series))
    nrows = math.ceil(len(series)/ncols)
    fig = Figure(figsize = (3.4*ncols + 1.8, 2.6*nrows + 1.0), layout = 'constrained')
    axes, _ = _grid_of_axes(fig, len(series), ncols)
    ns_values = sorted(set(n_steps.tolist()))
    others = computed & (np.arange(len(n_steps)) != ref)
    edges = {'zero': False, 'inf': False} # values drawn at the bottom (0) or top (inf) edge
    for ax, (values, title, limit, ylabel) in zip(axes, series):
        finite = values[others & np.isfinite(values) & (values > 0)]
        y_lo = min(finite.min() if finite.size else limit, limit/10)/3
        y_hi = max(finite.max() if finite.size else limit, limit)*3
        for k, ns in enumerate(ns_values):
            sel = np.flatnonzero(others & (n_steps == ns))
            if not sel.size:
                continue
            sel = sel[np.argsort(x[sel])]
            colour = CATEGORICAL[k] if k < len(CATEGORICAL) else INK2
            y = np.clip(values[sel], y_lo, y_hi)
            ax.plot(x[sel], y, color = colour, lw = 1.6, marker = 'o', ms = 4.5, mec = 'white', mew = 1,
                    label = f'n_steps = {fmt_n(ns)}', clip_on = False, zorder = 3)
            for key, mask, marker in (('zero', values[sel] == 0, 'v'), ('inf', np.isinf(values[sel]), '^')):
                if mask.any():
                    edges[key] = True
                    ax.plot(x[sel][mask], y[mask], ls = 'none', marker = marker, ms = 6.5, color = colour,
                            mec = 'white', mew = 1, clip_on = False, zorder = 4)
        ax.axhline(limit, color = INK2, ls = '--', lw = 1)
        ax.annotate('tol', xy = (1, limit), xycoords = ('axes fraction', 'data'), xytext = (-2, 2),
                    textcoords = 'offset points', ha = 'right', va = 'bottom', color = INK2, fontsize = 7.5)
        ax.set_xscale('log')
        if np.log10(x[computed].max()/x[computed].min()) < 2.5: # few decades: label 1, 2, 5 x 10^n
            drawn = x[others] if others.any() else x[computed]
            narrow = np.log10(drawn.max()/drawn.min()) < 1 # e.g. one n_atmodeller: add 3 x 10^n
            ax.xaxis.set_major_locator(LogLocator(subs = (1, 2, 3, 5) if narrow else (1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}' if v < 1e4 else fmt_n(round(v))))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yscale('log')
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc = 'left')
        ax.set_ylabel(ylabel)
    fig.supxlabel(xlabel, fontsize = 9)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    for key, marker, text in (('zero', 'v', '0: equal to the\nreference'), ('inf', '^', 'inf: defined at one\n'
                                                                                         'resolution only')):
        if edges[key]:
            handles.append(Line2D([], [], ls = 'none', marker = marker, ms = 6.5, color = INK2))
            legend_labels.append(text)
    if handles:
        fig.legend(handles, legend_labels, loc = 'outside right upper')
    fig.suptitle(f"{r['name']}: convergence of each observable\nworst d over the test points against the reference "
                 f"cell\n(n_steps = {fmt_n(n_steps[ref])}, n_atmodeller = {n_atm[ref]}; not drawn)", x = 0.01,
                 ha = 'left', fontsize = 9.5)
    return _save(fig, res_dir, 'resolution_convergence.png')


def plot_resolution(res_dir):
    '''
    resolution_heatmap.png (worst d per cell, with pass marks and the recommendation) and
    resolution_convergence.png (worst d against Atmodeller calls per observable, one line per n_steps) of a
    resolution study directory. Returns the paths written.
    '''
    r = _read_results(res_dir)
    with _style():
        return [_heatmap(r, res_dir), _convergence(r, res_dir)]


###_____Scans_____###

def _param_axis(name, cfg, values):
    '''
    Display values and axis label of a parameter (age in Gyr).
    '''
    if name == 'age':
        return np.asarray(values, dtype = float)/1e9, 'age [Gyr]'
    p = cfg['params'][name]
    return np.asarray(values, dtype = float), _axis_label(p['label'], p['unit'])


def _scan_values(blobs, name, cfg):
    '''
    (values, label, ocfg, log): an active observable in its likelihood space with its config; any other blob
    field as it is, on a log axis when positive over more than two decades.
    '''
    ocfg = cfg['observables'].get(name)
    if ocfg is not None:
        return (_space(blobs[name], ocfg), _axis_label(ocfg['label'], ocfg['unit'], ocfg['space']), ocfg,
                False)
    reg = config.OBSERVABLES.get(name, {'label': name, 'unit': ''})
    values = np.asarray(blobs[name], dtype = float)
    finite = values[np.isfinite(values)]
    log = bool(finite.size) and finite.min() > 0 and finite.max()/finite.min() > 100
    return values, _axis_label(reg['label'], reg['unit']) + ' (not fitted)', None, log


def _data_band(ax, ocfg, horizontal = True):
    '''
    The observed value (line) and +-1 sigma (band; for a limit, a dashed line and an arrow to the allowed side)
    in the likelihood space. Returns the (lo, hi) it spans.
    '''
    v = likelihood.observed(ocfg)
    lo_s, hi_s = _sigma_sides(ocfg)
    if ocfg['kind'] == 'detection':
        ax.axhspan(v - lo_s, v + hi_s, color = BAND, lw = 0, alpha = 0.7, zorder = 0)
        ax.axhline(v, color = INK, lw = 1.0, zorder = 1)
        return v - lo_s, v + hi_s
    sign = -1 if ocfg['kind'] == 'upper' else 1
    ax.axhline(v, color = INK, lw = 1.0, ls = '--', zorder = 1)
    ax.annotate('', xy = (0.04, v + sign*lo_s), xytext = (0.04, v), xycoords = ('axes fraction', 'data'),
                arrowprops = {'arrowstyle': '-|>', 'color': INK, 'lw': 1.2, 'shrinkA': 0, 'shrinkB': 0})
    return min(v, v + sign*lo_s), max(v, v + sign*lo_s)


def _scan_1d(data, scan_dir):
    cfg, blobs, lp = data['cfg'], data['blobs'], np.asarray(data['lp'], dtype = float)
    names = [str(n) for n in data['obs']]
    x, xlabel = _param_axis(str(data['x_name']), cfg, data['x'])
    finite_lp = np.isfinite(lp)
    best = int(np.argmax(np.where(finite_lp, lp, -np.inf))) if finite_lp.any() else None
    panels = ['lp'] + names
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels)/ncols)
    fig = Figure(figsize = (3.7*ncols, 2.6*nrows + 1.0), layout = 'constrained')
    axes, _ = _grid_of_axes(fig, len(panels), ncols)
    ms = 3.5 if len(x) <= 120 else 0
    flags = {'undefined': False, 'band': False, 'limit': False}
    for ax, name in zip(axes, panels):
        if name == 'lp':
            values, label, ocfg, log = lp, 'log-probability', None, False
            title = 'log-probability (prior + likelihood)'
        else:
            values, label, ocfg, log = _scan_values(blobs, name, cfg)
            title = name
        finite = np.isfinite(values)
        ax.plot(x[finite], values[finite], color = BLUE, lw = 1.6, marker = 'o', ms = ms, mec = 'white', mew = 0.6,
                zorder = 3)
        span = [values[finite].min(), values[finite].max()] if finite.any() else []
        if ocfg is not None:
            span += list(_data_band(ax, ocfg))
            flags['band' if ocfg['kind'] == 'detection' else 'limit'] = True
        below = 0
        if name == 'lp' and finite.any(): # walkers never see lp far below the best: keep the top 300
            lo_cut = max(values[finite].min(), values[finite].max() - 300)
            below = int(np.sum(values[finite] < lo_cut))
            span = [lo_cut, values[finite].max()]
        if log:
            ax.set_yscale('log')
        elif span:
            lo, hi = min(span), max(span)
            pad = 0.06*(hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        missing = ~finite # undefined (NaN) or infinite: marked along the bottom edge
        if missing.any():
            flags['undefined'] = True
            ax.plot(x[missing], np.full(missing.sum(), 0.03), transform = ax.get_xaxis_transform(), ls = 'none',
                    marker = 'x', ms = 4, color = INK2, mew = 1, clip_on = False, zorder = 4)
        if best is not None:
            ax.axvline(x[best], color = INK2, ls = ':', lw = 1.0, zorder = 2)
        notes = []
        if missing.any():
            notes.append(f'{int(missing.sum())} of {len(x)} undefined or infinite')
        if below:
            notes.append(f'{below} below the axis')
        if notes: # an empty legend: placed where it hides the fewest points ('best'), on a light background
            note = ax.legend(handles = [], title = '\n'.join(notes), loc = 'best', frameon = True, framealpha = 0.85,
                             edgecolor = 'none', borderpad = 0.3, title_fontsize = 7.5)
            note.get_title().set_color(INK2)
            note.set_zorder(5)
        ax.set_title(title, loc = 'left')
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for ax in axes[-ncols:]:
        ax.set_xlabel(xlabel)
    handles = [Line2D([], [], color = BLUE, lw = 1.6, marker = 'o', ms = 3.5, mec = 'white', label = 'model')]
    if flags['band']:
        handles += [Patch(color = BAND, label = r'observed $\pm1\sigma$'),
                    Line2D([], [], color = INK, lw = 1.0, label = 'observed value')]
    if flags['limit']:
        handles.append(Line2D([], [], color = INK, lw = 1.0, ls = '--', label = 'limit (arrow: allowed side)'))
    if flags['undefined']:
        handles.append(Line2D([], [], color = INK2, ls = 'none', marker = 'x', ms = 4,
                              label = 'undefined (NaN) or inf'))
    if best is not None:
        handles.append(Line2D([], [], color = INK2, ls = ':', lw = 1.0, label = 'max log-probability'))
    fig.legend(handles = handles, loc = 'outside lower center', ncols = len(handles))
    fixed = _fixed_text(cfg, data['theta'][0], (str(data['x_name']),))
    fig.suptitle(f"{cfg['name']}: scan of {data['x_name']} ({fixed or 'no other free parameter'}; "
                 f"n_steps = {cfg['model']['n_steps']}, n_atmodeller = {cfg['model']['n_atmodeller']})", x = 0.01,
                 ha = 'left', fontsize = 10)
    return _save(fig, scan_dir, 'scan.png')


def _fixed_text(cfg, theta, scanned):
    '''
    'Mp = 5.6, age = 7.6 Gyr': the free parameters a scan keeps fixed.
    '''
    parts = []
    for p, v in zip(cfg['free'], theta):
        if p not in scanned:
            parts.append(f'age = {v/1e9:.4g} Gyr' if p == 'age' else f'{p} = {v:.4g}')
    return ', '.join(parts)


def _edges(centres):
    '''
    Cell edges around grid centres (for pcolormesh).
    '''
    c = np.asarray(centres, dtype = float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5*(c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _at_floor(values, ocfg):
    '''
    Mask of the values (in the likelihood space) at the log10 floor of an observable, i.e. model values of 0 or
    below the floor, e.g. no escape from a stripped planet. In a 2-D scan they would set the colour scale and
    flatten everything else, so the scale spans the other cells and these get the FLOORED colour. All False
    unless some, but not all, finite values are at the floor (or the observable is linear, or not fitted).
    '''
    values = np.asarray(values, dtype = float)
    out = np.zeros(values.shape, dtype = bool)
    if ocfg is None or ocfg['space'] != 'log10' or not ocfg['floor'] > 0:
        return out
    finite = np.isfinite(values)
    out = finite & (np.where(finite, values, 0.0) <= math.log10(ocfg['floor']) + 1e-9)
    return out if out.any() and not out[finite].all() else np.zeros(values.shape, dtype = bool)


def _scan_2d(data, scan_dir):
    cfg, blobs, lp = data['cfg'], data['blobs'], np.asarray(data['lp'], dtype = float)
    shape = tuple(int(s) for s in data['shape'])
    names = [str(n) for n in data['obs']]
    x, xlabel = _param_axis(str(data['x_name']), cfg, data['x'])
    y, ylabel = _param_axis(str(data['y_name']), cfg, data['y'])
    xe, ye = _edges(x), _edges(y)
    cmap = LinearSegmentedColormap.from_list('blues', BLUES)
    finite_lp = np.isfinite(lp)
    best = int(np.argmax(np.where(finite_lp, lp, -np.inf))) if finite_lp.any() else None
    panels = ['lp'] + names
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels)/ncols)
    fig = Figure(figsize = (4.0*ncols, 3.2*nrows + 1.0), layout = 'constrained')
    axes, _ = _grid_of_axes(fig, len(panels), ncols)
    flags = {'floor': False}
    for ax, name in zip(axes, panels):
        # undefined cells show the hatched background
        ax.add_patch(Rectangle((0, 0), 1, 1, transform = ax.transAxes, facecolor = SKIPPED, edgecolor = HATCH,
                               hatch = '///', lw = 0, zorder = 0))
        if name == 'lp':
            top = lp[finite_lp].max() if finite_lp.any() else 0.0
            values = np.where(finite_lp, np.maximum(lp - top, -50.0), np.nan).reshape(shape)
            label, ocfg, log = r'log-probability $-$ max (floor $-50$)', None, False
            title = 'log-probability'
        else:
            values, label, ocfg, log = _scan_values(blobs, name, cfg)
            values = np.where(np.isfinite(values), values, np.nan).reshape(shape)
            title = name
        masked = np.ma.masked_invalid(values)
        at_floor = _at_floor(values, ocfg)
        if masked.count():
            norm, extend = (LogNorm() if log else None), 'neither'
            if at_floor.any():
                above = values[np.isfinite(values) & ~at_floor]
                lo_c, hi_c = above.min(), above.max()
                norm, extend = Normalize(lo_c, hi_c if hi_c > lo_c else lo_c + 1.0), 'min'
                flags['floor'] = True
            mesh = ax.pcolormesh(xe, ye, masked, cmap = cmap.with_extremes(under = FLOORED), norm = norm,
                                 shading = 'flat', zorder = 1)
            bar = fig.colorbar(mesh, ax = ax, fraction = 0.07, pad = 0.02, extend = extend)
            bar.set_label(label, color = INK)
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors = INK2, labelsize = 7.5)
        if ocfg is not None and masked.count() > 3 and len(x) > 1 and len(y) > 1:
            v = likelihood.observed(ocfg)
            lo_s, hi_s = _sigma_sides(ocfg)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for level, style in ((v, '-'), (v - lo_s, '--'), (v + hi_s, '--')):
                    if np.nanmin(values) < level < np.nanmax(values):
                        ax.contour(x, y, masked, levels = [level], colors = INK, linewidths = 1.3 if style == '-'
                                   else 0.8, linestyles = style, zorder = 3)
            if not np.nanmin(values) < v < np.nanmax(values):
                side = 'above' if v > np.nanmax(values) else 'below'
                ax.text(0.98, 0.03, f'data ({v:.3g}) {side} the whole map', transform = ax.transAxes, ha = 'right',
                        va = 'bottom', color = INK, fontsize = 7.5, zorder = 5,
                        bbox = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1.5})
        if best is not None:
            ax.plot([x[best % shape[1]]], [y[best//shape[1]]], marker = 'o', ms = 6, mfc = 'white', mec = INK,
                    mew = 1.2, ls = 'none', zorder = 4)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(False)
        ax.set_title(title, loc = 'left')
    for ax in axes[-ncols:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color = INK, lw = 1.3, label = 'observed value'),
               Line2D([], [], color = INK, lw = 0.8, ls = '--', label = r'observed $\pm1\sigma$'),
               Patch(facecolor = SKIPPED, edgecolor = HATCH, hatch = '///', lw = 0, label = 'undefined or lp = -inf'),
               Line2D([], [], marker = 'o', ms = 6, mfc = 'white', mec = INK, ls = 'none',
                      label = 'max log-probability')]
    if flags['floor']:
        handles.insert(3, Patch(facecolor = FLOORED, lw = 0, label = 'at the log$_{10}$ floor (e.g. a value of 0)'))
    fig.legend(handles = handles, loc = 'outside lower center', ncols = len(handles))
    fixed = _fixed_text(cfg, data['theta'][0], (str(data['x_name']), str(data['y_name'])))
    fig.suptitle(f"{cfg['name']}: scan of {data['x_name']} x {data['y_name']} ({fixed or 'nothing else free'}"
                 f"; n_steps = {cfg['model']['n_steps']}, n_atmodeller = {cfg['model']['n_atmodeller']})", x = 0.01,
                 ha = 'left', fontsize = 10)
    return _save(fig, scan_dir, 'scan.png')


def plot_scan(scan_dir):
    '''
    scan.png of a scan directory (scan.npz): 1-D line panels or 2-D heat maps (see scan.py). Returns its path.
    '''
    from isofate.mcmc import scan
    data = scan.load_scan(scan_dir)
    with _style():
        return _scan_2d(data, scan_dir) if 'y' in data else _scan_1d(data, scan_dir)


###_____Comparing two runs_____###

def _compare_corner(runs, posts, labels, colours, out_dir):
    import corner
    free = [p for p in runs[0]['cfg']['free'] if all(p in r['cfg']['free'] for r in runs[1:])]
    if not free or any(len(post[0]) < 2 for post in posts):
        return None
    datas = []
    for run, post in zip(runs, posts):
        cols = [run['cfg']['free'].index(p) for p in free]
        sub = dict(run['cfg'], free = free)
        datas.append(_param_columns(post[0][:, cols], sub)[0])
    labels_axes = _param_columns(np.zeros((1, len(free))), dict(runs[0]['cfg'], free = free))[1]
    ranges = []
    for k in range(len(free)):
        lo = min(_ranges(d[:, [k]])[0][0] for d in datas)
        hi = max(_ranges(d[:, [k]])[0][1] for d in datas)
        ranges.append((lo, hi))
    K = len(free)
    size = min(1.0 + 2.0*K + 0.1*(K - 1) + 0.4, 16.0)
    fig = Figure(figsize = (size, size))
    quiet = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i, (data, colour) in enumerate(zip(datas, colours)):
                ls = RUN_STYLES[i % len(RUN_STYLES)]
                corner.corner(data, fig = fig, labels = labels_axes, color = colour, range = ranges,
                              weights = np.full(len(data), 1/len(data)), levels = (0.393, 0.865),
                              plot_datapoints = False, plot_density = False, fill_contours = False,
                              max_n_ticks = 4, # full names: hist ignores the lw/ls aliases for step histograms
                              hist_kwargs = {'color': colour, 'linewidth': _run_lw(i, len(datas)),
                                             'linestyle': ls},
                              contour_kwargs = {'linewidths': _run_lw(i, len(datas), 1.2), 'linestyles': ls},
                              label_kwargs = {'fontsize': 9})
    finally:
        logging.disable(quiet)
    for ax in fig.axes:
        ax.grid(False)
        ax.tick_params(labelsize = 7.5, colors = INK2)
    handles = [Line2D([], [], color = c, lw = _run_lw(i, len(datas)), ls = RUN_STYLES[i % len(RUN_STYLES)],
                      label = textwrap.fill(f'{label} ({len(d)} samples)', LABEL_WIDTH))
               for i, (c, label, d) in enumerate(zip(colours, labels, datas))]
    fig.legend(handles = handles, loc = 'upper right', bbox_to_anchor = (0.98, 0.98), title = '39% and 86% contours',
               title_fontsize = 8)
    path = os.path.join(out_dir, 'compare_corner.png')
    fig.savefig(path, dpi = DPI)
    return path


def _compare_predictive(runs, posts, labels, colours, out_dir):
    cfg0 = runs[0]['cfg']
    names = [n for n in cfg0['observables'] if all(n in r['cfg']['observables'] for r in runs[1:])]
    if not names or any(p[1] is None for p in posts):
        return None
    ncols = min(3, len(names))
    nrows = math.ceil(len(names)/ncols)
    fig = Figure(figsize = (3.5*ncols, 2.4*nrows + 0.9), layout = 'constrained')
    axes, _ = _grid_of_axes(fig, len(names), ncols)
    for ax, name in zip(axes, names):
        ocfg = cfg0['observables'][name]
        scale = 1e9 if name == 'age' and ocfg['space'] == 'linear' else 1.0
        values = []
        for post in posts:
            x = _space(post[1][name], ocfg)/scale if name in post[1].dtype.names else np.array([np.nan])
            values.append(x[np.isfinite(x)])
        v = likelihood.observed(ocfg)/scale
        lo_s, hi_s = (s/scale for s in _sigma_sides(ocfg))
        pool = np.concatenate([x for x in values if x.size]) if any(x.size for x in values) else np.array([v])
        p_lo, p_hi = _robust_range(pool)
        lo, hi = min(p_lo, v - 2*lo_s), max(p_hi, v + 2*hi_s)
        pad = 0.05*(hi - lo) if hi > lo else 1.0
        bins = np.linspace(lo - pad, hi + pad, 41)
        if ocfg['kind'] == 'detection':
            ax.axvspan(v - lo_s, v + hi_s, color = BAND, lw = 0, alpha = 0.7)
        ax.axvline(v, color = INK, lw = 1.2, ls = '-' if ocfg['kind'] == 'detection' else '--')
        for i, (x, colour) in enumerate(zip(values, colours)):
            ls = RUN_STYLES[i % len(RUN_STYLES)]
            if x.size and np.ptp(x) > 1e-9*max(np.max(np.abs(x)), 1e-300):
                ax.hist(np.clip(x, bins[0], bins[-1]), bins = bins, density = True, histtype = 'step',
                        linewidth = _run_lw(i, len(values)), color = colour, linestyle = ls)
            elif x.size:
                ax.axvline(np.median(x), color = colour, lw = _run_lw(i, len(values), 2.0), ls = ls)
        others = [r['cfg']['observables'][name] for r in runs[1:]]
        if any((o['value'], o['sigma'], o['space'], o['kind']) != (ocfg['value'], ocfg['sigma'], ocfg['space'],
                                                                   ocfg['kind']) for o in others):
            ax.text(0.98, 0.97, f'data of {labels[0]} shown;\nthe runs differ', transform = ax.transAxes,
                    ha = 'right', va = 'top', color = INK2, fontsize = 7.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_xlabel(_axis_label(ocfg['label'], 'Gyr' if scale != 1 else ocfg['unit'], ocfg['space']))
        ax.set_title(f"{name}{' (input)' if ocfg['input'] else ''}", loc = 'left')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.grid(False)
    handles = [Line2D([], [], color = c, lw = _run_lw(i, len(labels)), ls = RUN_STYLES[i % len(RUN_STYLES)],
                      label = textwrap.fill(label, LABEL_WIDTH)) for i, (c, label) in enumerate(zip(colours, labels))]
    handles += [Patch(color = BAND, label = r'observed $\pm1\sigma$'),
                Line2D([], [], color = INK, lw = 1.2, label = 'observed value (dashed: limit)')]
    # two columns at most: a row of long run labels is wider than a narrow figure
    fig.legend(handles = handles, loc = 'outside upper center', ncols = min(len(handles), 2 if ncols < 3 else 4))
    return _save(fig, out_dir, 'compare_predictive.png')


def _compare_tracks(runs, posts, labels, colours, out_dir):
    sets = [_track_sets(r['tracks']) for r in runs]
    if any(s is None for s in sets):
        return None
    panel_lists = [_track_panels(r['tracks'], r['cfg'], s[0], s[3], n_vmr = 2) for r, s in zip(runs, sets)]
    keys = [set(p[0] for p in panels) for panels in panel_lists[1:]]
    panels = [p for p in panel_lists[0] if all(p[0] in k for k in keys)]
    if not panels:
        return None
    n = len(runs)
    fig = Figure(figsize = (3.6*n + 0.4, 1.75*len(panels) + 1.0), layout = 'constrained')
    axes = fig.subplots(len(panels), n, squeeze = False)
    t_lo = min(s[0][0] for s in sets)
    t_hi = max(s[0][-1] for s in sets)
    for row, (key, label, log) in enumerate(panels):
        cores, individuals = [], []
        for col, (run, post, s, colour) in enumerate(zip(runs, posts, sets, colours)):
            ax = axes[row, col]
            core, individual = _draw_track_panel(ax, run['tracks'], s, key, log, run['cfg'],
                                                 _observed_age_gyr(run, post), _shown_tracks(s[3]), colour)
            cores.append(core)
            individuals.append(np.ravel(individual))
            ax.set_xlim(t_lo, t_hi)
            if row == 0:
                ax.set_title(f'{textwrap.fill(labels[col], 34)}\n{label}', loc = 'left')
            else:
                ax.set_title(label, loc = 'left')
        # the same y limits in every column of a row
        limits = (_log_limits if log else _linear_limits)(np.concatenate(cores), np.concatenate(individuals))
        for col in range(n):
            ax = axes[row, col]
            if log:
                ax.set_yscale('log')
            if limits:
                ax.set_ylim(*limits)
            if log:
                _short_log_ticks(ax)
    for col in range(n):
        axes[-1, col].set_xlabel('time [Gyr]')
    handles = [Line2D([], [], color = c, lw = 1.6,
                      label = textwrap.fill(f'{label}: median, 68% and 95% bands, 20 tracks', LABEL_WIDTH))
               for c, label in zip(colours, labels)]
    handles += [Line2D([], [], color = INK2, lw = 1.1, ls = '--', label = 'MAP (dashed, run colour)'),
                Line2D([], [], color = INK, marker = 'o', ms = 4.5, lw = 1.2, label = 'observed (at the age)')]
    fig.legend(handles = handles, loc = 'outside upper center', ncols = 1 if n < 3 else 2)
    return _save(fig, out_dir, 'compare_tracks.png')


def _run_stats(run, post):
    '''
    Sampling and convergence numbers of a run for compare.txt: from summary.json when it matches the chain,
    else computed from the chain.
    '''
    chain, extra = run['chain'], run['extra']
    n = chain.shape[0]
    acceptance = run['accepted']/max(n, 1)
    s = run.get('summary') or {}
    if s.get('steps') == n:
        tau, converged = s.get('tau_max', np.nan), s.get('converged')
    else:
        tau = np.nanmax(sampler.autocorr_time(chain, acceptance)) if n >= 3 else np.nan
        converged = bool(sampler._get(extra, 'converged', False))
    counts = np.asarray(sampler._get(extra, 'counts_sampling', np.zeros(sampler.N_STATUS)), dtype = int)
    steps_timed = int(sampler._get(extra, 'steps_timed', 0))
    s_per_step = float(sampler._get(extra, 'sampling_s', np.nan))/steps_timed if steps_timed else np.nan
    return {'steps': n, 'walkers': chain.shape[1], 'samples': len(post[0]), 'acceptance': float(acceptance.mean()),
            'tau': tau, 'n_over_tau': n/tau if tau and np.isfinite(tau) else np.nan, 'converged': converged,
            'status': sampler.status_text(counts), 's_per_step': s_per_step,
            'workers': int(sampler._get(extra, 'workers', 0))}


def _median_cell(x):
    x = np.asarray(x, dtype = float)
    x = x[np.isfinite(x)]
    if not x.size:
        return 'undefined'
    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    return f'{p50:.5g} +{p84 - p50:.3g} -{p50 - p16:.3g}'


def compare_text(runs, posts, labels):
    '''
    compare.txt: per run the sampling numbers, then the median and 16/84 interval of every free parameter and
    observable of either run (and the data).
    '''
    width = 34
    stats = [_run_stats(r, p) for r, p in zip(runs, posts)]
    head = '  ' + ' '*22 + ''.join(f'{label[:width - 2]:>{width}}' for label in labels)
    out = ['COMPARISON OF MCMC RUNS', '='*(24 + width*len(runs))]
    for label, run in zip(labels, runs):
        cfg = run['cfg']
        out.append(f"  {label}: {run['run_dir']}")
        out.append(f"      {cfg['name']}; n_steps = {cfg['model']['n_steps']}, n_atmodeller = "
                   f"{cfg['model']['n_atmodeller']}, dynamic_phi = {str(cfg['model']['dynamic_phi']).lower()}, "
                   f"backend {cfg['model']['backend']}, init {cfg['sampler']['init']}, seed {cfg['sampler']['seed']}")
    out += ['', 'Sampling', head]

    def row(name, values):
        return f'  {name:<22}' + ''.join(f'{v:>{width}}' for v in values)
    out.append(row('steps x walkers', [f"{s['steps']} x {s['walkers']}" for s in stats]))
    out.append(row('posterior samples', [str(s['samples']) for s in stats]))
    out.append(row('acceptance (mean)', [f"{s['acceptance']:.3f}" for s in stats]))
    out.append(row('tau max [steps]', [f"{s['tau']:.3g}" for s in stats]))
    out.append(row('N/tau', [f"{s['n_over_tau']:.3g} ({'converged' if s['converged'] else 'not converged'})"
                             for s in stats]))
    out.append(row('s/step', [f"{s['s_per_step']:.3g} ({s['workers']} workers)" for s in stats]))
    for i, label in enumerate(labels):
        out.append(f"  evaluations of {label} (sampling): {stats[i]['status']}")
    out += ['', 'Parameters (median +(84th - median) -(median - 16th))', head]
    free = list(dict.fromkeys(p for r in runs for p in r['cfg']['free']))
    for p in free:
        cells = []
        for run, post in zip(runs, posts):
            cells.append(_median_cell(post[0][:, run['cfg']['free'].index(p)]) if p in run['cfg']['free']
                         else 'fixed')
        out.append(row(p, cells))
    out += ['', 'Observables (posterior of the model value; data of the first run that fits it)',
            head + f"{'data':>{width}}"]
    names = list(dict.fromkeys(n for r in runs for n in r['cfg']['observables']))
    for name in names:
        cells = []
        for run, post in zip(runs, posts):
            blobs = post[1]
            cells.append(_median_cell(blobs[name]) if blobs is not None and name in blobs.dtype.names else '-')
        data = next(r['cfg']['observables'][name] for r in runs if name in r['cfg']['observables'])
        out.append(row(name, cells) + f'{sampler._data_text(data):>{width}}')
    return '\n'.join(out) + '\n'


LABEL_WIDTH = 40 # characters per line of a run label in comparison legends


def default_labels(runs):
    '''
    Short labels of runs: the run directory names when the runs share a config name, else the names (both,
    <name>/<run dir>, when that is still ambiguous).
    '''
    names = [r['cfg']['name'] for r in runs]
    dirs = [os.path.basename(r['run_dir']) for r in runs]
    labels = dirs if len(set(names)) == 1 else names
    if len(set(labels)) < len(labels):
        labels = [f'{n}/{d}' for n, d in zip(names, dirs)]
    return labels


def compare_runs(run_dirs, labels = None, out_dir = None):
    '''
    Compares MCMC runs (two, or more): compare_corner.png (the shared free parameters overlaid), compare_predictive.png
    (the shared active observables overlaid on the data), compare_tracks.png (one column per run, the same panels
    and y limits per row) and compare.txt, in out_dir (default: the first run's
    <output_root>/<name>/compare_<stamp>/, never overwritten). Returns (out_dir, paths).
    '''
    runs = [read_run(d) for d in run_dirs]
    posts = [physical_posterior(r) for r in runs] # the physical parameters (P* of a solve-mode run)
    if labels is None:
        labels = default_labels(runs)
    if len(labels) != len(runs):
        raise ValueError(f'{len(labels)} labels for {len(runs)} runs')
    if len(runs) > len(CATEGORICAL):
        raise ValueError(f'at most {len(CATEGORICAL)} runs can be compared')
    colours = CATEGORICAL[:len(runs)]
    if out_dir is None:
        cfg = runs[0]['cfg']
        base = os.path.join(cfg['output_root'], cfg['name'], f"compare_{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir = config.new_output_dir(base, first = 2)
    else:
        out_dir = config.empty_output_dir(out_dir, 'a comparison')
    text = compare_text(runs, posts, labels)
    with open(os.path.join(out_dir, 'compare.txt'), 'w') as f:
        f.write(text)
    paths = [os.path.join(out_dir, 'compare.txt')]
    with _style():
        for plot in (_compare_corner, _compare_predictive, _compare_tracks):
            try:
                path = plot(runs, posts, labels, colours, out_dir)
            except Exception as e:
                print(f'[plots] warning: {plot.__name__[1:]} failed ({type(e).__name__}: {e})', flush = True)
                continue
            if path:
                paths.append(path)
    return out_dir, paths


def plot_dir(path):
    '''
    plot_resolution for a resolution study directory (results.npz), plot_run for an MCMC run (chain.npz),
    plot_scan for a scan (scan.npz).
    '''
    if os.path.exists(os.path.join(path, 'results.npz')):
        return plot_resolution(path)
    if os.path.exists(os.path.join(path, 'chain.npz')):
        return plot_run(path)
    if os.path.exists(os.path.join(path, 'scan.npz')):
        return [plot_scan(path)]
    raise FileNotFoundError(f'{path} has neither chain.npz (an MCMC run), results.npz (a resolution study) nor '
                            'scan.npz (a scan)')
