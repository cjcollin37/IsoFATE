'''
Root finding along one free parameter: the bisection initializer (sampler.init = "bisect", resolution source =
"bisect", check --bisect) and the solve of solve mode (solve.py), with one root finder (Root) for both.

Along a parameter p, inside its prior bounds [lo, hi], with the other parameters fixed, the model is driven to a
target value of one observable k:

    g(p) = T(model value of k at p) - target

T is the space the likelihood compares k in (linear, or log10 with the floor). The bisection initializer targets
the data of k = sampler.bisect_observable along p = sampler.bisect_param and stops when |g| <= bisect_tol*sigma
(sigma: the effective sigma of k on the side of the data the model is on). Solve mode targets u, the sampled
coordinate, along solve.param and stops when |g| <= solve.tol*sigma.

The root finder:
- Cold start: both bounds are evaluated. A bracket needs a sign change of g between them.
- Warm start (solve mode with solve_table.npz): the guess is evaluated, then a bracket is expanded from it toward
  the root (the expected direction from the sign of g and of dT/dp), until g changes sign. The first step is 1.5
  Newton steps (|g| over the table's expected slope |dT/dp|); a first step that takes g away from the target
  turns the expansion around; later steps are 1.5 times the secant's distance to the root (within 0.5 and 4 times
  the previous step), or doubling. At a prior bound without a sign change the other bound is tried. When the
  expansion meets an undefined region, the bound on the other side of the guess decides, by the NaN-end rule of
  the cold search, whether the root lies toward that bound or toward the undefined region.
- Then each step evaluates a point inside the bracket and keeps the part whose ends differ in sign: the midpoint
  (the geometric mean for a loguniform prior), or, while both ends are finite, the Illinois (modified regula
  falsi) point. When 3 Illinois steps in a row have not halved the bracket the next step bisects, so the bracket
  halves at least every 4 steps.
- It stops when |g| <= tol*sigma (converged), when the bracket is narrower than 1e-9 of the prior width without
  that (g jumps across the target: "jump", no root), or at the evaluation limit (bisect_iters rounds, or
  solve.max_evals).

Rules shared by every use:
- g need not be monotonic: any sign change brackets a root, and each search finds one root (not all of them).
- An undefined value (NaN, e.g. the outflow composition of a stripped planet) takes the sign opposite to the
  defined end of the bracket: the undefined region is treated as lying on the other side of the target. When both
  ends are defined, a NaN inside the bracket stops the search ("undefined inside the bracket"). An infinite g (e.g.
  log10 of 0 with floor = 0) is defined: it has a sign (but no Illinois step uses it).
- The outflow fractions (outflow_H_fraction, outflow_He_fraction and the legacy escape_H_He, escape_He_H) count
  as undefined where H and He carry less than half of the escaping mass (Mdot_HHe < 0.5 Mdot): a metal-dominated
  wind, e.g. at the stripping edge with dynamic_phi = true, where H and He are nearly gone and their ratio swings
  through any value within 1e-3 dex of log_f_atm. That is not the H/He wind the p-winds fractions describe, so
  those points join the stripped side and the search finds the root of the H/He wind beyond them. The
  likelihood itself is unchanged: such points keep their (very low) likelihood when the walkers are moved by the
  parameter itself.
- A detection targets its value. An upper or lower limit targets the limit value, so the bisection starts the
  walkers on the boundary of the allowed region (the config warns).
- A search ends without a root when g has the same sign at both ends (no sign change), when both ends are
  undefined, at a jump, or when the model fails there. failure_text explains a failed initialization from the
  counts of each case.

The bisection only finds starting points: it puts the walkers on the thin surface where k fits, e.g. the
H-depleted, He-dominated outflow island of LHS 1140 b, which no prior draw hits by chance. Its evaluations go
through a pool's map of model.log_prob_physical (a WorkerPool, the sampler's counting pool, or model.LocalPool in
one process): one map per round over every draw still running. The solve runs inside one evaluation (find_root).
'''

import math
import os
import time

import numpy as np

from isofate.mcmc import config, likelihood, model
from isofate.mcmc import params as prm

nan, inf = math.nan, math.inf
RAN = (model.OK, model.NO_LIKELIHOOD) # statuses of evaluations whose model ran to the end
BATCHES = 3 # at most this many batches of draws before giving up
CLOSED = 1e-9 # a bracket narrower than this fraction of the prior width is closed
ILLINOIS_MARGIN = 1e-4 # an Illinois point stays this fraction of the bracket away from its ends
ILLINOIS_STEPS = 3 # Illinois steps that must halve the bracket between them, else the next step bisects
NEWTON_REACH = 1.5 # the first step of a warm start, in Newton steps |g|/slope
WIND_FRACTIONS = ('outflow_H_fraction', 'outflow_He_fraction', 'escape_H_He', 'escape_He_H')
HHE_DOMINANT = 0.5 # an outflow fraction is defined when H and He carry at least this fraction of the escaping mass

# outcome codes of a search
CONVERGED, ABOVE, BELOW, UNDEFINED, FAILED, HOLE, UNCONVERGED, NO_LP, JUMP = range(9)
OUTCOMES = {CONVERGED: 'converged', ABOVE: 'g > 0 at both ends', BELOW: 'g < 0 at both ends',
            UNDEFINED: 'undefined at both ends', FAILED: 'model failed', HOLE: 'undefined inside the bracket',
            UNCONVERGED: 'not converged', NO_LP: 'converged, lp = -inf', JUMP: 'jump across the target'}
NO_ROOT = (ABOVE, BELOW, UNDEFINED, JUMP) # no parameter value inside the prior gives the target
# what "model above / below the data at both ends" usually means, per observable
EXAMPLES = {'Rp': (', e.g. an observed radius below the bare-core radius',
                   ', e.g. an observed radius above what the largest {p} gives'),
            'outflow_H_fraction': (', e.g. an H-rich outflow (He not escaping) at both ends',
                                   ', e.g. an outflow poorer in H than observed at both ends'),
            'fatm': (', e.g. too much atmosphere left even at the smallest {p}', ', e.g. stripped at both ends')}


###_____The settings and g_____###

def bracket(cfg, param):
    '''
    {param, index (in the physical theta, cfg['free']), lo, hi (its prior support, moved 1e-9 of the box inside an
    open hard limit such as log_f_atm < 0), log (loguniform prior: geometric midpoints)}.
    '''
    j = cfg['free'].index(param)
    lo, hi = (float(x[j]) for x in prm.prior_box(cfg))
    if not prm.in_support(param, lo, cfg):
        lo += 1e-9*(hi - lo)
    if not prm.in_support(param, hi, cfg):
        hi -= 1e-9*(hi - lo)
    return {'param': param, 'index': j, 'lo': lo, 'hi': hi,
            'log': cfg['params'][param]['prior'] == 'loguniform' and lo > 0}


def settings(cfg):
    '''
    The bisection settings of cfg: bracket() of sampler.bisect_param plus {observable, ocfg, target (its data in the
    likelihood space), tol, iters}. Raises ConfigError when the parameter is not free or the observable not active.
    '''
    param, name = config.bisect_target(cfg)
    ocfg = cfg['observables'][name]
    s = cfg['sampler']
    return dict(bracket(cfg, param), observable = name, ocfg = ocfg, target = likelihood.observed(ocfg),
                tol = float(s['bisect_tol']), iters = int(s['bisect_iters']))


def search_values(values, name):
    '''
    The values of observable name as the root finder sees them, from a blob array or an obs dict: NaN for an
    outflow fraction where H and He carry less than half of the escaping mass (the module docstring).
    '''
    x = np.asarray(values[name], dtype = float)
    if name in WIND_FRACTIONS:
        with np.errstate(invalid = 'ignore'):
            mdot = np.asarray(values['Mdot'], dtype = float)
            hhe = np.asarray(values['Mdot_HHe'], dtype = float) >= HHE_DOMINANT*mdot
        x = np.where(hhe, x, nan)
    return x


def g_of(values, name, ocfg, target):
    '''
    g = T(model value) - target of a blob array or an obs dict (NaN when undefined).
    '''
    x = np.atleast_1d(search_values(values, name))
    T = np.array([likelihood.to_space(v, ocfg) for v in x], dtype = float)
    return T - target


def _g(blobs, st):
    '''
    g of each blob for the bisection settings st, and whether its model ran.
    '''
    return g_of(blobs, st['observable'], st['ocfg'], st['target']), np.isin(blobs['status'], RAN)


def within(g, ocfg, tol):
    '''
    |g| <= tol*sigma, sigma on the side of the target the model is on.
    '''
    if not math.isfinite(g):
        return False
    return abs(g) <= tol*likelihood.sigma_eff(ocfg, above = g > 0)


def _within(g, st):
    return within(g, st['ocfg'], st['tol'])


def _sign(g, nan_sign):
    return nan_sign if math.isnan(g) else float(np.sign(g))


###_____The root finder_____###

class Root:
    '''
    One root search of g along a parameter in [lo, hi] (the module docstring), driven from outside:

        root = Root(st, target, tol)               # cold; Root(..., guess = p0, step = dp, orient = +1) warm
        while root.outcome is None:
            xs = root.pending()                    # one or two parameter values to evaluate now
            root.tell([(x, g, ran, payload), ...]) # their g, whether the model ran, anything to keep (e.g. a blob)

    st: bracket() plus ocfg. At the end: outcome (OUTCOMES), x and g (the root, or the last point), payload (of the
    root), a and b (the bracket; None without one), sign_a (sign of g at a; dT/dp has the sign -sign_a), evals.
    '''

    def __init__(self, st, target, tol, guess = None, step = None, orient = 0.0, slope = None):
        self.ocfg, self.target, self.tol = st['ocfg'], float(target), float(tol)
        self.lo, self.hi, self.log = st['lo'], st['hi'], bool(st['log'])
        self.width = self._y(self.hi) - self._y(self.lo)
        self.outcome, self.x, self.g, self.payload = None, nan, nan, None
        self.a = self.b = None
        self.sign_a, self.nan_sign, self.evals = 0.0, 0.0, 0
        if guess is None or not math.isfinite(guess):
            self.phase, self._pending = 'ends', [self.lo, self.hi]
        else:
            self.phase, self._pending = 'guess', [min(max(float(guess), self.lo), self.hi)]
            self.step = abs(float(step)) if step else 1e-3*self.width # [in y]
            self.orient = 1.0 if orient >= 0 else -1.0 # expected sign of dT/dp
            self.slope = abs(float(slope)) if slope and math.isfinite(slope) else None # expected |dT/dy|

    def _y(self, x):
        return math.log(x) if self.log else x

    def _x(self, y):
        return math.exp(y) if self.log else y

    def pending(self):
        return [] if self.outcome is not None else list(self._pending)

    def stop(self, outcome):
        '''
        Ends the search from outside (e.g. at the evaluation limit), keeping the last point.
        '''
        if self.outcome is None:
            self.outcome = outcome

    def _finish(self, outcome, x = nan, g = nan, payload = None):
        self.outcome, self.x, self.g, self.payload = outcome, x, g, payload
        self._pending = []

    def _fits(self, g):
        return within(g, self.ocfg, self.tol)

    def tell(self, items):
        '''
        items: [(x, g, ran, payload)] for the pending values, in order.
        '''
        self.evals += len(items)
        getattr(self, f'_tell_{self.phase}')(items)

    ###_____phases_____###

    def _tell_ends(self, items):
        (x_lo, g_lo, ran_lo, p_lo), (x_hi, g_hi, ran_hi, p_hi) = items
        if not (ran_lo and ran_hi):
            return self._finish(FAILED)
        # an end that already fits (the smaller |g| if both do; lo first on a tie)
        fits = [(abs(g), k, x, g, p) for k, (x, g, p) in enumerate(((x_lo, g_lo, p_lo), (x_hi, g_hi, p_hi)))
                if self._fits(g)]
        if fits:
            _, _, x, g, p = min(fits, key = lambda f: f[:2])
            return self._finish(CONVERGED, x, g, p)
        defined_lo, defined_hi = not math.isnan(g_lo), not math.isnan(g_hi)
        if not defined_lo and not defined_hi:
            return self._finish(UNDEFINED)
        s_lo = float(np.sign(g_lo)) if defined_lo else -float(np.sign(g_hi))
        s_hi = float(np.sign(g_hi)) if defined_hi else -float(np.sign(g_lo))
        if s_lo == s_hi:
            return self._finish(ABOVE if s_lo > 0 else BELOW)
        self._bracket(x_lo, g_lo, s_lo, x_hi, g_hi, s_hi)

    def _tell_guess(self, items):
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        if math.isnan(g): # no direction to go: the cold search
            self.phase, self._pending = 'ends', [self.lo, self.hi]
            return
        self.x0, self.g0, self.s0 = x, g, float(np.sign(g))
        self.dir = -self.s0*self.orient # toward the root if T(p) is monotonic with the expected orientation
        if self.slope and math.isfinite(g): # 1.5 Newton steps, within 1/16 and 16 times the table's step
            self.step = min(max(NEWTON_REACH*abs(g)/self.slope, self.step/16), 16*self.step)
        self.prev, self.length, self.reversed, self.x_nan = (x, g), self.step, False, None
        self._probe(self._y(x) + self.dir*self.length)

    def _probe(self, y):
        '''
        Evaluates y next in the expansion (clipped to the prior bounds), or the other bound once the expansion
        stands on one.
        '''
        x = self.hi if y >= self._y(self.hi) else self.lo if y <= self._y(self.lo) else self._x(y)
        if x == self.prev[0]: # already on the bound it would step to
            return self._other()
        self.phase, self._pending = 'expand', [x]

    def _other(self):
        '''
        No sign change up to a bound: the root, if any, lies on the other side of the guess.
        '''
        x = self.hi if self.dir < 0 else self.lo
        if x == self.x0:
            return self._finish(ABOVE if self.s0 > 0 else BELOW, self.x0, self.g0)
        self.phase, self._pending = 'other', [x]

    def _tell_expand(self, items):
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        if math.isnan(g): # an undefined region that way: the far bound decides, as in the cold search
            self.x_nan = x
            return self._far()
        px, pg = self.prev
        s = float(np.sign(g))
        if s != self.s0:
            return self._bracket(px, pg, self.s0, x, g, s)
        if not self.reversed and px == self.x0 and abs(g) > abs(pg): # the first step went away from the target
            self.reversed, self.dir = True, -self.dir
            return self._probe(self._y(self.x0) + self.dir*self.length)
        if x in (self.lo, self.hi):
            return self._other()
        dy = abs(self._y(x) - self._y(px))
        step = 2*dy # doubling, or 1.5 times the secant's distance to the root (within 0.5 to 4 steps)
        if abs(g) < abs(pg):
            step = min(max(NEWTON_REACH*abs(g)*dy/(abs(pg) - abs(g)), 0.5*dy), 4*dy)
        self.prev, self.length = (x, g), step
        self._probe(self._y(x) + self.dir*step)

    def _far(self):
        '''
        The expansion met an undefined region at x_nan: the bound on the other side of the guess decides whether
        the root lies between the guess and that bound or toward the undefined region (the NaN-end rule of the
        cold search).
        '''
        x = self.hi if self.dir < 0 else self.lo
        if x in (self.x0, self.prev[0]): # the guess is on that bound: the undefined region takes its opposite sign
            return self._bracket(self.x_nan, nan, -self.s0, self.prev[0], self.prev[1], self.s0)
        self.phase, self._pending = 'far', [x]

    def _tell_far(self, items):
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        if math.isnan(g): # undefined beyond both sides: as the cold search with both ends undefined
            return self._finish(UNDEFINED)
        s = float(np.sign(g))
        if s != self.s0:
            return self._bracket(self.x0, self.g0, self.s0, x, g, s)
        self._bracket(self.x_nan, nan, -s, self.prev[0], self.prev[1], self.s0)

    def _tell_other(self, items):
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        s = _sign(g, -self.s0)
        if s != self.s0:
            return self._bracket(self.x0, self.g0, self.s0, x, g, s)
        self._finish(ABOVE if self.s0 > 0 else BELOW, x, g)

    def _bracket(self, x1, g1, s1, x2, g2, s2):
        '''
        Starts the refinement between x1 and x2 (signs s1 != s2, NaN ends included through the NaN-end rule).
        '''
        if x1 > x2:
            x1, g1, s1, x2, g2, s2 = x2, g2, s2, x1, g1, s1
        self.a, self.ga, self.b, self.gb, self.sign_a = x1, g1, x2, g2, s1
        self.nan_sign = s1 if math.isnan(g1) else s2 if math.isnan(g2) else 0.0
        self.ga_s, self.gb_s, self.side = g1, g2, None
        self.w_ref, self.n_illinois = self._y(x2) - self._y(x1), 0 # the width ILLINOIS_STEPS steps must halve
        self.phase = 'refine'
        self._next()

    def _next(self):
        a, b = self.a, self.b
        ya, yb = self._y(a), self._y(b)
        self.kind = 'bisect'
        y = 0.5*(ya + yb)
        interpolate = self.n_illinois < ILLINOIS_STEPS
        if interpolate and math.isfinite(self.ga) and math.isfinite(self.gb) and self.ga_s != self.gb_s:
            yi = (ya*self.gb_s - yb*self.ga_s)/(self.gb_s - self.ga_s)
            margin = ILLINOIS_MARGIN*(yb - ya)
            if ya < yi < yb:
                y, self.kind = min(max(yi, ya + margin), yb - margin), 'illinois'
        self._pending = [self._x(y) if self.kind == 'illinois' or not self.log else math.sqrt(a*b)]

    def _tell_refine(self, items):
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        s = _sign(g, self.nan_sign)
        if s == 0:
            return self._finish(HOLE, x)
        side = 'a' if s == self.sign_a else 'b'
        if side == 'a':
            if self.kind == 'illinois' and self.side == 'a':
                self.gb_s *= 0.5 # b kept twice in a row: the Illinois halving
            self.a, self.ga, self.ga_s = x, g, g
        else:
            if self.kind == 'illinois' and self.side == 'b':
                self.ga_s *= 0.5
            self.b, self.gb, self.gb_s = x, g, g
        after = self._y(self.b) - self._y(self.a)
        if self.kind == 'bisect': # a bisection step resets the Illinois weights
            self.ga_s, self.gb_s, self.side = self.ga, self.gb, None
            self.w_ref, self.n_illinois = after, 0
        else:
            self.side = side
            self.n_illinois += 1
            if after <= 0.5*self.w_ref: # halved: Illinois may go on
                self.w_ref, self.n_illinois = after, 0
        if after < CLOSED*self.width:
            return self._finish(JUMP, x, g)
        self._next()


###_____One search in this process (solve mode)_____###

def find_root(evaluate, theta, st, target, tol, max_evals, guess = None, step = None, orient = 0.0, slope = None):
    '''
    One search along st['index'] of theta, evaluating in this process: evaluate(theta) -> (g, ran, payload).
    Stops with UNCONVERGED when the next evaluations would exceed max_evals. Returns (root, history): history
    lists (phase, x, g, payload) of every evaluation.
    '''
    root = Root(st, target, tol, guess, step, orient, slope)
    theta = np.array(theta, dtype = float)
    history = []
    while root.outcome is None:
        xs = root.pending()
        if root.evals + len(xs) > max_evals:
            root.stop(UNCONVERGED)
            break
        phase, items = root.phase, []
        for x in xs:
            point = theta.copy()
            point[st['index']] = x
            g, ran, payload = evaluate(point)
            items.append((x, g, ran, payload))
            history.append((phase, x, g, payload))
        root.tell(items)
    return root, history


###_____The bisection initializer_____###

def run_bisection(cfg, thetas, pool, st = None):
    '''
    Searches every draw of thetas (n, ndim; physical) along the bisection parameter, from both prior bounds (see
    the module docstring), in synchronized rounds: one pool map per round over every draw still running. Returns a
    dict of arrays over the draws: theta (the draws), point (the final point), x (its parameter value), g, a and b
    (the final bracket), outcome (OUTCOMES codes), rounds, lp and blobs (at the final point; NaN blob when the
    draw stopped without one), exhausted_lo/exhausted_hi, the history of every evaluation (hist_x, hist_value,
    hist_g, hist_status: (n, iters + 2), the two ends first, NaN padded), evaluations and seconds.
    '''
    st = settings(cfg) if st is None else st
    start = time.perf_counter()
    thetas = np.atleast_2d(np.asarray(thetas, dtype = float))
    n, j, name = len(thetas), st['index'], st['observable']
    width = st['iters'] + 2
    out = {'theta': thetas.copy(), 'point': thetas.copy(), 'x': np.full(n, nan), 'g': np.full(n, nan),
           'a': np.full(n, st['lo']), 'b': np.full(n, st['hi']), 'outcome': np.full(n, -1),
           'rounds': np.zeros(n, dtype = int), 'lp': np.full(n, -inf),
           'blobs': np.full(n, np.array(model.NAN_BLOB, dtype = model.blob_dtype())),
           'exhausted_lo': np.zeros(n, dtype = bool), 'exhausted_hi': np.zeros(n, dtype = bool),
           'hist_x': np.full((n, width), nan), 'hist_value': np.full((n, width), nan),
           'hist_g': np.full((n, width), nan), 'hist_status': np.full((n, width), nan)}
    roots = [Root(st, st['target'], st['tol']) for _ in range(n)]
    evaluations = 0

    def at(i, x):
        theta = thetas[i].copy()
        theta[j] = x
        return theta

    for k in range(st['iters'] + 1): # round 0: the two ends
        tasks = [(i, x) for i in range(n) for x in roots[i].pending()]
        if not tasks:
            break
        lp, blobs = model.evaluate_many(pool, [at(i, x) for i, x in tasks], physical = True)
        evaluations += len(tasks)
        g, ran = _g(blobs, st)
        items = {}
        for m, (i, x) in enumerate(tasks):
            column = len(items.get(i, [])) if k == 0 else k + 1
            out['hist_x'][i, column], out['hist_value'][i, column] = x, blobs[name][m]
            out['hist_g'][i, column], out['hist_status'][i, column] = g[m], blobs['status'][m]
            items.setdefault(i, []).append((x, g[m], bool(ran[m]), (lp[m], blobs[m])))
            if k == 0:
                if x == st['lo']:
                    out['exhausted_lo'][i] = blobs['exhausted'][m] == 1
                else:
                    out['exhausted_hi'][i] = blobs['exhausted'][m] == 1
        for i, its in items.items():
            if k > 0:
                out['rounds'][i] = k
            roots[i].tell(its)
    for i, root in enumerate(roots):
        root.stop(UNCONVERGED)
        out['outcome'][i] = root.outcome
        out['x'][i], out['g'][i] = root.x, root.g
        if root.a is not None:
            out['a'][i], out['b'][i] = root.a, root.b
        if root.outcome == CONVERGED:
            lp_i, blob = root.payload
            out['point'][i] = at(i, root.x)
            out['lp'][i], out['blobs'][i] = lp_i, blob
            if not math.isfinite(lp_i):
                out['outcome'][i] = NO_LP
    out['evaluations'] = evaluations
    out['seconds'] = time.perf_counter() - start
    return out


def _concatenate(parts):
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if np.ndim(parts[0][key])}
    out['evaluations'] = sum(p['evaluations'] for p in parts)
    out['seconds'] = sum(p['seconds'] for p in parts)
    return out


def good(result):
    '''
    Indices of the converged draws with a finite log-probability.
    '''
    return np.flatnonzero(result['outcome'] == CONVERGED)


def bisect_points(cfg, pool, n_needed, rng, log = print, draws = None, batches = BATCHES):
    '''
    Batches of Latin-hypercube prior draws (default sampler.bisect_draws, 'auto' = 3*n_needed per batch),
    bisected until at least n_needed have converged with a finite lp, at most `batches` batches. Returns the
    concatenated run_bisection arrays plus settings, batches, draws_per_batch and n_needed. The caller checks
    len(good(result)) and explains a shortfall with failure_text(result).
    '''
    st = settings(cfg)
    if draws is None:
        draws = cfg['sampler']['bisect_draws']
    draws = 3*n_needed if draws == 'auto' else int(draws)
    parts = []
    for batch in range(1, batches + 1):
        thetas = prm.sample_prior(cfg, draws, rng)
        parts.append(run_bisection(cfg, thetas, pool, st))
        result = _concatenate(parts)
        log(summary_line(result, st, batch))
        if good(result).size >= n_needed:
            break
    result.update(settings = st, batches = len(parts), draws_per_batch = draws, n_needed = n_needed)
    return result


def counts(result):
    '''
    {outcome name: number of draws}, plus 'bracketed' and 'exhausted at both ends'.
    '''
    outcome = result['outcome']
    out = {OUTCOMES[k]: int(np.sum(outcome == k)) for k in OUTCOMES}
    out['bracketed'] = int(np.sum(np.isin(outcome, (CONVERGED, HOLE, UNCONVERGED, NO_LP, JUMP))
                                  | ((outcome == FAILED) & (result['rounds'] > 0))))
    out['exhausted at both ends'] = int(np.sum(result['exhausted_lo'] & result['exhausted_hi']))
    return out


def summary_line(result, st, batch = None):
    c = counts(result)
    head = f'bisection of {st["param"]} on {st["observable"]}'
    if batch is not None:
        head += f' (batch {batch})'
    total = ' so far' if batch is not None and batch > 1 else '' # the counts are cumulative over the batches
    return (f"{head}: {len(result['outcome'])} draws{total}, {c['bracketed']} bracketed, {c['converged']} converged, "
            f"{result['evaluations']} evaluations in {result['seconds']:.1f} s")


def no_root_hint(name, outcome, param):
    '''
    What "the model is above (below) the target at both ends" usually means for observable name ('' if unknown).
    '''
    above, below = EXAMPLES.get(name, ('', ''))
    return {ABOVE: above, BELOW: below}.get(outcome, '').format(p = param)


def failure_text(result, n_needed, cfg = None):
    '''
    Why fewer than n_needed draws converged, from the counts, with what to try instead.
    '''
    st = result['settings']
    c, n = counts(result), len(result['outcome'])
    p, k = st['param'], st['observable']
    lines = [f'the bisection of {p} on {k} converged in {c["converged"]} of {n} draws ({result["batches"]} '
             f'batch{"es" if result["batches"] > 1 else ""}), {n_needed} needed:']
    if c[OUTCOMES[ABOVE]]:
        lines.append(f'  - no sign change: g > 0 at both ends in {c[OUTCOMES[ABOVE]]} draws (the model is above the '
                     f'data across the whole {p} prior{no_root_hint(k, ABOVE, p)})')
    if c[OUTCOMES[BELOW]]:
        lines.append(f'  - no sign change: g < 0 at both ends in {c[OUTCOMES[BELOW]]} draws (the model is below the '
                     f'data across the whole {p} prior{no_root_hint(k, BELOW, p)})')
    if c[OUTCOMES[UNDEFINED]]:
        lines.append(f'  - {k} undefined (NaN) at both ends in {c[OUTCOMES[UNDEFINED]]} draws (e.g. a stripped '
                     'atmosphere at both ends)')
    if c['exhausted at both ends']:
        lines.append(f"  - of all draws, {c['exhausted at both ends']} lose the atmosphere at both ends")
    if c[OUTCOMES[FAILED]]:
        lines.append(f'  - the model failed (status 2, 4 or 5) in {c[OUTCOMES[FAILED]]} draws')
    if c[OUTCOMES[HOLE]]:
        lines.append(f'  - {k} undefined inside the bracket in {c[OUTCOMES[HOLE]]} draws')
    if c[OUTCOMES[UNCONVERGED]]:
        lines.append(f"  - not within {st['tol']:g} sigma after {st['iters']} rounds in {c[OUTCOMES[UNCONVERGED]]} "
                     'draws (g may jump across the data instead of crossing it, e.g. at the edge of the stripped '
                     'region)')
    if c[OUTCOMES[JUMP]]:
        lines.append(f'  - g jumps across the data (the bracket closed without a root) in {c[OUTCOMES[JUMP]]} draws')
    if c[OUTCOMES[NO_LP]]:
        lines.append(f'  - converged but with lp = -inf (another observable rules the point out) in '
                     f'{c[OUTCOMES[NO_LP]]} draws')
    lines.append('  Try another sampler.bisect_observable (--bisect-observable) or sampler.bisect_param '
                 '(--bisect-param), widen the prior of the parameter, or see where the data lie with: python -m '
                 f'isofate.mcmc check CONFIG --bisect, and python -m isofate.mcmc scan CONFIG --x {p}=LO:HI:41')
    return '\n'.join(lines)


def choose(result, n, rng):
    '''
    n of the converged draws: a random draw without replacement from the 2n with the highest lp (diversity
    over the best). Returns their indices (fewer if fewer converged).
    '''
    ok = good(result)
    best = ok[np.argsort(-result['lp'][ok], kind = 'stable')][:2*n]
    return rng.permutation(best)[:n]


def save(result, path):
    '''
    bisect.npz: every draw, its bracket, final point, g, lp, blobs, outcome and the evaluation history.
    '''
    st = result['settings']
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays, param = st['param'], observable = st['observable'], target = st['target'],
                        tol = st['tol'], iters = st['iters'], lo = st['lo'], hi = st['hi'],
                        evaluations = result['evaluations'], seconds = result['seconds'],
                        outcome_names = np.array([OUTCOMES[k] for k in sorted(OUTCOMES)]))
    os.replace(tmp, path)
    return path


###_____Text_____###

def _row(label, x, value, g, status, ocfg):
    if not math.isnan(g):
        s = likelihood.sigma_eff(ocfg, above = g > 0)
        g_text, z_text = f'{g:+.4g}', f'{g/s:+.3g}'
    else:
        g_text, z_text = 'NaN', '-'
    status_text = model.STATUS_NAMES.get(int(status), '?') if math.isfinite(status) else '-'
    return f'  {label:<7} {x:>14.7g} {value:>22.5g} {g_text:>16} {z_text:>9}  {status_text}'


def _header(param, name, unit):
    return f'  {"round":<7} {param:>14} {name + unit:>22} {"g = T(m) - target":>16} {"g/sigma":>9}  status'


def _footnote(name, ocfg, target, tol):
    space = 'log10 ' if ocfg['space'] == 'log10' else 'linear '
    lines = [f'  (T is the {space}space of {name}; target = {target:.5g}; converged when |g| <= {tol:g} sigma)']
    if name in WIND_FRACTIONS:
        lines.append('  (NaN: no H/He wind, or H and He carry less than half of the escaping mass)')
    return lines


def iteration_table(result, i, cfg):
    '''
    The evaluations of draw i of run_bisection, one line each: the parameter value, the model value, g, g in
    sigma, the status (check --bisect).
    '''
    st = result['settings'] if 'settings' in result else settings(cfg)
    o = st['ocfg']
    lines = [_header(st['param'], st['observable'], f' [{o["unit"]}]' if o['unit'] else '')]
    for k in range(result['hist_x'].shape[1]):
        x = result['hist_x'][i, k]
        if math.isnan(x):
            continue
        label = 'lo' if k == 0 else 'hi' if k == 1 else str(k - 1)
        lines.append(_row(label, x, result['hist_value'][i, k], result['hist_g'][i, k], result['hist_status'][i, k],
                          o))
    return lines + _footnote(st['observable'], o, st['target'], st['tol'])


def history_table(history, st, name, target, tol):
    '''
    The evaluations of one find_root search (history entries (phase, x, g, result), result an evaluate dict):
    one line each, labelled by phase (lo/hi: the bounds, guess, expand, other: the other bound, then the
    refinement steps 1, 2, ...).
    '''
    o = st['ocfg']
    lines = [_header(st['param'], name, f' [{o["unit"]}]' if o['unit'] else '')]
    step = 0
    for phase, x, g, res in history:
        if phase == 'ends':
            label = 'lo' if x == st['lo'] else 'hi'
        elif phase == 'refine':
            step += 1
            label = str(step)
        else:
            label = phase
        lines.append(_row(label, x, res['obs'].get(name, nan), g, res['status'], o))
    return lines + _footnote(name, o, target, tol)
