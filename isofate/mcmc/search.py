'''
Root finding along one free parameter: the bisection initializer and the solve of solve mode

One root finder (Root) serves both the bisection initializer (sampler.init = "bisect", resolution source =
"bisect", check --bisect) and the solve of solve mode (solve.py). Along a parameter p, inside its prior bounds
[lo, hi], with the other parameters fixed, it drives the model to a target value of one observable k:

    g(p) = T(model value of k at p) - target

T is the space the likelihood compares k in: linear, or log10 with the floor. g and the target are in T units
(dex for a log10 observable, the unit of k for a linear one). The bisection initializer targets the data of
k = sampler.bisect_observable along p = sampler.bisect_param, and stops when |g| <= bisect_tol*sigma, sigma being
the effective sigma of k on the side of the data the model is on. Solve mode targets u, the sampled coordinate,
along solve.param, and stops when |g| <= solve.tol*sigma.

How a search runs:
- Cold start: both bounds are evaluated. A bracket needs a sign change of g between them.
- Warm start (solve mode with solve_table.npz): the table's guess is evaluated, then a bracket is expanded from it
  toward the root until g changes sign. The expected direction comes from the sign of g and of dT/dp. The first
  step is 1.5 Newton steps (|g| over the table's expected slope |dT/dp|): it aims past the root, so that one step
  usually brackets it. A first step that takes g away from the target turns the expansion around. Later steps
  are 1.5 times the secant's distance to the root (within 0.5 and 4 times the previous step), or doubling. At a
  prior bound without a sign change the other bound is tried. When the expansion meets an undefined region, the
  bound on the other side of the guess decides, by the NaN-end rule of the cold search, whether the root lies
  toward that bound or toward the undefined region. An undefined guess gives no direction: the cold start.
- Refinement: each step evaluates a point inside the bracket and keeps the part whose ends differ in sign. The
  point is the midpoint (the geometric mean for a loguniform prior), or, while both ends are finite, the Illinois
  point (modified regula falsi, Dowell & Jarratt 1971). Plain regula falsi slows down when one end of the bracket
  stays put, as it does on a convex g; the Illinois rule halves the g of an end kept twice in a row, which pulls
  the next point toward that end. When 3 Illinois steps in a row have not halved the bracket the next step
  bisects, so the bracket halves at least every 4 steps.
- Stop: when |g| <= tol*sigma (converged); when the bracket is narrower than 1e-9 of the prior width without that
  (g jumps across the target: "jump", no root); or at the evaluation limit (bisect_iters rounds, or
  solve.max_evals).

Rules shared by every use:
- g need not be monotonic: any sign change brackets a root, and each search finds one root (not all of them).
- The NaN-end rule: an undefined value (NaN, e.g. the outflow composition of a stripped planet) takes the sign
  opposite to the defined end of the bracket, i.e. the undefined region is treated as lying on the other side of
  the target. When both ends are defined, a NaN inside the bracket stops the search ("undefined inside the
  bracket"). An infinite g (e.g. log10 of 0 with floor = 0) is defined: it has a sign, but no Illinois step uses
  it.
- The H/He-wind rule: the outflow fractions (outflow_H_fraction, outflow_He_fraction and the legacy escape_H_He,
  escape_He_H) count as undefined where H and He carry less than half of the escaping mass (Mdot_HHe < 0.5 Mdot).
  That is a metal-dominated wind, e.g. at the stripping edge with dynamic_phi = true, where H and He are nearly
  gone and their ratio swings through any value within about 0.01 dex of log_f_atm. It is not the H/He wind the
  p-winds fractions describe, so those points join the stripped side and the search finds the root of the H/He
  wind beyond them. The likelihood itself is unchanged: such points keep their (very low) likelihood when the
  walkers move in the parameter itself.
- A detection targets its value. An upper or lower limit targets the limit value, so the bisection starts the
  walkers on the boundary of the allowed region (the config warns).
- A search ends without a root when g has the same sign at both ends (no sign change), when both ends are
  undefined, at a jump, or when the model fails there. failure_text explains a failed initialization from the
  counts of each case.

The bisection only finds starting points. emcee's moves only combine the positions of the walkers
(Foreman-Mackey et al 2013), so a thin mode that no walker starts in is not found. The bisection puts the walkers
on the thin surface where k fits, e.g. the H-depleted, He-dominated outflow island of LHS 1140 b, which no prior
draw hits by chance. Its evaluations go through a pool's map of model.log_prob_physical (a WorkerPool, the
sampler's counting pool, or model.LocalPool in one process), one map per round over every draw still running. A
solve runs inside one evaluation (find_root). See "Starting the walkers" and "Solve mode" in the README.
'''

import math
import os
import time

import numpy as np

from isofate.mcmc import config, likelihood, model
from isofate.mcmc import params as prm

nan, inf = math.nan, math.inf
RAN = (model.OK, model.NO_LIKELIHOOD) # statuses whose model ran to the end: 0 (ok) and 3 (likelihood -inf)
BATCHES = 3 # the bisection initializer draws at most this many batches before giving up
CLOSED = 1e-9 # a bracket narrower than this fraction of the prior width is closed (a jump, no root)
ILLINOIS_MARGIN = 1e-4 # an Illinois point stays this fraction of the bracket away from its ends
ILLINOIS_STEPS = 3 # Illinois steps that must halve the bracket between them, else the next step bisects
NEWTON_REACH = 1.5 # the first step of a warm start in Newton steps |g|/slope, later ones in secant distances
WIND_FRACTIONS = ('outflow_H_fraction', 'outflow_He_fraction', 'escape_H_He', 'escape_He_H') # need an H/He wind
HHE_DOMINANT = 0.5 # an outflow fraction is defined when H and He carry at least this fraction of the escaping mass

# outcome codes of a search (Root.outcome, and the outcome array of run_bisection)
CONVERGED, ABOVE, BELOW, UNDEFINED, FAILED, HOLE, UNCONVERGED, NO_LP, JUMP = range(9)
# their names, in messages and in bisect.npz (outcome_names)
OUTCOMES = {CONVERGED: 'converged', ABOVE: 'g > 0 at both ends', BELOW: 'g < 0 at both ends',
            UNDEFINED: 'undefined at both ends', FAILED: 'model failed', HOLE: 'undefined inside the bracket',
            UNCONVERGED: 'not converged', NO_LP: 'converged, lp = -inf', JUMP: 'jump across the target'}
NO_ROOT = (ABOVE, BELOW, UNDEFINED, JUMP) # no parameter value inside the prior gives the target
# what "model above / below the data at both ends" usually means, per observable ({p}: the searched parameter)
EXAMPLES = {'Rp': (', e.g. an observed radius below the bare-core radius',
                   ', e.g. an observed radius above what the largest {p} gives'),
            'outflow_H_fraction': (', e.g. an H-rich outflow (He not escaping) at both ends',
                                   ', e.g. an outflow poorer in H than observed at both ends'),
            'fatm': (', e.g. too much atmosphere left even at the smallest {p}', ', e.g. stripped at both ends')}


###_____The settings and g_____###

def bracket(cfg, param):
    '''
    Returns the bracket of a free parameter for the root finder: its prior bounds
    An end that sits on an open hard limit (e.g. log_f_atm < 0) is not a valid input itself, so it is moved 1e-9
    of the box width inside.

    Inputs:
        - cfg: resolved config [dict]
        - param: the free parameter [str]

    Output: dict
        - param: its name [str]
        - index: its index in the physical theta (cfg['free']) [int]
        - lo, hi: its prior bounds (params.prior_box: mu -+ 4 sigma within the hard limits for a normal prior
          without bounds) [unit of param]
        - log: True for a loguniform prior (with lo > 0): the search then works in ln param, with geometric
          midpoints [Bool]
    '''
    j = cfg['free'].index(param)
    lo, hi = (float(x[j]) for x in prm.prior_box(cfg))
    if not prm.in_support(param, lo, cfg): # on an open hard limit: step just inside
        lo += 1e-9*(hi - lo)
    if not prm.in_support(param, hi, cfg):
        hi -= 1e-9*(hi - lo)
    return {'param': param, 'index': j, 'lo': lo, 'hi': hi,
            'log': cfg['params'][param]['prior'] == 'loguniform' and lo > 0}


def settings(cfg):
    '''
    Returns the settings of the bisection initializer
    The bracket of sampler.bisect_param, plus the observable sampler.bisect_observable and its target.

    Inputs:
        - cfg: resolved config [dict]

    Output: dict, bracket() plus
        - observable: the observable k [str]
        - ocfg: its config (space, floor, sigma, kind, ...) [dict]
        - target: its data value in the likelihood space; the limit value for an upper or lower limit [T units]
        - tol: sampler.bisect_tol, the convergence tolerance in units of sigma [ndim]
        - iters: sampler.bisect_iters, the most rounds after the two ends [int]

    Raises ConfigError when the parameter is not free or the observable is not active and computed by the model.
    '''
    param, name = config.bisect_target(cfg)
    ocfg = cfg['observables'][name]
    s = cfg['sampler']
    return dict(bracket(cfg, param), observable = name, ocfg = ocfg, target = likelihood.observed(ocfg),
                tol = float(s['bisect_tol']), iters = int(s['bisect_iters']))


def search_values(values, name):
    '''
    Returns the values of an observable as the root finder sees them
    The H/He-wind rule (see the module docstring): an outflow fraction is NaN, undefined, where H and He carry
    less than half of the escaping mass (Mdot_HHe < 0.5 Mdot). Other observables come back as they are.

    Inputs:
        - values: model values, a blob array (structured) or an obs dict [array or dict]
        - name: the observable [str]

    Output: its values, NaN where undefined [array]
    '''
    x = np.asarray(values[name], dtype = float)
    if name in WIND_FRACTIONS:
        with np.errstate(invalid = 'ignore'): # a NaN rate compares False: undefined
            mdot = np.asarray(values['Mdot'], dtype = float)
            hhe = np.asarray(values['Mdot_HHe'], dtype = float) >= HHE_DOMINANT*mdot # an H/He wind
        x = np.where(hhe, x, nan)
    return x


def g_of(values, name, ocfg, target):
    '''
    Calculates g = T(model value) - target of an observable
    T is the likelihood space of the observable (likelihood.to_space: linear, or log10 clipped to the floor and
    the ceiling). With target = 0 it gives T itself (solve.to_sampled, scan).

    Inputs:
        - values: model values, a blob array (structured) or an obs dict [array or dict]
        - name: the observable [str]
        - ocfg: its config [dict]
        - target: the target value of T [T units]

    Output: g, NaN where undefined (the H/He-wind rule included), +-inf where T is infinite [array]
    '''
    x = np.atleast_1d(search_values(values, name))
    T = np.array([likelihood.to_space(v, ocfg) for v in x], dtype = float)
    return T - target


def _g(blobs, st):
    '''
    g of each blob for the bisection settings st, and whether its model ran (status 0 or 3)
    '''
    return g_of(blobs, st['observable'], st['ocfg'], st['target']), np.isin(blobs['status'], RAN)


def within(g, ocfg, tol):
    '''
    Tests convergence: |g| <= tol*sigma, with sigma on the side of the target the model is on
    A non-finite g (NaN or +-inf) never converges.

    Inputs:
        - g: T - target [T units]
        - ocfg: config of the observable (its sigma, asymmetric or not, and sigma_model) [dict]
        - tol: the tolerance in units of sigma [ndim]

    Output: True when converged [Bool]
    '''
    if not math.isfinite(g):
        return False
    return abs(g) <= tol*likelihood.sigma_eff(ocfg, above = g > 0)


def _within(g, st):
    '''
    within() with the observable and the tolerance of the settings st
    '''
    return within(g, st['ocfg'], st['tol'])


def _sign(g, nan_sign):
    '''
    The sign of g as a float, or nan_sign where g is NaN (the NaN-end rule)
    '''
    return nan_sign if math.isnan(g) else float(np.sign(g))


###_____The root finder_____###

class Root:
    '''
    One root search of g along a parameter in [lo, hi] (see the module docstring), driven from outside
    The caller runs the model, so the same object serves a search in one process (find_root) and many searches
    stepped together through a pool, one map per round (run_bisection):

        root = Root(st, target, tol)               # cold; Root(..., guess = p0, step = dp, orient = +1) warm
        while root.outcome is None:
            xs = root.pending()                    # one or two parameter values to evaluate now
            root.tell([(x, g, ran, payload), ...]) # their g, whether the model ran, anything to keep (e.g. a blob)

    The search works in y = ln p for a loguniform prior (steps, midpoints and Illinois points), else in y = p.

    Attributes, at the end:
        - outcome: one of the OUTCOMES codes; None while the search runs [int]
        - x, g: the root; without one, the last point evaluated or the guess (NaN for an outcome decided at the
          two ends, and for UNDEFINED) [unit of p, T units]
        - payload: what tell received with the root (e.g. (lp, blob), or a model.evaluate result); None without a
          root
        - a, b: the bracket, a < b; None without one [unit of p]
        - sign_a: the sign of g at a; dT/dp has the sign -sign_a across the bracket [ndim]
        - nan_sign: the sign an undefined g takes inside the bracket; 0 when both ends are defined [ndim]
        - evals: evaluations told so far [int]
        - phase: 'ends', 'guess', 'expand', 'other', 'far' or 'refine' [str]
    '''

    def __init__(self, st, target, tol, guess = None, step = None, orient = 0.0, slope = None):
        '''
        Starts a search: cold, from both bounds, or warm, from a guess

        Inputs:
            - st: settings with lo, hi, log (bracket()) and ocfg [dict]
            - target: the target value of T, kept for the record: the g the caller tells must already be
              T - target [T units]
            - tol: the convergence tolerance in units of sigma [ndim]
            - guess: starting value of p, clipped to [lo, hi]; None or not finite: a cold start [unit of p]
            - step: first step of the expansion; None or 0: 1e-3 of the prior width [unit of y]
            - orient: the expected sign of dT/dp; 0 (unknown) counts as +1 [ndim]
            - slope: the expected |dT/dy| (per unit of ln p for a loguniform prior); None, 0 or not finite:
              unknown, and the first step is step [T units per unit of y]
        '''
        self.ocfg, self.target, self.tol = st['ocfg'], float(target), float(tol)
        self.lo, self.hi, self.log = st['lo'], st['hi'], bool(st['log'])
        self.width = self._y(self.hi) - self._y(self.lo) # the prior width [in y]
        self.outcome, self.x, self.g, self.payload = None, nan, nan, None
        self.a = self.b = None
        self.sign_a, self.nan_sign, self.evals = 0.0, 0.0, 0
        if guess is None or not math.isfinite(guess): # cold: both bounds at once
            self.phase, self._pending = 'ends', [self.lo, self.hi]
        else:
            self.phase, self._pending = 'guess', [min(max(float(guess), self.lo), self.hi)]
            self.step = abs(float(step)) if step else 1e-3*self.width # [in y]
            self.orient = 1.0 if orient >= 0 else -1.0 # expected sign of dT/dp
            self.slope = abs(float(slope)) if slope and math.isfinite(slope) else None # expected |dT/dy|

    def _y(self, x):
        '''
        The search coordinate of a parameter value: ln x for a loguniform prior, else x
        '''
        return math.log(x) if self.log else x

    def _x(self, y):
        '''
        The parameter value of a search coordinate (the inverse of _y)
        '''
        return math.exp(y) if self.log else y

    def pending(self):
        '''
        Returns the parameter values to evaluate now: the two bounds in the 'ends' phase, else one point

        Output: the values; [] once the search has ended [list]
        '''
        return [] if self.outcome is not None else list(self._pending)

    def stop(self, outcome):
        '''
        Ends the search from outside (e.g. at the evaluation limit), keeping the last point
        Does nothing once the search has ended.

        Inputs:
            - outcome: the outcome to end with, e.g. UNCONVERGED [int]
        '''
        if self.outcome is None:
            self.outcome = outcome

    def _finish(self, outcome, x = nan, g = nan, payload = None):
        '''
        Ends the search with outcome at the point (x, g, payload)
        '''
        self.outcome, self.x, self.g, self.payload = outcome, x, g, payload
        self._pending = []

    def _fits(self, g):
        '''
        Whether g has converged (within)
        '''
        return within(g, self.ocfg, self.tol)

    def tell(self, items):
        '''
        Takes the results of the pending evaluations and moves the search on
        The handler of the current phase (_tell_ends, _tell_guess, ...) sets the next pending values or ends the
        search.

        Inputs:
            - items: one (x, g, ran, payload) per pending value, in the order of pending() [list]
                - x: the parameter value [unit of p]
                - g: T - target there; NaN where undefined [T units]
                - ran: whether the model ran to the end (status 0 or 3) [Bool]
                - payload: anything to keep with the point; it becomes root.payload when the point is the root
        '''
        self.evals += len(items)
        getattr(self, f'_tell_{self.phase}')(items)

    ###_____phases_____###

    def _tell_ends(self, items):
        '''
        Phase 'ends': both bounds evaluated (a cold start)
        Converges on an end that already fits; else brackets the root where g changes sign between the ends, an
        undefined end taking the sign opposite to the defined one (the NaN-end rule). Ends with FAILED when the
        model failed at an end, UNDEFINED when both ends are NaN, and ABOVE or BELOW without a sign change.
        '''
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
        # the NaN-end rule: an undefined end takes the sign opposite to the defined one
        s_lo = float(np.sign(g_lo)) if defined_lo else -float(np.sign(g_hi))
        s_hi = float(np.sign(g_hi)) if defined_hi else -float(np.sign(g_lo))
        if s_lo == s_hi:
            return self._finish(ABOVE if s_lo > 0 else BELOW)
        self._bracket(x_lo, g_lo, s_lo, x_hi, g_hi, s_hi)

    def _tell_guess(self, items):
        '''
        Phase 'guess': the guess of a warm start evaluated
        Converges there if it fits. An undefined guess gives no direction, so the search falls back to the cold
        start. Else the expansion starts toward the root, with 1.5 Newton steps when the slope is known.
        '''
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        if math.isnan(g): # no direction to go: the cold search
            self.phase, self._pending = 'ends', [self.lo, self.hi]
            return
        self.x0, self.g0, self.s0 = x, g, float(np.sign(g)) # the guess, and the sign of g there
        self.dir = -self.s0*self.orient # toward the root if T(p) is monotonic with the expected orientation
        if self.slope and math.isfinite(g): # 1.5 Newton steps, within 1/16 and 16 times the table's step
            self.step = min(max(NEWTON_REACH*abs(g)/self.slope, self.step/16), 16*self.step)
        # the last point, the step, whether the expansion has turned around, where it met an undefined region
        self.prev, self.length, self.reversed, self.x_nan = (x, g), self.step, False, None
        self._probe(self._y(x) + self.dir*self.length)

    def _probe(self, y):
        '''
        Queues y as the next point of the expansion, clipped to the prior bounds
        When the expansion already stands on the bound it would step to, the other bound is tried (_other).
        '''
        x = self.hi if y >= self._y(self.hi) else self.lo if y <= self._y(self.lo) else self._x(y)
        if x == self.prev[0]: # already on the bound it would step to
            return self._other()
        self.phase, self._pending = 'expand', [x]

    def _other(self):
        '''
        Queues the bound on the other side of the guess: the expansion reached a bound without a sign change
        The root, if any, lies on the other side of the guess. When the guess is itself on that bound, g has the
        same sign everywhere the search looked: ABOVE or BELOW.
        '''
        x = self.hi if self.dir < 0 else self.lo
        if x == self.x0:
            return self._finish(ABOVE if self.s0 > 0 else BELOW, self.x0, self.g0)
        self.phase, self._pending = 'other', [x]

    def _tell_expand(self, items):
        '''
        Phase 'expand': one step of the bracket expansion evaluated
        Brackets the root at a sign change. A first step that took g away from the target turns the expansion
        around (once); at a prior bound the other bound is tried; an undefined point lets the far bound decide
        (_far). Else the expansion steps on: 1.5 times the secant's distance to the root (within 0.5 and 4 times
        the last step) while |g| shrinks, doubling otherwise.
        '''
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
        if s != self.s0: # a sign change: the root lies between the last two points
            return self._bracket(px, pg, self.s0, x, g, s)
        if not self.reversed and px == self.x0 and abs(g) > abs(pg): # the first step went away from the target
            self.reversed, self.dir = True, -self.dir
            return self._probe(self._y(self.x0) + self.dir*self.length)
        if x in (self.lo, self.hi):
            return self._other()
        dy = abs(self._y(x) - self._y(px)) # the last step, as taken [in y]
        step = 2*dy # doubling, or 1.5 times the secant's distance to the root (within 0.5 to 4 steps)
        if abs(g) < abs(pg): # the secant through the last two |g| reaches 0 at |g| dy/(|pg| - |g|) beyond x
            step = min(max(NEWTON_REACH*abs(g)*dy/(abs(pg) - abs(g)), 0.5*dy), 4*dy)
        self.prev, self.length = (x, g), step
        self._probe(self._y(x) + self.dir*step)

    def _far(self):
        '''
        Handles an undefined region met by the expansion at x_nan, by the NaN-end rule of the cold search
        The bound on the other side of the guess decides whether the root lies between the guess and that bound,
        or toward the undefined region. When the guess is itself on that bound, the undefined region takes the
        sign opposite to the guess's, and the bracket runs from the last defined point to x_nan.
        '''
        x = self.hi if self.dir < 0 else self.lo
        if x in (self.x0, self.prev[0]): # the guess is on that bound: the undefined region takes its opposite sign
            return self._bracket(self.x_nan, nan, -self.s0, self.prev[0], self.prev[1], self.s0)
        self.phase, self._pending = 'far', [x]

    def _tell_far(self, items):
        '''
        Phase 'far': the bound on the other side of the guess evaluated, after the expansion met an undefined region
        A sign change there brackets the root between the guess and that bound. The guess's sign there puts the
        root toward the undefined region, which takes the opposite sign (the NaN-end rule): the bracket runs from
        the last defined point of the expansion to x_nan. Undefined there too: UNDEFINED, as a cold search with
        both ends undefined.
        '''
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
        '''
        Phase 'other': the bound on the other side of the guess evaluated, after the expansion reached a bound
        A sign change there brackets the root between the guess and that bound; else g has the same sign
        everywhere the search looked: ABOVE or BELOW.
        '''
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        s = _sign(g, -self.s0) # the NaN-end rule: an undefined bound takes the sign opposite to the guess's
        if s != self.s0:
            return self._bracket(self.x0, self.g0, self.s0, x, g, s)
        self._finish(ABOVE if self.s0 > 0 else BELOW, x, g)

    ###_____refinement inside the bracket_____###

    def _bracket(self, x1, g1, s1, x2, g2, s2):
        '''
        Starts the refinement between x1 and x2, whose signs differ (s1 != s2)
        An end may be NaN, with its sign from the NaN-end rule: an undefined point inside the bracket then takes
        that sign (nan_sign). With both ends defined nan_sign is 0, and an undefined point inside is a HOLE.
        '''
        if x1 > x2: # a < b
            x1, g1, s1, x2, g2, s2 = x2, g2, s2, x1, g1, s1
        self.a, self.ga, self.b, self.gb, self.sign_a = x1, g1, x2, g2, s1
        self.nan_sign = s1 if math.isnan(g1) else s2 if math.isnan(g2) else 0.0
        # the g of each end in the Illinois step (halved while the end is kept), and the end replaced last
        self.ga_s, self.gb_s, self.side = g1, g2, None
        self.w_ref, self.n_illinois = self._y(x2) - self._y(x1), 0 # the width ILLINOIS_STEPS steps must halve
        self.phase = 'refine'
        self._next()

    def _next(self):
        '''
        Queues the next point inside the bracket [a, b]
        The Illinois point while both ends are finite and fewer than ILLINOIS_STEPS Illinois steps have gone by
        without halving the bracket, kept ILLINOIS_MARGIN of the bracket away from its ends; else the midpoint in
        y (the geometric mean of a and b for a loguniform prior).
        '''
        a, b = self.a, self.b
        ya, yb = self._y(a), self._y(b)
        self.kind = 'bisect'
        y = 0.5*(ya + yb)
        interpolate = self.n_illinois < ILLINOIS_STEPS
        if interpolate and math.isfinite(self.ga) and math.isfinite(self.gb) and self.ga_s != self.gb_s:
            yi = (ya*self.gb_s - yb*self.ga_s)/(self.gb_s - self.ga_s) # the zero of the line through both ends
            margin = ILLINOIS_MARGIN*(yb - ya)
            if ya < yi < yb:
                y, self.kind = min(max(yi, ya + margin), yb - margin), 'illinois'
        # a loguniform bisection takes the geometric mean sqrt(a b) directly
        self._pending = [self._x(y) if self.kind == 'illinois' or not self.log else math.sqrt(a*b)]

    def _tell_refine(self, items):
        '''
        Phase 'refine': a point inside the bracket evaluated
        Converges if it fits; else the point replaces the end with its sign (an undefined point takes nan_sign,
        or ends the search as a HOLE when both ends are defined). The Illinois rule halves the g of an end kept
        twice in a row; a bisection step resets those weights. A bracket narrower than CLOSED of the prior width
        ends as a JUMP: g jumps across the target there instead of crossing it.
        '''
        ((x, g, ran, p),) = items
        self.x, self.g = x, g
        if not ran:
            return self._finish(FAILED, x)
        if self._fits(g):
            return self._finish(CONVERGED, x, g, p)
        s = _sign(g, self.nan_sign)
        if s == 0: # undefined inside a bracket whose ends are both defined
            return self._finish(HOLE, x)
        side = 'a' if s == self.sign_a else 'b' # the end with the same sign is replaced
        if side == 'a':
            if self.kind == 'illinois' and self.side == 'a':
                self.gb_s *= 0.5 # b kept twice in a row: the Illinois halving
            self.a, self.ga, self.ga_s = x, g, g
        else:
            if self.kind == 'illinois' and self.side == 'b':
                self.ga_s *= 0.5 # a kept twice in a row
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
        if after < CLOSED*self.width: # closed without a root, e.g. a step in T at the edge of the stripped region
            return self._finish(JUMP, x, g)
        self._next()


###_____One search in this process (solve mode)_____###

def find_root(evaluate, theta, st, target, tol, max_evals, guess = None, step = None, orient = 0.0, slope = None):
    '''
    Runs one root search in this process (solve mode, check, tests)
    Evaluates the pending values one at a time with the caller's evaluate, at theta with the searched parameter
    (st['index']) set, until the search ends. Stops with UNCONVERGED when the next evaluations would exceed
    max_evals.

    Inputs:
        - evaluate: theta -> (g, ran, payload), g = T - target [function]
        - theta: the physical theta; its st['index'] entry is ignored [array]
        - st: settings with index, lo, hi, log and ocfg (bracket() plus ocfg) [dict]
        - target, tol, guess, step, orient, slope: as in Root
        - max_evals: the most evaluations [int]

    Output: (root, history)
        - root: the ended search [Root]
        - history: one (phase, x, g, payload) per evaluation, in order [list]
    '''
    root = Root(st, target, tol, guess, step, orient, slope)
    theta = np.array(theta, dtype = float)
    history = []
    while root.outcome is None:
        xs = root.pending()
        if root.evals + len(xs) > max_evals:
            root.stop(UNCONVERGED)
            break
        phase, items = root.phase, [] # the phase that asked for these points, for the history
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
    Runs the bisection of many draws together, through a pool
    Every draw of thetas is searched along the bisection parameter from both prior bounds (a cold start; see the
    module docstring), in synchronized rounds: one pool map per round over every draw still running. Round 0
    evaluates the two ends, then come at most st['iters'] rounds of refinement.

    Inputs:
        - cfg: resolved config [dict]
        - thetas: the draws (n, ndim), physical [array]
        - pool: anything with a map of model.log_prob_physical (a WorkerPool, the sampler's counting pool,
          model.LocalPool)
        - st: the bisection settings; None: settings(cfg) [dict]

    Output: dict of arrays over the draws, plus two totals
        - theta: the draws (n, ndim) [array]
        - point: the converged point (NO_LP too); the draw itself otherwise (n, ndim) [array]
        - x, g: the root, or the last point evaluated (NaN: none) [unit of p, T units]
        - a, b: the final bracket; the prior bounds without one [unit of p]
        - outcome: OUTCOMES codes [int]
        - rounds: refinement rounds the draw took (0: the ends only) [int]
        - lp, blobs: log-probability and blob at the converged point (NO_LP too); -inf and a NaN blob otherwise
        - exhausted_lo, exhausted_hi: the atmosphere is lost at the lower (upper) bound [Bool]
        - hist_x, hist_value, hist_g, hist_status: every evaluation, (n, iters + 2): the parameter value, the model
          value of the observable, g and the status; the two ends first, NaN padded
        - evaluations: model evaluations in all [int]
        - seconds: wall time [s]
    '''
    st = settings(cfg) if st is None else st
    start = time.perf_counter()
    thetas = np.atleast_2d(np.asarray(thetas, dtype = float))
    n, j, name = len(thetas), st['index'], st['observable']
    width = st['iters'] + 2 # history columns: the two ends, then one per round
    out = {'theta': thetas.copy(), 'point': thetas.copy(), 'x': np.full(n, nan), 'g': np.full(n, nan),
           'a': np.full(n, st['lo']), 'b': np.full(n, st['hi']), 'outcome': np.full(n, -1),
           'rounds': np.zeros(n, dtype = int), 'lp': np.full(n, -inf),
           'blobs': np.full(n, np.array(model.NAN_BLOB, dtype = model.blob_dtype())),
           'exhausted_lo': np.zeros(n, dtype = bool), 'exhausted_hi': np.zeros(n, dtype = bool),
           'hist_x': np.full((n, width), nan), 'hist_value': np.full((n, width), nan),
           'hist_g': np.full((n, width), nan), 'hist_status': np.full((n, width), nan)}
    roots = [Root(st, st['target'], st['tol']) for _ in range(n)] # cold searches
    evaluations = 0

    def at(i, x):
        '''
        Draw i with the bisection parameter at x
        '''
        theta = thetas[i].copy()
        theta[j] = x
        return theta

    for k in range(st['iters'] + 1): # round 0: the two ends
        tasks = [(i, x) for i in range(n) for x in roots[i].pending()] # every draw still running
        if not tasks:
            break
        lp, blobs = model.evaluate_many(pool, [at(i, x) for i, x in tasks], physical = True) # one map per round
        evaluations += len(tasks)
        g, ran = _g(blobs, st)
        items = {}
        for m, (i, x) in enumerate(tasks):
            column = len(items.get(i, [])) if k == 0 else k + 1 # lo, hi in columns 0, 1; round k in column k + 1
            out['hist_x'][i, column], out['hist_value'][i, column] = x, blobs[name][m]
            out['hist_g'][i, column], out['hist_status'][i, column] = g[m], blobs['status'][m]
            items.setdefault(i, []).append((x, g[m], bool(ran[m]), (lp[m], blobs[m]))) # payload: (lp, blob)
            if k == 0: # whether the atmosphere is lost at each end, for failure_text
                if x == st['lo']:
                    out['exhausted_lo'][i] = blobs['exhausted'][m] == 1
                else:
                    out['exhausted_hi'][i] = blobs['exhausted'][m] == 1
        for i, its in items.items():
            if k > 0:
                out['rounds'][i] = k
            roots[i].tell(its)
    for i, root in enumerate(roots):
        root.stop(UNCONVERGED) # still running after the last round
        out['outcome'][i] = root.outcome
        out['x'][i], out['g'][i] = root.x, root.g
        if root.a is not None:
            out['a'][i], out['b'][i] = root.a, root.b
        if root.outcome == CONVERGED:
            lp_i, blob = root.payload
            out['point'][i] = at(i, root.x)
            out['lp'][i], out['blobs'][i] = lp_i, blob
            if not math.isfinite(lp_i): # k fits, but another observable (or the prior) rules the point out
                out['outcome'][i] = NO_LP
    out['evaluations'] = evaluations
    out['seconds'] = time.perf_counter() - start
    return out


def _concatenate(parts):
    '''
    Joins the run_bisection results of several batches: the arrays concatenated, evaluations and seconds summed
    '''
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0] if np.ndim(parts[0][key])}
    out['evaluations'] = sum(p['evaluations'] for p in parts)
    out['seconds'] = sum(p['seconds'] for p in parts)
    return out


def good(result):
    '''
    Returns the indices of the converged draws
    They all have a finite log-probability: a converged draw with lp = -inf is NO_LP.

    Inputs:
        - result: run_bisection or bisect_points result [dict]

    Output: the indices [array]
    '''
    return np.flatnonzero(result['outcome'] == CONVERGED)


def bisect_points(cfg, pool, n_needed, rng, log = print, draws = None, batches = BATCHES):
    '''
    Finds starting points by bisection: batches of prior draws, bisected until enough converge
    Each batch is Latin-hypercube prior draws, bisected with run_bisection, until at least n_needed have
    converged with a finite lp, at most `batches` batches. Logs one summary line per batch. The caller checks
    len(good(result)) and explains a shortfall with failure_text(result).

    Inputs:
        - cfg: resolved config [dict]
        - pool: anything with a map of model.log_prob_physical
        - n_needed: converged points needed, e.g. nwalkers [int]
        - rng: numpy Generator or seed
        - log: where the summary lines go; default print [function]
        - draws: draws per batch; None: sampler.bisect_draws, whose 'auto' is 3*n_needed [int or str]
        - batches: the most batches; default BATCHES (3) [int]

    Output: the run_bisection arrays of all batches, concatenated, plus
        - settings: the bisection settings (settings()) [dict]
        - batches: batches drawn [int]
        - draws_per_batch: draws per batch [int]
        - n_needed: as given [int]
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


###_____Counts and messages_____###

def counts(result):
    '''
    Counts the outcomes of a bisection

    Inputs:
        - result: run_bisection or bisect_points result [dict]

    Output: dict of numbers of draws [int]
        - one key per outcome name (OUTCOMES)
        - bracketed: every draw but those with no sign change, undefined at both ends, or a model failure at the
          ends
        - exhausted at both ends: draws that lose the atmosphere at both prior bounds
    '''
    outcome = result['outcome']
    out = {OUTCOMES[k]: int(np.sum(outcome == k)) for k in OUTCOMES}
    out['bracketed'] = int(np.sum(np.isin(outcome, (CONVERGED, HOLE, UNCONVERGED, NO_LP, JUMP))
                                  | ((outcome == FAILED) & (result['rounds'] > 0))))
    out['exhausted at both ends'] = int(np.sum(result['exhausted_lo'] & result['exhausted_hi']))
    return out


def summary_line(result, st, batch = None):
    '''
    Returns the summary line of a bisection: draws, bracketed, converged, evaluations and time

    Inputs:
        - result: run_bisection or bisect_points result [dict]
        - st: the bisection settings [dict]
        - batch: the batch number, for the label; None: no label [int]

    Output: 'bisection of P on k (batch b): N draws [so far], B bracketed, C converged, E evaluations in S s' [str]
    '''
    c = counts(result)
    head = f'bisection of {st["param"]} on {st["observable"]}'
    if batch is not None:
        head += f' (batch {batch})'
    total = ' so far' if batch is not None and batch > 1 else '' # the counts are cumulative over the batches
    return (f"{head}: {len(result['outcome'])} draws{total}, {c['bracketed']} bracketed, {c['converged']} converged, "
            f"{result['evaluations']} evaluations in {result['seconds']:.1f} s")


def no_root_hint(name, outcome, param):
    '''
    Returns what "the model is above (below) the target at both ends" usually means for an observable

    Inputs:
        - name: the observable [str]
        - outcome: ABOVE or BELOW; any other gives '' [int]
        - param: the searched parameter, for the text [str]

    Output: e.g. ', e.g. stripped at both ends'; '' for an observable without examples [str]
    '''
    above, below = EXAMPLES.get(name, ('', ''))
    return {ABOVE: above, BELOW: below}.get(outcome, '').format(p = param)


def failure_text(result, n_needed, cfg = None):
    '''
    Explains why fewer than n_needed draws converged, from the counts, with what to try instead

    Inputs:
        - result: bisect_points result (with settings and batches) [dict]
        - n_needed: converged points needed [int]
        - cfg: not used [dict]

    Output: the explanation, one line per case [str]
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


###_____Choosing the starting points, and bisect.npz_____###

def choose(result, n, rng):
    '''
    Chooses n of the converged draws to start the walkers from
    A random draw without replacement from the 2n with the highest lp: diversity over the best.

    Inputs:
        - result: bisect_points result [dict]
        - n: points to choose, e.g. nwalkers [int]
        - rng: numpy Generator

    Output: their indices; fewer when fewer converged [array]
    '''
    ok = good(result)
    best = ok[np.argsort(-result['lp'][ok], kind = 'stable')][:2*n]
    return rng.permutation(best)[:n]


def save(result, path):
    '''
    Saves a bisection to bisect.npz: every draw, its bracket, final point, g, lp, blobs, outcome and history
    Every array of result is saved, plus the settings (param, observable, target, tol, iters, lo, hi),
    evaluations, seconds and outcome_names (the OUTCOMES names, indexed by the outcome codes). It is written to a
    temporary file first and then moved into place, so a crash never leaves a partial bisect.npz.

    Inputs:
        - result: bisect_points result [dict]
        - path: the output file [str]

    Output: path [str]
    '''
    st = result['settings']
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays, param = st['param'], observable = st['observable'], target = st['target'],
                        tol = st['tol'], iters = st['iters'], lo = st['lo'], hi = st['hi'],
                        evaluations = result['evaluations'], seconds = result['seconds'],
                        outcome_names = np.array([OUTCOMES[k] for k in sorted(OUTCOMES)]))
    os.replace(tmp, path) # atomic on one file system
    return path


###_____Text_____###

def _row(label, x, value, g, status, ocfg):
    '''
    One line of an iteration table: label, parameter value, model value, g, g in sigma and status
    '''
    if not math.isnan(g):
        s = likelihood.sigma_eff(ocfg, above = g > 0) # sigma on the side of the target the model is on
        g_text, z_text = f'{g:+.4g}', f'{g/s:+.3g}'
    else:
        g_text, z_text = 'NaN', '-'
    status_text = model.STATUS_NAMES.get(int(status), '?') if math.isfinite(status) else '-'
    return f'  {label:<7} {x:>14.7g} {value:>22.5g} {g_text:>16} {z_text:>9}  {status_text}'


def _header(param, name, unit):
    '''
    The header line of an iteration table
    '''
    return f'  {"round":<7} {param:>14} {name + unit:>22} {"g = T(m) - target":>16} {"g/sigma":>9}  status'


def _footnote(name, ocfg, target, tol):
    '''
    The notes under an iteration table: the space of T, the target and the tolerance, and what NaN means for an
    outflow fraction
    '''
    space = 'log10 ' if ocfg['space'] == 'log10' else 'linear '
    lines = [f'  (T is the {space}space of {name}; target = {target:.5g}; converged when |g| <= {tol:g} sigma)']
    if name in WIND_FRACTIONS:
        lines.append('  (NaN: no H/He wind, or H and He carry less than half of the escaping mass)')
    return lines


def iteration_table(result, i, cfg):
    '''
    Returns the evaluations of draw i of a bisection as a table, one line each (check --bisect)
    Each line holds the round (lo, hi, then 1, 2, ...), the parameter value, the model value, g, g in sigma and
    the status.

    Inputs:
        - result: run_bisection result [dict]
        - i: the draw [int]
        - cfg: resolved config, for the settings when result has none [dict]

    Output: the lines [list of str]
    '''
    st = result['settings'] if 'settings' in result else settings(cfg)
    o = st['ocfg']
    lines = [_header(st['param'], st['observable'], f' [{o["unit"]}]' if o['unit'] else '')]
    for k in range(result['hist_x'].shape[1]):
        x = result['hist_x'][i, k]
        if math.isnan(x): # NaN padding: the draw ended earlier
            continue
        label = 'lo' if k == 0 else 'hi' if k == 1 else str(k - 1)
        lines.append(_row(label, x, result['hist_value'][i, k], result['hist_g'][i, k], result['hist_status'][i, k],
                          o))
    return lines + _footnote(st['observable'], o, st['target'], st['tol'])


def history_table(history, st, name, target, tol):
    '''
    Returns the evaluations of one find_root search as a table, one line each (check in solve mode)
    The lines are labelled by phase: lo and hi (the bounds of a cold start), guess, expand, other (the other
    bound), far (the far bound after an undefined region), then the refinement steps 1, 2, ...

    Inputs:
        - history: the history of find_root: (phase, x, g, result) entries, result a model.evaluate dict [list]
        - st: settings with param, lo and ocfg [dict]
        - name: the observable [str]
        - target: the target value of T, for the footnote [T units]
        - tol: the tolerance in units of sigma, for the footnote [ndim]

    Output: the lines [list of str]
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
