'''
One model evaluation for the MCMC: the isocalc inputs of a parameter set, the observables of the final state,
the blob schema, evaluate(), and the module-level functions a worker pool calls (init_worker, log_prob,
eval_task).

Status codes, stored as floats in every blob:
    0  ok
    1  outside the prior (model not run)
    2  expected model failure (ValueError, RuntimeError, FloatingPointError, ArithmeticError, LinAlgError,
       JAX/equinox runtime errors raised by isocalc)
    3  log-likelihood -inf from the data (e.g. a detected species in a stripped atmosphere), or non-finite
    4  invalid inputs (e.g. N_H <= 0 in the legacy composition)
    5  unexpected exception, i.e. a bug (KeyError, TypeError, ...); the traceback goes to
       <run_dir>/worker_errors/<pid>.log
In solve mode (solve.py) status 1 also means that no value of the solved parameter inside its prior gives the
sampled u, and status 3 a solve that did not converge or a derivative that cannot be used.

The coupled model (jax, atmodeller, numba) is imported lazily, so importing this module is safe in the parent.
'''

import copy
import math
import os
import sys
import threading
import time
import traceback

import numpy as np

from isofate import constants as c
from isofate.orbit_params import EqTemp
from isofate.mcmc import config, env, likelihood
from isofate.mcmc import params as prm
from isofate.mcmc.params import centre_theta

nan, inf = math.nan, math.inf

OK, OUTSIDE_PRIOR, MODEL_FAILED, NO_LIKELIHOOD, INVALID_INPUTS, UNEXPECTED = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
STATUS_NAMES = {0: 'ok', 1: 'outside prior', 2: 'model failure', 3: 'likelihood -inf', 4: 'invalid inputs',
                5: 'unexpected error'}
EXPECTED_ERRORS = (ValueError, RuntimeError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError)

SPECIES = ('H', 'He', 'D', 'O', 'C', 'N', 'S') # escape kernel order
MU = {X: getattr(c, f'mu_{X}') for X in SPECIES} # atomic masses [kg/atom]
MOLECULES = config.VMR_MOLECULES # H2, He, H2O, CO, CO2, CH4, N2, O2, S2, SO2, H2SO4
GASES = tuple(mol for mol in MOLECULES if mol != 'He') # the 10 Atmodeller gases
ISOCALC_NAME = {mol: 'H2O4S' if mol == 'H2SO4' else mol for mol in GASES} # names in isocalc's output keys
MOLAR_MASS = {'H2': 2.016, 'He': 4.0026, 'H2O': 18.015, 'CO': 28.010, 'CO2': 44.009, 'CH4': 16.043, 'N2': 28.014,
              'O2': 31.998, 'S2': 64.13, 'SO2': 64.066, 'H2SO4': 98.079} # [g/mol]

EXTRAS = ('exhausted', 't_lost', 'Matm', 'subcritical', 'n_atmod_calls') # T_surf is an observable already
OBS_FIELDS = tuple(config.OBSERVABLES) + EXTRAS
# solve mode (solve.py): solve_value = the solved parameter P* (NaN without solve mode), solve_evals = model runs
# of the evaluation (0 or 1 without solve mode), log_jac = ln|dP*/du| (0 without solve mode: the identity)
SOLVE_FIELDS = ('solve_value', 'solve_evals', 'log_jac')
BLOB_FIELDS = ('status', 'runtime', 'log_prior', 'log_like') + OBS_FIELDS + SOLVE_FIELDS
NAN_BLOB = (nan,)*len(BLOB_FIELDS)

# posterior tracks (tracks.py): time series on a common grid. N_X in float64: atom numbers overflow float32
TRACK_KEYS = (('Rp', 'fatm', 'Matm', 'Mdot', 'Mdot_H', 'Mdot_He', 'Mdot_HHe', 'outflow_H_fraction', 'escape_He_H',
               'subcritical', 'T_surf')
              + tuple(f'N_{X}' for X in SPECIES) + tuple(f'N_{X}_int' for X in SPECIES)
              + ('H_He', 'C_O', 'D_H') + tuple(f'n_{mol}_atm' for mol in MOLECULES)
              + tuple(f'VMR_{mol}' for mol in MOLECULES) + ('mmw',))
TRACK_DTYPE = {key: np.float64 if key.startswith('N_') else np.float32 for key in TRACK_KEYS}

# worker globals, set by init_worker
_CFG = None
_RUN_DIR = None
_N_UNEXPECTED = 0


def blob_dtype():
    return [(f, 'f8') for f in BLOB_FIELDS]


class InvalidInputs(ValueError):
    '''
    Inputs isocalc must not be run with (status 4).
    '''


###_____isocalc inputs_____###

def number_ratios(params):
    '''
    Initial atmospheric number ratios X/H of the 'ratios' composition: protosolar He/H and D/H; O, C, N, S
    scaled by Z = 10**log_metallicity, with the extra O/H and C/H enhancements.
    '''
    Z = 10**params['log_metallicity']
    return {'He': c.HetoH_protosolar, 'D': c.DtoH_solar,
            'O': c.OtoH_protosolar*Z*10**params['log_OtoH_enhancement'],
            'C': c.CtoH_protosolar*Z*10**params['log_CtoH_enhancement'],
            'N': c.NtoH_protosolar*Z, 'S': c.StoH_protosolar*Z}


def _ratio_composition(params, M_atm):
    r = number_ratios(params)
    N_H = M_atm/(c.mu_H + sum(r[X]*MU[X] for X in r)) # [atoms]
    N = {X: N_H if X == 'H' else r[X]*N_H for X in SPECIES}
    return N, M_atm/sum(N.values())


def _legacy_composition(params, M_atm):
    '''
    isofate_MCMC.py:139-149, operation for operation: mass ratios to H, with the enhanced O mass taken out of the
    H budget (so O/H is nonlinear in the enhancement and N_H < 0 above an enhancement of 108).
    '''
    OtoH_enhancement = 10**params['log_OtoH_enhancement']
    OtoH_enhanced = c.OtoH_protosolar*OtoH_enhancement
    OtoH_enhanced_mass = OtoH_enhanced*(c.mu_O/c.mu_H)
    He = c.HetoH_protosolar_mass
    N_He = (He/(1 + He))*M_atm/c.mu_He
    N_H = (1 - c.DtoH_solar_mass - OtoH_enhanced_mass - c.CtoH_protosolar_mass - c.StoH_protosolar_mass
           - c.NtoH_protosolar_mass)*M_atm/(1 + He)/c.mu_H
    N_D = c.DtoH_solar_mass*M_atm/(1 + He)/c.mu_D
    N_O = OtoH_enhanced_mass*M_atm/(1 + He)/c.mu_O
    N_C = c.CtoH_protosolar_mass*M_atm/(1 + He)/c.mu_C
    N_N = c.NtoH_protosolar_mass*M_atm/(1 + He)/c.mu_N
    N_S = c.StoH_protosolar_mass*M_atm/(1 + He)/c.mu_S
    if not N_H > 0:
        raise InvalidInputs(f"legacy composition gives N_H = {N_H:.3g} <= 0 at log_OtoH_enhancement = "
                            f"{params['log_OtoH_enhancement']:g}")
    mu_avg = (N_H*c.mu_H + N_He*c.mu_He + N_D*c.mu_D + N_O*c.mu_O + N_C*c.mu_C + N_N*c.mu_N + N_S*c.mu_S)/(
        N_H + N_He + N_D + N_O + N_C + N_N + N_S)
    return {'H': N_H, 'He': N_He, 'D': N_D, 'O': N_O, 'C': N_C, 'N': N_N, 'S': N_S}, mu_avg


def composition(params, cfg, M_atm):
    '''
    Initial atmospheric inventories {X: N_X [atoms]} and mean particle mass mu [kg] for M_atm [kg].
    '''
    if cfg['model']['composition'] == 'legacy':
        return _legacy_composition(params, M_atm)
    return _ratio_composition(params, M_atm)


def planet_inputs(params, cfg, n_steps = None, n_atmodeller = None, save_molecules = False):
    '''
    (args, kwargs) of the isocalc call for a parameter dict (see isofate_coupler.isocalc).
    n_steps and n_atmodeller default to cfg['model'].
    '''
    model, star, planet = cfg['model'], cfg['star'], cfg['planet']
    Mp = params['Mp']*c.Me # [kg]
    f_atm = 10**params['log_f_atm']
    Fp = planet['Fp'] # [W/m2]
    T = EqTemp(Fp, A = params['albedo']) # [K]
    F0 = Fp*10**params['log_F0_ratio'] # [W/m2]
    N, mu = composition(params, cfg, Mp*f_atm)
    # a constant radius [m] (config sets rad_evol = false with it); False: isocalc's own radius
    Rp_override = params['Rp_fixed']*c.Re if params.get('Rp_fixed') is not None else False
    args = (f_atm, Mp, star['M'], F0, Fp, T, planet['d'], params['age'], model['mechanism'], model['rad_evol'])
    kwargs = dict(N_H = N['H'], N_He = N['He'], N_D = N['D'], N_O = N['O'], N_C = N['C'], N_N = N['N'],
                  N_S = N['S'], melt_fraction_override = model['melt_fraction_override'], mu = mu,
                  eps = params['eps'], activity = model['activity'], flux_model = model['flux_model'],
                  stellar_type = model['stellar_type'], Rp_override = Rp_override, t_sat = params['t_sat'],
                  step_fn = model['step_fn'], F_final = params['F_final'], t_pms = params['t_pms'],
                  pms_factor = params['pms_factor'],
                  n_steps = int(model['n_steps'] if n_steps is None else n_steps), t0 = model['t0'],
                  rho_rcb = model['rho_rcb'], RR = model['RR'], thermal = model['thermal'], beta = params['beta'],
                  n_atmodeller = int(model['n_atmodeller'] if n_atmodeller is None else n_atmodeller),
                  save_molecules = bool(save_molecules),
                  mantle_iron_dict = copy.deepcopy(model['mantle_iron']), # isocalc mutates it
                  dynamic_phi = model['dynamic_phi'])
    return args, kwargs


def run_model(args, kwargs, cfg):
    '''
    isofate_coupler.isocalc, or toy.isocalc for backend = "toy". Imports the model lazily.
    '''
    if cfg['model']['backend'] == 'toy':
        from isofate.mcmc import toy
        return toy.isocalc(*args, **kwargs)
    from isofate.isofate_coupler import isocalc
    return isocalc(*args, **kwargs)


###_____Observables of the final state_____###

def _ratio(a, b):
    '''
    a/b with NaN for 0/0 and inf for x/0.
    '''
    if b != 0:
        return a/b
    return nan if a == 0 else math.copysign(inf, a)


def _ratio_array(a, b):
    out = np.full(np.shape(a), nan)
    np.divide(a, b, out = out, where = b != 0)
    out[(b == 0) & (a > 0)] = inf
    return out


def atmodeller_calls(n_end, n_steps, n_atmodeller):
    '''
    Number of Atmodeller calls of an isocalc run whose atmosphere lasted n_end of n_steps steps: one every
    n_atmodeller steps while it lasts, plus the final call at step n_steps - 1.
    '''
    if n_atmodeller == 0 or n_end == 0:
        return 0
    return (n_end - 1)//n_atmodeller + 1 + int(n_end == n_steps)


def final_molecules(sol):
    '''
    Atmospheric gas amounts [mol] at the end from sol['atmodeller_final'] (the 10 Atmodeller gases) and He from
    conservation through the final Atmodeller call. None when unavailable (no Atmodeller, exhausted, NaN).
    '''
    final = sol.get('atmodeller_final')
    if not final or not math.isfinite(final.get('H2_atm', nan)): # no Atmodeller, or NaN after exhaustion
        return None
    n = {mol: float(final[f'{ISOCALC_NAME[mol]}_atm']) for mol in GASES}
    n['He'] = max((sol['N_He'][-1] + sol['N_He_int'][-1])/c.avogadro - final['He_mantle'], 0.0)
    if not all(math.isfinite(v) for v in n.values()):
        return None
    return n


def extract(sol, params, cfg, n_atmodeller = None):
    '''
    Every registry observable (NaN where unavailable) and the extras, from an isocalc solution:
    exhausted (1 if fatm[-1] <= 0), t_lost [yr] (first step without atmosphere, NaN if never), Matm [kg],
    subcritical (1 if phi < phic at the end, NaN when exhausted), n_atmod_calls.
    Composition observables are NaN when the atmosphere is exhausted.
    '''
    n_atm = cfg['model']['n_atmodeller'] if n_atmodeller is None else n_atmodeller
    obs = dict.fromkeys(OBS_FIELDS, nan)
    obs.update(likelihood.input_values(params, cfg))
    fatm = sol['fatm']
    n_steps = fatm.size
    lost = np.flatnonzero(fatm <= 0)
    n_end = int(lost[0]) if lost.size else n_steps # first step without an atmosphere
    exhausted = bool(fatm[-1] <= 0)
    obs['exhausted'] = float(exhausted)
    obs['t_lost'] = float(sol['time'][n_end]*c.s2yr) if lost.size else nan
    obs['n_atmod_calls'] = float(atmodeller_calls(n_end, n_steps, n_atm))
    obs['fatm'] = fatm[-1]
    obs['Matm'] = sol['Matm'][-1]

    Rp = sol['Rp'][-1] # [m]
    area = 4*np.pi*Rp**2 # [m2]
    obs['Rp'] = Rp/c.Re
    rates = {X: sol[f'Phi_{X}'][-1]*MU[X]*area for X in SPECIES} # [kg/s]
    obs['Mdot'] = sum(rates.values())
    obs['Mdot_H'], obs['Mdot_He'] = rates['H'], rates['He']
    obs['Mdot_HHe'] = rates['H'] + rates['He']
    Phi_H, Phi_He = sol['Phi_H'][-1], sol['Phi_He'][-1] # [atoms/m2/s]
    # number fractions of the H+He outflow (p-winds h_fraction): exactly 1 for an H-only wind, NaN for no wind
    obs['outflow_H_fraction'] = _ratio(Phi_H, Phi_H + Phi_He)
    obs['outflow_He_fraction'] = _ratio(Phi_He, Phi_H + Phi_He)
    obs['escape_He_H'], obs['escape_H_He'] = _ratio(Phi_He, Phi_H), _ratio(Phi_H, Phi_He)

    if not exhausted:
        obs['subcritical'] = float(sol['phi'][-1] < sol['phic'][-1])
        N = {X: sol[f'N_{X}'][-1] for X in SPECIES} # [atoms], before the final Atmodeller call
        obs['H_He'] = _ratio(N['H'] + N['D'], N['He'])
        obs['D_H'] = _ratio(N['D'], N['H']) # D is lumped into H in Atmodeller
        obs['C_O'] = _ratio(N['C'], N['O'])
        n = final_molecules(sol)
        if n is not None:
            total = sum(n.values())
            if total > 0:
                for mol in MOLECULES:
                    obs[f'VMR_{mol}'] = n[mol]/total
                obs['mmw'] = sum(n[mol]*MOLAR_MASS[mol] for mol in MOLECULES)/total
            C = n['CO'] + n['CO2'] + n['CH4']
            O = n['H2O'] + 2*n['O2'] + n['CO'] + 2*n['CO2'] + 4*n['H2SO4'] + 2*n['SO2']
            obs['C_O'] = _ratio(C, O)
        if n_atm > 0:
            obs['T_surf'] = sol['T_surf_atmod'][-1]
    return {key: float(value) for key, value in obs.items()}


###_____Evaluation_____###

def _expected(error):
    '''
    True for the model failures counted as status 2.
    '''
    if isinstance(error, EXPECTED_ERRORS):
        return True
    names = [cls.__name__ for cls in type(error).__mro__]
    return any('XlaRuntimeError' in name or 'EqxRuntimeError' in name for name in names)


def _report_unexpected(theta, error):
    '''
    Logs a status-5 traceback to <run_dir>/worker_errors/<pid>.log and prints the first 3 per process.
    '''
    global _N_UNEXPECTED
    _N_UNEXPECTED += 1
    text = f'theta = {list(np.asarray(theta, dtype = float))}\n{traceback.format_exc()}'
    if _RUN_DIR:
        try:
            folder = os.path.join(_RUN_DIR, 'worker_errors')
            os.makedirs(folder, exist_ok = True)
            with open(os.path.join(folder, f'{os.getpid()}.log'), 'a') as f:
                f.write(f'--- {time.strftime("%Y-%m-%d %H:%M:%S")} {type(error).__name__}: {error}\n{text}\n')
        except OSError:
            pass
    if _N_UNEXPECTED <= 3:
        print(f'[pid {os.getpid()}] unexpected error ({_N_UNEXPECTED}): {type(error).__name__}: {error}\n{text}',
              file = sys.stderr, flush = True)


def _nan_obs():
    return dict.fromkeys(OBS_FIELDS, nan)


def evaluate(theta, cfg, n_steps = None, n_atmodeller = None, save_molecules = False, return_sol = False):
    '''
    Prior, input observables, isocalc, observables and likelihood at a physical theta (cfg['free'] order).
    Returns {status, runtime [s] (isocalc only), lp, log_prior, log_like, terms, obs, params, error, solve_value
    (NaN), solve_evals (model runs: 0 or 1), log_jac (0), (sol)}. lp = log_prior + log_like, -inf unless status 0,
    never NaN. solve.evaluate is the solve-mode counterpart for a sampler theta.
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    res = {'status': nan, 'runtime': 0.0, 'lp': -inf, 'log_prior': -inf, 'log_like': nan, 'terms': {},
           'obs': _nan_obs(), 'params': prm.theta_to_params(theta, cfg), 'error': None, 'solve_value': nan,
           'solve_evals': 0, 'log_jac': 0.0}
    if return_sol:
        res['sol'] = None
    params = res['params']
    res['log_prior'] = lp0 = prm.log_prior(theta, cfg)
    if not math.isfinite(lp0):
        res['status'] = OUTSIDE_PRIOR
        return res

    res['obs'].update(likelihood.input_values(params, cfg))
    ll_inputs, terms = likelihood.input_log_like(params, cfg)
    if ll_inputs == -inf:
        res.update(status = NO_LIKELIHOOD, log_like = -inf, terms = terms)
        return res

    n_atm = cfg['model']['n_atmodeller'] if n_atmodeller is None else int(n_atmodeller)
    try:
        args, kwargs = planet_inputs(params, cfg, n_steps, n_atm, save_molecules)
    except InvalidInputs as e:
        res.update(status = INVALID_INPUTS, error = f'{type(e).__name__}: {e}')
        return res
    except Exception as e:
        _report_unexpected(theta, e)
        res.update(status = UNEXPECTED, error = f'{type(e).__name__}: {e}')
        return res

    start = time.perf_counter()
    res['solve_evals'] = 1
    try:
        sol = run_model(args, kwargs, cfg)
    except Exception as e:
        res['runtime'] = time.perf_counter() - start
        if _expected(e):
            res.update(status = MODEL_FAILED, error = f'{type(e).__name__}: {e}')
        else:
            _report_unexpected(theta, e)
            res.update(status = UNEXPECTED, error = f'{type(e).__name__}: {e}')
        return res
    res['runtime'] = time.perf_counter() - start

    try:
        res['obs'] = extract(sol, params, cfg, n_atm)
    except Exception as e: # a bug in extract, never a model failure
        _report_unexpected(theta, e)
        res.update(status = UNEXPECTED, error = f'{type(e).__name__}: {e}')
        return res
    ll, terms = likelihood.log_like(res['obs'], cfg)
    res['log_like'], res['terms'] = ll, terms
    lp = lp0 + ll
    if math.isfinite(lp):
        res.update(status = OK, lp = lp)
    else:
        res['status'] = NO_LIKELIHOOD
    if return_sol:
        res['sol'] = sol
    return res


def blob(res):
    '''
    The blob tuple of an evaluate result, in BLOB_FIELDS order.
    '''
    head = (res['status'], res['runtime'], res['log_prior'], res['log_like'])
    tail = (res.get('solve_value', nan), res.get('solve_evals', 0), res.get('log_jac', 0.0))
    return (tuple(float(x) for x in head) + tuple(float(res['obs'][f]) for f in OBS_FIELDS)
            + tuple(float(x) for x in tail))


def sampled_evaluate(theta, cfg, table = None, **kwargs):
    '''
    The evaluation of a sampler theta: solve.evaluate in solve mode (with the warm-start table), else evaluate.
    '''
    if config.solve_on(cfg):
        from isofate.mcmc import solve
        return solve.evaluate(theta, cfg, table = table, **kwargs)
    return evaluate(theta, cfg, **kwargs)


def log_prob(theta):
    '''
    emcee log-probability with blobs: (lp,) + blob values in BLOB_FIELDS order, at a sampler theta (u in place of
    the solved parameter in solve mode; solve.evaluate with the run's warm-start table). Uses the cfg of
    init_worker.
    '''
    if _CFG is None:
        raise RuntimeError('model.log_prob needs init_worker(cfg) in this process first')
    table = None
    if config.solve_on(_CFG):
        from isofate.mcmc import solve
        table = solve.worker_table(_CFG)
    res = sampled_evaluate(theta, _CFG, table)
    return (res['lp'],) + blob(res)


def log_prob_physical(theta):
    '''
    log_prob at a physical theta (cfg['free'] order), without solving: the bisection's evaluations.
    '''
    if _CFG is None:
        raise RuntimeError('model.log_prob_physical needs init_worker(cfg) in this process first')
    res = evaluate(theta, _CFG)
    return (res['lp'],) + blob(res)


def evaluate_many(pool, thetas, physical = False):
    '''
    (lp (n,), blobs (n,) structured with blob_dtype()) of log_prob at sampler thetas (physical = True:
    log_prob_physical at physical thetas) through pool (a WorkerPool, or anything with its map, e.g. LocalPool).
    '''
    f = log_prob_physical if physical else log_prob
    results = pool.map(f, [np.asarray(theta, dtype = float) for theta in thetas])
    lp = np.array([r[0] for r in results], dtype = float)
    blobs = np.array([tuple(r[1:]) for r in results], dtype = blob_dtype())
    return lp, blobs


class LocalPool:
    '''
    In-process stand-in for a WorkerPool of log_prob or log_prob_physical, with cfg instead of the worker global
    (check, tests). table: the solve-mode warm-start table (default none: cold solves).
    '''

    def __init__(self, cfg, table = None):
        self.cfg, self.table = cfg, table

    def map(self, f, iterable):
        if f not in (log_prob, log_prob_physical):
            raise ValueError('LocalPool only evaluates model.log_prob and model.log_prob_physical')
        out = []
        for theta in iterable:
            if f is log_prob:
                res = sampled_evaluate(theta, self.cfg, self.table)
            else:
                res = evaluate(theta, self.cfg)
            out.append((res['lp'],) + blob(res))
        return out


###_____Posterior tracks_____###

def nan_track(n_points):
    return {key: np.full(n_points, nan, dtype = TRACK_DTYPE[key]) for key in TRACK_KEYS}


def make_track(sol, grid_yr, cfg):
    '''
    Time series of one isocalc solution sampled on grid_yr [yr] (np.interp over the run's own time), NaN after
    the end of the run. Keys TRACK_KEYS: Rp [R_earth], fatm, Matm [kg], Mdot* [kg/s], outflow_H_fraction,
    escape_He_H, subcritical,
    T_surf [K], N_X and N_X_int [atoms], H_He, C_O (N_C/N_O), D_H, n_<mol>_atm [mol] (n_*_a arrays, which need
    save_molecules; He from N_He), VMR_<mol>, mmw [g/mol]. Steps without atmosphere are NaN for composition.
    '''
    grid_yr = np.asarray(grid_yr, dtype = float)
    t_yr = sol['time']*c.s2yr
    alive = sol['fatm'] > 0
    Rp = sol['Rp']
    area = 4*np.pi*Rp**2
    series = {'Rp': Rp/c.Re, 'fatm': sol['fatm'], 'Matm': sol['Matm']}
    rates = {X: sol[f'Phi_{X}']*MU[X]*area for X in SPECIES}
    series['Mdot'] = sum(rates.values())
    series['Mdot_H'], series['Mdot_He'] = rates['H'], rates['He']
    series['Mdot_HHe'] = rates['H'] + rates['He']
    series['outflow_H_fraction'] = _ratio_array(sol['Phi_H'], sol['Phi_H'] + sol['Phi_He'])
    series['escape_He_H'] = _ratio_array(sol['Phi_He'], sol['Phi_H'])
    series['subcritical'] = np.where(alive, sol['phi'] < sol['phic'], nan)
    has_atmodeller = 'atmodeller_final' in sol
    series['T_surf'] = np.where(alive & has_atmodeller, sol['T_surf_atmod'], nan)
    for X in SPECIES:
        series[f'N_{X}'] = sol[f'N_{X}']
        series[f'N_{X}_int'] = sol[f'N_{X}_int']
    series['H_He'] = _ratio_array(sol['N_H'] + sol['N_D'], sol['N_He'])
    series['C_O'] = _ratio_array(sol['N_C'], sol['N_O'])
    series['D_H'] = _ratio_array(sol['N_D'], sol['N_H'])
    if 'n_H2_a' in sol:
        n = {mol: sol[f'n_{ISOCALC_NAME[mol]}_a'] for mol in GASES}
        n['He'] = sol['N_He']/c.avogadro
        total = sum(n.values())
        for mol in MOLECULES:
            series[f'n_{mol}_atm'] = n[mol]
            series[f'VMR_{mol}'] = _ratio_array(n[mol], total)
        series['mmw'] = _ratio_array(sum(n[mol]*MOLAR_MASS[mol] for mol in MOLECULES), total)
    for key in ('H_He', 'C_O', 'D_H', 'mmw') + tuple(f'VMR_{mol}' for mol in MOLECULES):
        if key in series:
            series[key] = np.where(alive, series[key], nan)

    after_end = grid_yr > t_yr[-1]
    track = {}
    for key in TRACK_KEYS:
        if key not in series:
            track[key] = np.full(grid_yr.size, nan, dtype = TRACK_DTYPE[key])
            continue
        values = np.interp(grid_yr, t_yr, np.asarray(series[key], dtype = float))
        values[after_end] = nan
        track[key] = values.astype(TRACK_DTYPE[key])
    return track


def eval_task(task):
    '''
    Pool task for resolution, scan, tracks and the preflight: task = {theta, n_steps, n_atmodeller,
    save_molecules, track_grid, id, solve} (all but theta optional). solve = True evaluates a sampler theta in solve
    mode (solve.evaluate, a cold start, no tracks); else theta is physical. Returns the evaluate result without
    sol, plus 'id' and 'theta', and 'track' (make_track, or NaN arrays when the model did not run) when track_grid
    is given.
    '''
    if _CFG is None:
        raise RuntimeError('model.eval_task needs init_worker(cfg) in this process first')
    if task.get('solve'):
        from isofate.mcmc import solve
        res = solve.evaluate(task['theta'], _CFG, task.get('n_steps'), task.get('n_atmodeller'))
        res['id'] = task.get('id')
        res['theta'] = [float(x) for x in np.atleast_1d(task['theta'])]
        return res
    grid = task.get('track_grid')
    res = evaluate(task['theta'], _CFG, task.get('n_steps'), task.get('n_atmodeller'),
                   task.get('save_molecules', False), return_sol = grid is not None)
    sol = res.pop('sol', None)
    res['id'] = task.get('id')
    res['theta'] = [float(x) for x in np.atleast_1d(task['theta'])]
    if grid is not None:
        res['track'] = nan_track(len(grid))
        if sol is not None:
            try:
                res['track'] = make_track(sol, grid, _CFG)
            except Exception as e:
                _report_unexpected(task['theta'], e)
                res['error'] = f'make_track: {type(e).__name__}: {e}'
    return res


###_____Worker set-up_____###

def warm_fallback():
    '''
    Compiles Atmodeller's cold-start solve and its multistart retry once (otherwise each costs about 2 s at
    their first use in each worker). Copied from scratchpad parallel/pooltools.py.
    '''
    from atmodeller import Planet
    from isofate import atmodeller_coupler as ac
    system = ac._compiled_system()
    Mp = 5.6*c.Me
    mass = {'H': 1e19, 'O': 1e17, 'C': 1e16, 'He': 3e18, 'N': 1e15, 'S': 1e15}
    system.interior_atmosphere.solve(planet = Planet(surface_temperature = 2000.0, mantle_melt_fraction = 1.0,
                                                     planet_mass = Mp, surface_radius = ac.R_core(Mp)),
                                     mass_constraints = mass)
    # an input whose cold start fails, so the multistart (repeat_solver) is compiled too. Captured from LHS 1140 b
    # near atmosphere exhaustion: every element at the 1e9 kg cutoff except S. Values are traced, so this warms
    # the same executable for any planet
    system.interior_atmosphere.solve(planet = Planet(surface_temperature = 955.467312638313,
                                                     mantle_melt_fraction = 0.0, planet_mass = 3.344432e+25,
                                                     surface_radius = 9811412.196650479),
                                     mass_constraints = {'H': 1e9, 'O': 1e9, 'C': 1e9, 'He': 1e9, 'N': 1e9,
                                                         'S': 8.193710271821442e+18})


def warm_up(cfg):
    '''
    One short isocalc run at the centre (compiles the Numba kernels and the JAX Atmodeller solve), then the
    Atmodeller fallback paths.
    '''
    n_atm = cfg['model']['n_atmodeller']
    evaluate(centre_theta(cfg), cfg, n_steps = 1000, n_atmodeller = min(max(n_atm, 1), 100) if n_atm > 0 else 0)
    if n_atm > 0:
        try:
            warm_fallback()
        except Exception as e:
            print(f'[pid {os.getpid()}] warning: Atmodeller fallback warm-up failed ({type(e).__name__}: {e})',
                  file = sys.stderr, flush = True)


def init_worker(cfg, run_dir = None, barrier = None):
    '''
    Pool initializer (also usable in-process): single-thread env, compile caches, the worker globals, warm-up,
    then barrier.wait. A failure aborts the barrier so the parent stops waiting.
    '''
    global _CFG, _RUN_DIR
    try:
        if barrier is not None and hasattr(sys.stdout, 'reconfigure'): # a spawned worker: print lines as they
            sys.stdout.reconfigure(line_buffering = True)                 # come, not in blocks
        env.set_single_thread_env()
        isocalc_backend = cfg['model']['backend'] == 'isocalc'
        if cfg['parallel']['cache'] and isocalc_backend:
            try:
                env.enable_compile_caches(cfg)
            except Exception as e:
                print(f'[pid {os.getpid()}] warning: compile caches not enabled ({type(e).__name__}: {e}); '
                      'continuing without them', file = sys.stderr, flush = True)
        _CFG, _RUN_DIR = cfg, run_dir
        if cfg['parallel']['warmup'] and isocalc_backend:
            warm_up(cfg)
    except BaseException:
        if barrier is not None:
            barrier.abort()
        raise
    if barrier is not None:
        try:
            barrier.wait(timeout = 600)
        except threading.BrokenBarrierError: # the parent gave up (it terminates the pool), or the pool is running
            pass
