'''
One model evaluation for the MCMC: the isocalc inputs of a parameter set, the observables of the final state,
the blobs and status codes, and the functions the worker processes run.

An evaluation (evaluate) takes a theta through the prior, the input observables, one isocalc run and the
likelihood. Its result travels as a blob, a fixed row of floats (BLOB_FIELDS) that emcee stores next to the chain
for every walker and step. A blob holds the status, the runtime, the prior and the likelihood, every observable
(active or not) and a few extras, so a run can be studied in any observable afterwards without running the model
again.

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

The workers are spawned processes (pool.py). Each runs init_worker once, which keeps the config in this module's
globals and compiles the model, and then evaluates one theta per task (log_prob, log_prob_physical, eval_task).
The coupled model (jax, atmodeller, numba) is imported lazily, so importing this module is safe in the parent,
which never imports JAX.
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

###_____Status codes, blob fields and track keys_____###

OK, OUTSIDE_PRIOR, MODEL_FAILED, NO_LIKELIHOOD, INVALID_INPUTS, UNEXPECTED = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0
STATUS_NAMES = {0: 'ok', 1: 'outside prior', 2: 'model failure', 3: 'likelihood -inf', 4: 'invalid inputs',
                5: 'unexpected error'}
# the model failures of status 2; _expected adds the JAX and equinox runtime errors by name
EXPECTED_ERRORS = (ValueError, RuntimeError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError)

SPECIES = ('H', 'He', 'D', 'O', 'C', 'N', 'S') # escape kernel order
MU = {X: getattr(c, f'mu_{X}') for X in SPECIES} # atomic masses [kg/atom]
MOLECULES = config.VMR_MOLECULES # H2, He, H2O, CO, CO2, CH4, N2, O2, S2, SO2, H2SO4
GASES = tuple(mol for mol in MOLECULES if mol != 'He') # the 10 Atmodeller gases
ISOCALC_NAME = {mol: 'H2O4S' if mol == 'H2SO4' else mol for mol in GASES} # names in isocalc's output keys
MOLAR_MASS = {'H2': 2.016, 'He': 4.0026, 'H2O': 18.015, 'CO': 28.010, 'CO2': 44.009, 'CH4': 16.043, 'N2': 28.014,
              'O2': 31.998, 'S2': 64.13, 'SO2': 64.066, 'H2SO4': 98.079} # [g/mol]

# the blob fields after the observables (see extract)
EXTRAS = ('exhausted', 't_lost', 'Matm', 'subcritical', 'n_atmod_calls') # T_surf is an observable already
OBS_FIELDS = tuple(config.OBSERVABLES) + EXTRAS
# solve mode (solve.py): solve_value = the solved parameter P* (NaN without solve mode), solve_evals = model runs
# of the evaluation (0 or 1 without solve mode), log_jac = ln|dP*/du| (0 without solve mode: the identity)
SOLVE_FIELDS = ('solve_value', 'solve_evals', 'log_jac')
BLOB_FIELDS = ('status', 'runtime', 'log_prior', 'log_like') + OBS_FIELDS + SOLVE_FIELDS # one blob, in this order
NAN_BLOB = (nan,)*len(BLOB_FIELDS) # the blob of a point not evaluated (yet)

# posterior tracks (tracks.py): time series on a common grid. N_X in float64: atom numbers (about 1e49 at the
# start) overflow float32, whose largest value is 3.4e38
TRACK_KEYS = (('Rp', 'fatm', 'Matm', 'Mdot', 'Mdot_H', 'Mdot_He', 'Mdot_HHe', 'outflow_H_fraction', 'escape_He_H',
               'subcritical', 'T_surf')
              + tuple(f'N_{X}' for X in SPECIES) + tuple(f'N_{X}_int' for X in SPECIES)
              + ('H_He', 'C_O', 'D_H') + tuple(f'n_{mol}_atm' for mol in MOLECULES)
              + tuple(f'VMR_{mol}' for mol in MOLECULES) + ('mmw',))
TRACK_DTYPE = {key: np.float64 if key.startswith('N_') else np.float32 for key in TRACK_KEYS}

# worker globals, set by init_worker: a task carries only theta (or a task dict), so the config travels to each
# worker once, at start-up
_CFG = None
_RUN_DIR = None # run directory: worker_errors/ and the solve-mode warm-start table
_N_UNEXPECTED = 0 # status-5 errors of this process so far (the first 3 are printed)


def blob_dtype():
    '''
    Returns the structured dtype of a blob: one float64 field per name of BLOB_FIELDS (a list numpy accepts).
    '''
    return [(f, 'f8') for f in BLOB_FIELDS]


class InvalidInputs(ValueError):
    '''
    Raised for inputs isocalc must not be run with (status 4), e.g. N_H <= 0 in the legacy composition.
    '''


###_____isocalc inputs_____###

def number_ratios(params):
    '''
    Calculates the initial atmospheric number ratios X/H of the 'ratios' composition
    Abundances from Lodders 2003 (constants.py)

    Protosolar He/H and solar D/H; O, C, N and S protosolar, scaled by the metallicity Z = 10**log_metallicity,
    with the extra O/H and C/H enhancements on top.

    Inputs:
        - params: parameter dict (params.theta_to_params); uses log_metallicity, log_OtoH_enhancement and
          log_CtoH_enhancement [dex]

    Output: {X: N_X/N_H} for X in He, D, O, C, N, S [ndim; dict]
    '''
    Z = 10**params['log_metallicity']
    return {'He': c.HetoH_protosolar, 'D': c.DtoH_solar,
            'O': c.OtoH_protosolar*Z*10**params['log_OtoH_enhancement'],
            'C': c.CtoH_protosolar*Z*10**params['log_CtoH_enhancement'],
            'N': c.NtoH_protosolar*Z, 'S': c.StoH_protosolar*Z}


def _ratio_composition(params, M_atm):
    '''
    Inventories of the 'ratios' composition: N_H from the mass budget M_atm = N_H (mu_H + sum_X r_X mu_X), every
    other species r_X N_H. Returns ({X: N_X} [atoms], mean particle mass [kg]).
    '''
    r = number_ratios(params)
    N_H = M_atm/(c.mu_H + sum(r[X]*MU[X] for X in r)) # [atoms]
    N = {X: N_H if X == 'H' else r[X]*N_H for X in SPECIES}
    return N, M_atm/sum(N.values())


def _legacy_composition(params, M_atm):
    '''
    Inventories of the 'legacy' composition, isofate_MCMC.py:139-149 operation for operation
    Mass ratios to H, with the enhanced O mass taken out of the H budget, so O/H is nonlinear in the enhancement
    and N_H < 0 above an enhancement of 108 (log_OtoH_enhancement = 2.03), which raises InvalidInputs (status 4).
    Returns ({X: N_X} [atoms], mean particle mass [kg]).
    '''
    OtoH_enhancement = 10**params['log_OtoH_enhancement']
    OtoH_enhanced = c.OtoH_protosolar*OtoH_enhancement # number ratio
    OtoH_enhanced_mass = OtoH_enhanced*(c.mu_O/c.mu_H) # mass ratio
    He = c.HetoH_protosolar_mass # He/H mass ratio
    N_He = (He/(1 + He))*M_atm/c.mu_He
    # the metals' mass ratios to H are taken out of the H budget M_atm/(1 + He): N_H < 0 once they add up to over 1
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
    Calculates the initial atmospheric inventories, for model.composition = 'ratios' or 'legacy'

    Inputs:
        - params: parameter dict (params.theta_to_params) [dict]
        - cfg: resolved config [dict]
        - M_atm: initial atmospheric mass [kg]

    Output: tuple
        - N: initial inventories {X: N_X} for X in H, He, D, O, C, N, S [atoms; dict]
        - mu: mean particle mass [kg]
    '''
    if cfg['model']['composition'] == 'legacy':
        return _legacy_composition(params, M_atm)
    return _ratio_composition(params, M_atm)


def planet_inputs(params, cfg, n_steps = None, n_atmodeller = None, save_molecules = False):
    '''
    Builds the isocalc call of a parameter set (see isofate_coupler.isocalc)

    Inputs:
        - params: parameter dict (params.theta_to_params), in the units of config.PARAMS [dict]
        - cfg: resolved config [dict]
        - n_steps: number of timesteps; default cfg['model']['n_steps']
        - n_atmodeller: timesteps between Atmodeller calls (0: no Atmodeller); default cfg['model']['n_atmodeller']
        - save_molecules: True to keep isocalc's molecule arrays n_*_a of every step (the tracks need them) [Bool]

    Output: (args, kwargs) of the isocalc call [tuple]; raises InvalidInputs for a legacy composition with
        N_H <= 0
    '''
    model, star, planet = cfg['model'], cfg['star'], cfg['planet']
    Mp = params['Mp']*c.Me # [kg]
    f_atm = 10**params['log_f_atm'] # initial atmospheric mass fraction [ndim]
    Fp = planet['Fp'] # incident bolometric flux [W/m2]
    T = EqTemp(Fp, A = params['albedo']) # equilibrium temperature, full redistribution [K]
    F0 = Fp*10**params['log_F0_ratio'] # initial XUV flux [W/m2]
    N, mu = composition(params, cfg, Mp*f_atm)
    # a constant radius [m] (config sets rad_evol = false with it); False: isocalc's own radius
    Rp_override = params['Rp_fixed']*c.Re if params.get('Rp_fixed') is not None else False
    # isocalc's positional arguments: Mstar [kg], d (orbital distance) [m], time = the age [yr]
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
    Runs isofate_coupler.isocalc, or toy.isocalc for model.backend = "toy"
    The model is imported here, at the first call, so the parent process never loads it through this module.

    Inputs:
        - args, kwargs: the isocalc call, as planet_inputs builds it [tuple, dict]
        - cfg: resolved config [dict]

    Output: the isocalc solution [dict]
    '''
    if cfg['model']['backend'] == 'toy':
        from isofate.mcmc import toy
        return toy.isocalc(*args, **kwargs)
    from isofate.isofate_coupler import isocalc
    return isocalc(*args, **kwargs)


###_____Observables of the final state_____###

def _ratio(a, b):
    '''
    Returns a/b, with NaN for 0/0 and +-inf (the sign of a) for x/0.
    '''
    if b != 0:
        return a/b
    return nan if a == 0 else math.copysign(inf, a)


def _ratio_array(a, b):
    '''
    Element-wise a/b, with NaN for 0/0 and inf for a > 0 over 0 (NaN for a < 0 over 0).
    '''
    out = np.full(np.shape(a), nan)
    np.divide(a, b, out = out, where = b != 0)
    out[(b == 0) & (a > 0)] = inf
    return out


def atmodeller_calls(n_end, n_steps, n_atmodeller):
    '''
    Counts the Atmodeller calls of an isocalc run
    One call every n_atmodeller steps (steps 0, n_atmodeller, 2 n_atmodeller, ...) while the atmosphere lasts,
    plus the final call at step n_steps - 1 when it lasts to the end: e.g. 1001 calls at n_steps = 1e5 and
    n_atmodeller = 100.

    Inputs:
        - n_end: steps the atmosphere lasted, n_steps if it was never lost [int]
        - n_steps: number of timesteps [int]
        - n_atmodeller: timesteps between Atmodeller calls; 0 for none [int]

    Output: number of Atmodeller calls [int]
    '''
    if n_atmodeller == 0 or n_end == 0:
        return 0
    return (n_end - 1)//n_atmodeller + 1 + int(n_end == n_steps)


def final_molecules(sol):
    '''
    Returns the atmospheric gas amounts at the end of a run
    The 10 Atmodeller gases come from sol['atmodeller_final'], the Atmodeller solve at the last step. He has no gas
    key there, so it comes from conservation through that final call: the kernel's He in the atmosphere and the
    mantle before the call, minus the He dissolved after it, (N_He + N_He_int)/N_A - He_mantle.

    Inputs:
        - sol: isocalc solution [dict]

    Output: {mol: amount} for the 11 molecules of MOLECULES [mol; dict]; None when unavailable (no Atmodeller, an
        exhausted atmosphere, or a NaN)
    '''
    final = sol.get('atmodeller_final')
    if not final or not math.isfinite(final.get('H2_atm', nan)): # no Atmodeller, or NaN after exhaustion
        return None
    n = {mol: float(final[f'{ISOCALC_NAME[mol]}_atm']) for mol in GASES}
    n['He'] = max((sol['N_He'][-1] + sol['N_He_int'][-1])/c.avogadro - final['He_mantle'], 0.0) # never negative
    if not all(math.isfinite(v) for v in n.values()): # e.g. a final solve that did not converge
        return None
    return n


def extract(sol, params, cfg, n_atmodeller = None):
    '''
    Reads every observable and the extras off the final state of an isocalc run
    Every value refers to the last step of the run, which isocalc puts one step past the age. Escape rates use the
    area 4 pi Rp^2 at that step and the atomic masses of constants.py. Composition observables are NaN when the
    atmosphere is exhausted; the molecules (VMRs and mmw, and C/O when they are available) come from the final
    Atmodeller solve (final_molecules).

    Inputs:
        - sol: isocalc solution [dict]
        - params: parameter dict of the run (params.theta_to_params) [dict]
        - cfg: resolved config [dict]
        - n_atmodeller: timesteps between Atmodeller calls in the run; default cfg['model']['n_atmodeller']

    Output: {name: value} for every name of OBS_FIELDS, NaN where unavailable [dict of floats]. The extras:
        - exhausted: 1 if fatm[-1] <= 0, else 0
        - t_lost: time of the first step without atmosphere [yr]; NaN if never
        - Matm: final atmospheric mass [kg]
        - subcritical: 1 if phi < phic at the end (He no longer escapes), else 0; NaN when exhausted
        - n_atmod_calls: Atmodeller calls of the run (atmodeller_calls)
    '''
    n_atm = cfg['model']['n_atmodeller'] if n_atmodeller is None else n_atmodeller
    obs = dict.fromkeys(OBS_FIELDS, nan)
    obs.update(likelihood.input_values(params, cfg)) # Mp, age and Teq, straight from the inputs
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
        obs['H_He'] = _ratio(N['H'] + N['D'], N['He']) # D counts as hydrogen
        obs['D_H'] = _ratio(N['D'], N['H']) # D is lumped into H in Atmodeller
        obs['C_O'] = _ratio(N['C'], N['O']) # the kernel's C/O, replaced below when the molecules are available
        n = final_molecules(sol)
        if n is not None:
            total = sum(n.values()) # [mol]
            if total > 0:
                for mol in MOLECULES:
                    obs[f'VMR_{mol}'] = n[mol]/total
                obs['mmw'] = sum(n[mol]*MOLAR_MASS[mol] for mol in MOLECULES)/total # [g/mol]
            # C and O atoms in the Atmodeller gases
            C = n['CO'] + n['CO2'] + n['CH4']
            O = n['H2O'] + 2*n['O2'] + n['CO'] + 2*n['CO2'] + 4*n['H2SO4'] + 2*n['SO2']
            obs['C_O'] = _ratio(C, O)
        if n_atm > 0:
            obs['T_surf'] = sol['T_surf_atmod'][-1] # base of the adiabat, capped at 6000 K [K]
    return {key: float(value) for key, value in obs.items()}


###_____Evaluation_____###

def _expected(error):
    '''
    Returns True for the model failures counted as status 2: EXPECTED_ERRORS, and the runtime errors of JAX
    (XlaRuntimeError) and equinox (EqxRuntimeError), matched by class name so that this module never imports them.
    '''
    if isinstance(error, EXPECTED_ERRORS):
        return True
    names = [cls.__name__ for cls in type(error).__mro__]
    return any('XlaRuntimeError' in name or 'EqxRuntimeError' in name for name in names)


def _report_unexpected(theta, error):
    '''
    Logs a status-5 traceback (with theta) to <run_dir>/worker_errors/<pid>.log and prints the first 3 per process.
    Called inside the except block, where traceback.format_exc sees the error. Without a run directory nothing is
    written, and a failure to write is ignored, so the report never fails an evaluation itself.
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
    '''
    Returns every observable and extra (OBS_FIELDS) as NaN.
    '''
    return dict.fromkeys(OBS_FIELDS, nan)


def evaluate(theta, cfg, n_steps = None, n_atmodeller = None, save_molecules = False, return_sol = False):
    '''
    Evaluates the model at a physical theta: prior, input observables, isocalc, observables and likelihood
    The cheap checks come first, so a point that its prior or the input observables (Mp, age, Teq) rule out costs
    no model run. solve.evaluate is the solve-mode counterpart for a sampler theta.

    Inputs:
        - theta: free parameters in cfg['free'] order, in the units of config.PARAMS [array]
        - cfg: resolved config [dict]
        - n_steps, n_atmodeller: resolution of the run; default those of cfg['model'] [int]
        - save_molecules: True to keep isocalc's molecule arrays (the tracks need them) [Bool]
        - return_sol: True to add the isocalc solution to the result [Bool]

    Output: dict
        - status: status code (see the module docstring) [float]
        - runtime: wall time of isocalc alone [s]; 0 when it did not run
        - lp: log_prior + log_like; -inf unless status 0, never NaN
        - log_prior: log prior up to a constant (params.log_prior); -inf outside the prior
        - log_like: total log-likelihood; NaN when the model did not run (-inf when the inputs ruled the point out)
        - terms: {name: term} of the active observables [dict]
        - obs: every observable and extra (extract), NaN where unavailable [dict]
        - params: every parameter (params.theta_to_params) [dict]
        - error: 'Type: message' of a failure, else None [str]
        - solve_value: NaN, the solved parameter in solve mode
        - solve_evals: model runs of the evaluation: 0 or 1 [int]
        - log_jac: 0, ln|dP*/du| in solve mode
        - sol: with return_sol only: the isocalc solution; None unless the model ran and its observables were
          read [dict]
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype = float))
    # the result of a point that is never run; filled in step by step
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

    # the input observables need no model run: a point they rule out is rejected for free
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
    except Exception as e: # anything else here is a bug
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
    else: # -inf from the data, e.g. a detected species in a stripped atmosphere
        res['status'] = NO_LIKELIHOOD
    if return_sol:
        res['sol'] = sol
    return res


def blob(res):
    '''
    Packs an evaluate result into its blob, in BLOB_FIELDS order
    The solve-mode fields default to NaN, 0 and 0 when the result lacks them.

    Inputs:
        - res: result of evaluate or solve.evaluate [dict]

    Output: the blob values [tuple of floats]
    '''
    head = (res['status'], res['runtime'], res['log_prior'], res['log_like'])
    tail = (res.get('solve_value', nan), res.get('solve_evals', 0), res.get('log_jac', 0.0))
    return (tuple(float(x) for x in head) + tuple(float(res['obs'][f]) for f in OBS_FIELDS)
            + tuple(float(x) for x in tail))


###_____Log-probability for emcee and the pools_____###

def sampled_evaluate(theta, cfg, table = None, **kwargs):
    '''
    Evaluates a sampler theta: solve.evaluate in solve mode (with the warm-start table), else evaluate

    Inputs:
        - theta: sampler coordinates, in config.sampled_names order [array]
        - cfg: resolved config [dict]
        - table: the warm-start table of solve mode (solve.fit_table); None: a cold solve [dict]
        - kwargs: passed on to the evaluation, e.g. n_steps and n_atmodeller

    Output: the result of solve.evaluate or evaluate [dict]
    '''
    if config.solve_on(cfg):
        from isofate.mcmc import solve
        return solve.evaluate(theta, cfg, table = table, **kwargs)
    return evaluate(theta, cfg, **kwargs)


def log_prob(theta):
    '''
    Returns the log-probability with blobs, the function emcee calls in the workers
    Blobs are extra values returned after lp, which emcee stores with the chain for every walker and step
    (Foreman-Mackey et al 2013). theta is a sampler theta: in solve mode it holds u in place of the solved
    parameter, and solve.evaluate solves it with the run's warm-start table (solve.worker_table). Uses the cfg of
    init_worker, so that a task only sends theta.

    Inputs:
        - theta: sampler coordinates, in config.sampled_names order [array]

    Output: (lp,) + the blob values in BLOB_FIELDS order [tuple of floats]; raises RuntimeError before init_worker
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
    Returns log_prob at a physical theta, without solving: the evaluations of the bisection

    Inputs:
        - theta: physical free parameters, in cfg['free'] order [array]

    Output: (lp,) + the blob values in BLOB_FIELDS order, as log_prob [tuple of floats]; raises RuntimeError
        before init_worker
    '''
    if _CFG is None:
        raise RuntimeError('model.log_prob_physical needs init_worker(cfg) in this process first')
    res = evaluate(theta, _CFG)
    return (res['lp'],) + blob(res)


def evaluate_many(pool, thetas, physical = False):
    '''
    Evaluates many thetas through a pool, one task each

    Inputs:
        - pool: a WorkerPool, or anything with its map, e.g. LocalPool
        - thetas: sampler thetas (n, ndim), or physical ones with physical = True [array]
        - physical: True to map log_prob_physical instead of log_prob [Bool]

    Output: tuple
        - lp: log-probabilities (n,) [array]
        - blobs: their blobs (n,), structured with blob_dtype() [array]
    '''
    f = log_prob_physical if physical else log_prob
    results = pool.map(f, [np.asarray(theta, dtype = float) for theta in thetas])
    lp = np.array([r[0] for r in results], dtype = float)
    blobs = np.array([tuple(r[1:]) for r in results], dtype = blob_dtype())
    return lp, blobs


class LocalPool:
    '''
    In-process stand-in for a WorkerPool of log_prob or log_prob_physical (check, tests)
    It holds the cfg itself instead of the worker global, so it needs no init_worker; the model runs in the
    calling process, which then imports it. table: the solve-mode warm-start table (default none: cold solves).
    '''

    def __init__(self, cfg, table = None):
        '''
        Keeps the config and the warm-start table for the evaluations

        Inputs:
            - cfg: resolved config [dict]
            - table: solve-mode warm-start table, or None for cold solves [dict]
        '''
        self.cfg, self.table = cfg, table

    def map(self, f, iterable):
        '''
        Evaluates each theta of iterable in turn, as f would in a worker
        f only selects the evaluation: log_prob runs sampled_evaluate with the pool's table, log_prob_physical runs
        evaluate. Any other f raises ValueError.

        Output: (lp,) + the blob values of each theta, in order [list of tuples]
        '''
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
    '''
    Returns a track of NaN arrays (n_points,) under every key of TRACK_KEYS, for a run that did not happen.
    '''
    return {key: np.full(n_points, nan, dtype = TRACK_DTYPE[key]) for key in TRACK_KEYS}


def make_track(sol, grid_yr, cfg):
    '''
    Samples the time series of one isocalc solution on a time grid
    Linear interpolation (np.interp) over the run's own time; NaN after the end of the run, and grid times before
    its first step take the first step's values. Steps without atmosphere are NaN for composition.

    Inputs:
        - sol: isocalc solution [dict]
        - grid_yr: the time grid [yr; array]
        - cfg: resolved config [dict]

    Output: {key: values on the grid} for every key of TRACK_KEYS, in the dtypes of TRACK_DTYPE [dict]
        - Rp [R_earth], fatm [ndim], Matm [kg]
        - Mdot, Mdot_H, Mdot_He, Mdot_HHe [kg/s], outflow_H_fraction, escape_He_H
        - subcritical (1 below the critical flux for He escape) and T_surf [K]: NaN without atmosphere, T_surf
          also without Atmodeller
        - N_X and N_X_int [atoms]
        - H_He, C_O (the kernel's N_C/N_O; inf once all O is dissolved), D_H
        - n_<mol>_atm [mol] (from the n_*_a arrays, which need save_molecules; He from N_He), VMR_<mol>,
          mmw [g/mol]; NaN without save_molecules. The n_*_a arrays hold the values of the most recent Atmodeller
          call, so these step every n_atmodeller steps
    '''
    grid_yr = np.asarray(grid_yr, dtype = float)
    t_yr = sol['time']*c.s2yr # [yr]
    alive = sol['fatm'] > 0 # steps with an atmosphere
    Rp = sol['Rp'] # [m]
    area = 4*np.pi*Rp**2 # [m2]
    series = {'Rp': Rp/c.Re, 'fatm': sol['fatm'], 'Matm': sol['Matm']} # per step of the run
    rates = {X: sol[f'Phi_{X}']*MU[X]*area for X in SPECIES} # [kg/s]
    series['Mdot'] = sum(rates.values())
    series['Mdot_H'], series['Mdot_He'] = rates['H'], rates['He']
    series['Mdot_HHe'] = rates['H'] + rates['He']
    series['outflow_H_fraction'] = _ratio_array(sol['Phi_H'], sol['Phi_H'] + sol['Phi_He'])
    series['escape_He_H'] = _ratio_array(sol['Phi_He'], sol['Phi_H'])
    series['subcritical'] = np.where(alive, sol['phi'] < sol['phic'], nan) # 1 below the critical flux for He
    has_atmodeller = 'atmodeller_final' in sol # isocalc writes it whenever n_atmodeller > 0
    series['T_surf'] = np.where(alive & has_atmodeller, sol['T_surf_atmod'], nan)
    for X in SPECIES:
        series[f'N_{X}'] = sol[f'N_{X}']
        series[f'N_{X}_int'] = sol[f'N_{X}_int']
    series['H_He'] = _ratio_array(sol['N_H'] + sol['N_D'], sol['N_He']) # D counts as hydrogen
    series['C_O'] = _ratio_array(sol['N_C'], sol['N_O'])
    series['D_H'] = _ratio_array(sol['N_D'], sol['N_H'])
    if 'n_H2_a' in sol: # the molecule arrays: only with save_molecules
        n = {mol: sol[f'n_{ISOCALC_NAME[mol]}_a'] for mol in GASES}
        n['He'] = sol['N_He']/c.avogadro # [mol]; Atmodeller has no He gas, so the kernel's He
        total = sum(n.values())
        for mol in MOLECULES:
            series[f'n_{mol}_atm'] = n[mol]
            series[f'VMR_{mol}'] = _ratio_array(n[mol], total)
        series['mmw'] = _ratio_array(sum(n[mol]*MOLAR_MASS[mol] for mol in MOLECULES), total)
    # composition is undefined once the atmosphere is gone
    for key in ('H_He', 'C_O', 'D_H', 'mmw') + tuple(f'VMR_{mol}' for mol in MOLECULES):
        if key in series:
            series[key] = np.where(alive, series[key], nan)

    after_end = grid_yr > t_yr[-1] # the run ends one step past the age
    track = {}
    for key in TRACK_KEYS:
        if key not in series: # e.g. the molecules without save_molecules
            track[key] = np.full(grid_yr.size, nan, dtype = TRACK_DTYPE[key])
            continue
        values = np.interp(grid_yr, t_yr, np.asarray(series[key], dtype = float))
        values[after_end] = nan
        track[key] = values.astype(TRACK_DTYPE[key])
    return track


def eval_task(task):
    '''
    Runs one pool task: the evaluations of the resolution study, scan, tracks and the preflight
    solve = True evaluates a sampler theta in solve mode (solve.evaluate, a cold start, no tracks); else theta is
    physical, evaluated by evaluate. Uses the cfg of init_worker.

    Inputs:
        - task: {theta, n_steps, n_atmodeller, save_molecules, track_grid, id, solve}, all but theta optional [dict]

    Output: the evaluate (or solve.evaluate) result without sol, plus [dict]
        - id: the task's id (None if it had none)
        - theta: the task's theta [list of floats]
        - track: with track_grid only (not in solve mode): make_track on that grid, or NaN arrays when the model
          did not run
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
    sol = res.pop('sol', None) # a whole solution is too big to send back (31 MB at 1e5 steps)
    res['id'] = task.get('id')
    res['theta'] = [float(x) for x in np.atleast_1d(task['theta'])]
    if grid is not None:
        res['track'] = nan_track(len(grid))
        if sol is not None:
            try:
                res['track'] = make_track(sol, grid, _CFG)
            except Exception as e: # a bug in make_track: the track stays NaN, the evaluation stands
                _report_unexpected(task['theta'], e)
                res['error'] = f'make_track: {type(e).__name__}: {e}'
    return res


###_____Worker set-up_____###

def warm_fallback():
    '''
    Compiles Atmodeller's fallback paths once: the cold-start solve and its multistart retry
    AtmodellerSolver falls back to them when its compiled solve does not converge. Otherwise each costs about 2 s
    of compiling at its first use in each worker, in the middle of a run.
    '''
    from atmodeller import Planet
    from isofate import atmodeller_coupler as ac
    system = ac._compiled_system()
    Mp = 5.6*c.Me # [kg]
    mass = {'H': 1e19, 'O': 1e17, 'C': 1e16, 'He': 3e18, 'N': 1e15, 'S': 1e15} # element masses [kg]
    # a fully molten 5.6 M_earth planet at 2000 K: compiles the cold-start solve
    system.interior_atmosphere.solve(planet = Planet(surface_temperature = 2000.0, mantle_melt_fraction = 1.0,
                                                     planet_mass = Mp, surface_radius = ac.R_core(Mp)),
                                     mass_constraints = mass)
    # an input whose cold start fails, so the multistart (repeat_solver) is compiled too. Captured from LHS 1140 b
    # near atmosphere exhaustion: every element at the 1e9 kg cutoff except S. Values are traced (JAX compiles for
    # the shapes, not the values), so this warms the same executable for any planet
    system.interior_atmosphere.solve(planet = Planet(surface_temperature = 955.467312638313,
                                                     mantle_melt_fraction = 0.0, planet_mass = 3.344432e+25,
                                                     surface_radius = 9811412.196650479),
                                     mass_constraints = {'H': 1e9, 'O': 1e9, 'C': 1e9, 'He': 1e9, 'N': 1e9,
                                                         'S': 8.193710271821442e+18})


def warm_up(cfg):
    '''
    Warms up a worker, so that the compiles happen before its first task
    One short isocalc run at the centre (1000 steps, with Atmodeller every min(n_atmodeller, 100) steps when it is
    on) compiles the Numba kernels and the JAX Atmodeller solve; then warm_fallback compiles the Atmodeller
    fallback paths. A failure of the fallback warm-up is only a warning. Measured on the M1 laptop, the whole
    warm-up takes 6-8 s per worker, 2.4-5 s with the compile caches.

    Inputs:
        - cfg: resolved config [dict]
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
    Sets up a worker process: the pool initializer, also usable in-process
    In order: line-buffered output (spawned workers), the single-thread environment, the compile caches, the
    worker globals, the warm-up, then barrier.wait. The caches and the warm-up apply to the isocalc backend only
    (parallel.cache, parallel.warmup). The parent waits at the same barrier until every worker is warm, so all
    workers compile at once, before the first task. A failure aborts the barrier, so that the parent stops waiting.

    Inputs:
        - cfg: resolved config [dict]
        - run_dir: run directory, for worker_errors/ and the solve-mode warm-start table; None for none [str]
        - barrier: the pool's start-up barrier (multiprocessing Barrier); None in-process
    '''
    global _CFG, _RUN_DIR
    try:
        if barrier is not None and hasattr(sys.stdout, 'reconfigure'): # a spawned worker: print lines as they
            sys.stdout.reconfigure(line_buffering = True)                 # come, not in blocks
        env.set_single_thread_env() # before jax is imported: XLA's default threading costs a third of the throughput
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
    except BaseException: # KeyboardInterrupt and SystemExit too: free the parent, then re-raise
        if barrier is not None:
            barrier.abort()
        raise
    if barrier is not None:
        try:
            barrier.wait(timeout = 600) # [s]
        except threading.BrokenBarrierError: # the parent gave up (it terminates the pool), or the pool is running
            # already: the parent aborts the barrier once the pool is up, so a worker that replaces a crashed one
            # does not wait
            pass
